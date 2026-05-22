#!/usr/bin/env -S uv run --project /Users/baneet/Desktop/claude/component_mapper
"""
Build the component-mapper signature index with full resumability.

Fetching strategy:
  Phase 0  Auto-discover components from all 142+ external registry.json files
           (parallel, 10 concurrent). Each registry exposes its component list
           at url_template.replace("{name}", "registry").
  Phase 1  Official Shadcn UI components via styles/{style}/registry.json
  Phase 2  All discovered external components (Phase 0 results + any
           hard-coded in OPEN_SOURCE_REGISTRIES[].components)

  All HTTP calls use `curl --max-time` so they always terminate.

Usage:
    python scripts/build_registry_cache.py            # build (or resume)
    python scripts/build_registry_cache.py --publish  # build + publish to GitHub
    python scripts/build_registry_cache.py --reset    # discard checkpoint, start fresh
    python scripts/build_registry_cache.py --reset-http-cache
    python scripts/build_registry_cache.py --no-discovery   # skip Phase 0, use only hard-coded lists
    python scripts/build_registry_cache.py --style default-v4

Requires: pip install -e "."   (component-mapper)
          curl  (always available on macOS / Linux)
Publish:  gh auth login
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DEFAULT    = pathlib.Path(__file__).parent.parent / "dist-cache"
SHADCN_STYLE   = "new-york-v4"
RELEASE_TAG    = "registry-cache-latest"
RELEASE_TITLE  = "Registry cache (pre-built)"
RELEASE_NOTES  = (
    "Pre-built component-mapper signature index.\n"
    "Downloaded by the sitesudharo Dockerfile during `docker build`.\n\n"
    "Regenerate: `python scripts/build_registry_cache.py --publish`"
)

log = logging.getLogger(__name__)


# ── HTTP via curl ─────────────────────────────────────────────────────────────
# subprocess.run(timeout=N) sends SIGKILL to curl after N seconds — the only
# Python mechanism that is truly guaranteed to terminate a blocking HTTP call.

def _curl(url: str, timeout: int = 10) -> dict:
    """
    Fetch *url* with curl, writing output to a temp file (avoids pipe deadlock
    on macOS where capture_output=True can hang during kill/drain cleanup).
    Returns parsed JSON dict or {} on any error.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(tmp_fd)
    try:
        subprocess.run(
            [
                "curl", "-sf",
                "--max-time", str(timeout),
                "--connect-timeout", "5",
                "--compressed",
                "-H", "Accept: application/json",
                "-o", tmp_path,
                url,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout + 3,
        )
        raw = pathlib.Path(tmp_path).read_text(encoding="utf-8").strip()
        if raw:
            return json.loads(raw)
    except subprocess.TimeoutExpired:
        log.warning("curl timeout: %s", url)
    except json.JSONDecodeError:
        log.warning("bad JSON from: %s", url)
    except FileNotFoundError:
        sys.exit("Error: `curl` not found — install curl and retry")
    except Exception as exc:
        log.warning("curl error %s: %s", url, exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return {}


# ── Disk cache ────────────────────────────────────────────────────────────────

def _cached_fetch(url: str, cache_dir: pathlib.Path, key: str,
                  timeout: int = 10, ttl: int = 86400) -> dict:
    """Fetch with a 24-hour disk cache. Only caches successful responses."""
    f = cache_dir / f"{key}.json"
    if f.exists() and (time.time() - f.stat().st_mtime) < ttl:
        try:
            data = json.loads(f.read_text())
            if data:
                return data
        except Exception:
            pass

    data = _curl(url, timeout)

    if data:
        try:
            f.write_text(json.dumps(data))
        except Exception:
            pass
    return data


# ── Checkpoint ────────────────────────────────────────────────────────────────

class Checkpoint:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.done:       set[str]        = set()
        self.components: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            self.done       = set(raw.get("done", []))
            self.components = raw.get("components", {})
            if self.done:
                print(f"Resuming — {len(self.done)} done, {len(self.components)} in index")
        except Exception as exc:
            print(f"Warning: checkpoint unreadable ({exc}), starting fresh")

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done": sorted(self.done), "components": self.components}))
        tmp.replace(self.path)

    def record(self, name: str, sig_dict: dict) -> None:
        self.components[name] = sig_dict
        self.done.add(name)
        self.save()

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


# ── Core builder ─────────────────────────────────────────────────────────────

# Types to skip when parsing registry.json component lists
_SKIP_TYPES = frozenset({
    "registry:style", "registry:example", "registry:internal",
    "registry:theme", "registry:hook", "registry:lib",
})


def _discover_registry(reg, cache_dir: pathlib.Path) -> tuple[str, list[str]]:
    """Fetch registry.json for one registry and return (name, [component_names])."""
    url  = reg.url_template.replace("{name}", "registry")
    data = _cached_fetch(url, cache_dir, f"idx_{reg.name}", timeout=8, ttl=86400 * 7)
    if not data:
        return reg.name, []
    items: list = (
        data.get("items", []) if isinstance(data, dict)
        else (data if isinstance(data, list) else [])
    )
    names = [
        item["name"] for item in items
        if isinstance(item, dict)
        and item.get("name")
        and item.get("name") != "registry"
        and item.get("type", "registry:ui") not in _SKIP_TYPES
    ]
    return reg.name, names


def build(out: pathlib.Path, style: str, reset: bool, discovery: bool = True) -> int:
    from component_mapper.config import OPEN_SOURCE_REGISTRIES
    from component_mapper.registry.signature_index import (
        COMPONENT_HINTS,
        _build_signature_from_hints,
        _build_signature_from_parsed,
    )
    from component_mapper.utils.source_parser import parse_source

    out.mkdir(parents=True, exist_ok=True)
    cache_dir  = out / "_http_cache"
    cache_dir.mkdir(exist_ok=True)
    checkpoint = Checkpoint(out / ".checkpoint.json")

    if reset:
        checkpoint.done.clear()
        checkpoint.components.clear()
        print("Checkpoint cleared — starting fresh")

    base_url = f"https://ui.shadcn.com/r/styles/{style}"

    def _sig(name: str, reg_data: dict) -> dict:
        files = reg_data.get("files", [])
        src   = files[0].get("content", "") if files else ""
        if src:
            return _build_signature_from_parsed(name, parse_source(src)).model_dump(mode="json")
        hints = COMPONENT_HINTS.get(name, {"skeleton": "div", "root": "div", "compatible": [], "classes": []})
        return _build_signature_from_hints(name, hints).model_dump(mode="json")

    interrupted = False

    # ── Phase 0: Discover all external registry component lists ──────────────
    discovered: dict[str, list[str]] = {}   # registry_name → [component_names]
    if discovery:
        regs = [r for r in OPEN_SOURCE_REGISTRIES if r.open_source]
        print(f"\nPhase 0 — Discovering components from {len(regs)} registry.json files ...")
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_discover_registry, r, cache_dir): r for r in regs}
            for fut in as_completed(futures):
                reg = futures[fut]
                try:
                    rname, comps = fut.result()
                    if comps:
                        discovered[rname] = comps
                except Exception:
                    pass
        total_comps = sum(len(v) for v in discovered.values())
        print(f"  {len(discovered)}/{len(regs)} registries reachable, "
              f"{total_comps} components discovered")
    else:
        print("\nPhase 0 — Discovery skipped (--no-discovery)")

    # ── Phase 1: Official Shadcn UI components from registry.json ────────────
    print(f"\nPhase 1 — Fetching registry index from {base_url}/registry.json ...")
    registry = _cached_fetch(f"{base_url}/registry.json", cache_dir, f"registry_{style}", timeout=15)
    ui_items  = [i for i in registry.get("items", []) if i.get("type") == "registry:ui"]

    if not ui_items:
        print("  Warning: registry.json empty or unreachable — falling back to hardcoded list")
        from component_mapper.registry.signature_index import KNOWN_SHADCN_COMPONENTS
        ui_items = [{"name": n} for n in KNOWN_SHADCN_COMPONENTS]

    names     = [i["name"] for i in ui_items]
    todo      = [n for n in names if n not in checkpoint.done]
    print(f"  {len(names)} ui components in registry, {len(names)-len(todo)} already done")

    for i, name in enumerate(todo, len(names) - len(todo) + 1):
        print(f"  [{i:>3}/{len(names)}] {name} ...", end="", flush=True)
        try:
            data = _cached_fetch(f"{base_url}/{name}.json", cache_dir, f"shadcn_{name}")
            checkpoint.record(name, _sig(name, data))
            label = "✓" if data.get("files") else "✓ (hint)"
            print(f"\r  [{i:>3}/{len(names)}] {name:<35} {label}")
        except KeyboardInterrupt:
            print(f"\nInterrupted at {name}")
            interrupted = True
            break
        except Exception as exc:
            print(f"\r  [{i:>3}/{len(names)}] {name:<35} WARN: {exc}")
            checkpoint.record(name, _sig(name, {}))

    # ── Phase 2: All external registry components (discovered + hard-coded) ──
    # Merge hard-coded components with discovered ones
    ext_items = []
    for r in OPEN_SOURCE_REGISTRIES:
        if not r.open_source:
            continue
        all_comps = sorted({*r.components, *discovered.get(r.name, [])})
        for c in all_comps:
            ext_items.append((f"{r.name}/{c}", r.url_template, c, r.name))

    ext_todo  = [(k, ut, c, r) for k, ut, c, r in ext_items if k not in checkpoint.done]

    if not interrupted and ext_todo:
        total_ext = len(ext_items)
        done_ext  = total_ext - len(ext_todo)
        print(f"\nPhase 2 — External registry components  ({done_ext}/{total_ext} done)")

        for i, (compound, url_tpl, comp, reg) in enumerate(ext_todo, done_ext + 1):
            print(f"  [{i:>3}/{total_ext}] {compound} ...", end="", flush=True)
            try:
                url  = url_tpl.replace("{name}", comp)
                data = _cached_fetch(url, cache_dir, f"ext_{reg}_{comp}")
                sig  = _sig(compound, data)
                checkpoint.record(compound, sig)
                label = "✓" if data.get("files") else "✓ (hint)"
                print(f"\r  [{i:>3}/{total_ext}] {compound:<45} {label}")
            except KeyboardInterrupt:
                print(f"\nInterrupted at {compound}")
                interrupted = True
                break
            except Exception as exc:
                print(f"\r  [{i:>3}/{total_ext}] {compound:<45} WARN: {exc}")
                checkpoint.record(compound, _sig(compound, {}))

    # ── Write final signature_index.json ─────────────────────────────────────
    n        = len(checkpoint.components)
    sig_path = out / "signature_index.json"
    tmp      = sig_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"built_at": time.time(), "components": checkpoint.components}, indent=2))
    tmp.replace(sig_path)
    size_kb = sig_path.stat().st_size // 1024

    print(f"\n{'Partial' if interrupted else 'Complete'} index: {n} components  {size_kb} KB  →  {sig_path}")

    if interrupted:
        print(f"Checkpoint: {checkpoint.path}")
        print("Resume:     python scripts/build_registry_cache.py")
    else:
        checkpoint.delete()
        print("Checkpoint cleared.")

    return n


# ── GitHub publish ────────────────────────────────────────────────────────────

def _gh_path() -> str | None:
    """Return path to gh CLI, or None if not installed."""
    for candidate in (
        "gh",
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        "/usr/bin/gh",
    ):
        r = subprocess.run([candidate, "--version"], capture_output=True)
        if r.returncode == 0:
            return candidate
    return None


def _publish_via_gh(gh: str, files: list[str]) -> None:
    subprocess.run([gh, "release", "delete", RELEASE_TAG, "--yes"], capture_output=True)
    subprocess.run(
        [gh, "release", "create", RELEASE_TAG,
         "--title", RELEASE_TITLE, "--notes", RELEASE_NOTES, *files],
        check=True,
    )


def _publish_via_api(token: str, files: list[str]) -> None:
    """Publish via GitHub REST API using curl (no gh CLI needed)."""
    import json as _json

    # Parse owner/repo from git remote
    r = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    remote = r.stdout.strip()
    # Handle both https and ssh formats
    import re
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote)
    if not m:
        sys.exit(f"Cannot parse GitHub repo from remote: {remote!r}\nSet GITHUB_REPO=owner/repo and retry.")
    repo = m.group(1)

    api = f"https://api.github.com/repos/{repo}"
    auth = f"Authorization: Bearer {token}"
    ct_json = "Content-Type: application/json"

    # Delete existing release
    r = subprocess.run(
        ["curl", "-sf", "-H", auth, f"{api}/releases/tags/{RELEASE_TAG}"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        rel_id = _json.loads(r.stdout).get("id")
        if rel_id:
            subprocess.run(["curl", "-sf", "-X", "DELETE", "-H", auth,
                            f"{api}/releases/{rel_id}"], capture_output=True)
    # Delete tag
    subprocess.run(["curl", "-sf", "-X", "DELETE", "-H", auth,
                    f"{api}/git/refs/tags/{RELEASE_TAG}"], capture_output=True)

    # Create release
    body = _json.dumps({"tag_name": RELEASE_TAG, "name": RELEASE_TITLE,
                        "body": RELEASE_NOTES, "draft": False})
    r = subprocess.run(
        ["curl", "-sf", "-X", "POST", "-H", auth, "-H", ct_json, "-d", body,
         f"{api}/releases"],
        capture_output=True, text=True, check=True,
    )
    rel = _json.loads(r.stdout)
    upload_url = rel["upload_url"].split("{")[0]  # strip {?name,label}
    rel_id = rel["id"]

    # Upload assets
    for fpath in files:
        name = pathlib.Path(fpath).name
        size = pathlib.Path(fpath).stat().st_size
        print(f"  uploading {name} ({size // 1024} KB) ...", end="", flush=True)
        subprocess.run(
            ["curl", "-sf", "-X", "POST",
             "-H", auth,
             "-H", "Content-Type: application/json",
             "--data-binary", f"@{fpath}",
             f"{upload_url}?name={name}"],
            capture_output=True, check=True,
        )
        print(" ✓")


def publish(out: pathlib.Path) -> None:
    sig = out / "signature_index.json"
    if not sig.exists():
        sys.exit(f"Error: {sig} not found — build first")

    files = [str(sig)]
    custom = out / "custom_registry.json"
    if custom.exists():
        files.append(str(custom))

    print(f"\nPublishing  tag={RELEASE_TAG}  ({len(files)} file(s)) ...")

    gh = _gh_path()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if gh:
        _publish_via_gh(gh, files)
    elif token:
        _publish_via_api(token, files)
    else:
        print("\nNeither `gh` CLI nor GITHUB_TOKEN found.")
        print("Install gh:       brew install gh && gh auth login")
        print("Or set token:     export GITHUB_TOKEN=ghp_...")
        sys.exit(1)

    print("Published.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--publish",          action="store_true", help="Upload to GitHub Releases")
    p.add_argument("--reset",            action="store_true", help="Discard checkpoint, rebuild")
    p.add_argument("--reset-http-cache", action="store_true", help="Delete HTTP disk cache")
    p.add_argument("--no-discovery",    action="store_true", help="Skip Phase 0 registry.json discovery; use only hard-coded component lists")
    p.add_argument("--style", default=SHADCN_STYLE,           help=f"Shadcn style (default: {SHADCN_STYLE})")
    p.add_argument("--out",   type=pathlib.Path, default=OUT_DEFAULT)
    args = p.parse_args()

    if args.reset_http_cache:
        import shutil
        hc = args.out / "_http_cache"
        if hc.exists():
            shutil.rmtree(hc)
            print(f"HTTP cache cleared: {hc}")

    sig_path = args.out / "signature_index.json"
    already_built = sig_path.exists() and not args.reset

    if already_built and args.publish:
        # Index is ready — skip rebuild, go straight to publish
        print(f"Using existing index: {sig_path}  ({sig_path.stat().st_size // 1024} KB)")
    else:
        try:
            build(args.out, args.style, args.reset, discovery=not args.no_discovery)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(1)

    if args.publish:
        publish(args.out)
    else:
        print("\nTo publish:  .venv/bin/python scripts/build_registry_cache.py --publish")


if __name__ == "__main__":
    main()
