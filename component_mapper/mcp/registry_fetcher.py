import asyncio
import json
import logging
import time
from pathlib import Path
import aiohttp
import aiofiles
from component_mapper.config import RegistryConfig
from component_mapper.models import RegistrySource

logger = logging.getLogger(__name__)


class RegistryFetcher:
    def __init__(self, config: RegistryConfig):
        self.config = config
        self._memory_cache: dict[str, dict] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_fetches)
        self._disk_cache_dir = Path(".cache/registry_http")
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, name: str, source: RegistrySource) -> str:
        return f"{source.value}:{name}"

    def _disk_cache_path(self, name: str, source: RegistrySource) -> Path:
        return self._disk_cache_dir / f"{source.value}_{name}.json"

    def _is_cache_fresh(self, cache_key: str) -> bool:
        ts = self._cache_timestamps.get(cache_key, 0)
        ttl_seconds = self.config.http_cache_ttl_hours * 3600
        return (time.time() - ts) < ttl_seconds

    def _base_url(self, source: RegistrySource) -> str:
        if source == RegistrySource.SHADCN:
            return self.config.shadcn_registry_base_url
        return (
            self.config.custom_registry_base_url or self.config.shadcn_registry_base_url
        )

    async def fetch_component(
        self,
        name: str,
        source: RegistrySource = RegistrySource.SHADCN,
    ) -> dict:
        cache_key = self._cache_key(name, source)

        # Memory cache
        if cache_key in self._memory_cache and self._is_cache_fresh(cache_key):
            return self._memory_cache[cache_key]

        # Disk cache
        disk_path = self._disk_cache_path(name, source)
        if disk_path.exists():
            stat_age = time.time() - disk_path.stat().st_mtime
            if stat_age < self.config.http_cache_ttl_hours * 3600:
                try:
                    async with aiofiles.open(disk_path, "r") as f:
                        data = json.loads(await f.read())
                    self._memory_cache[cache_key] = data
                    self._cache_timestamps[cache_key] = time.time()
                    return data
                except Exception:
                    pass

        # HTTP fetch
        async with self._semaphore:
            url = f"{self._base_url(source)}/{name}.json"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(
                            total=self.config.fetch_timeout_seconds,
                            connect=5,
                            sock_read=self.config.fetch_timeout_seconds,
                        ),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                        else:
                            logger.warning(
                                "Registry fetch %s returned %d", url, resp.status
                            )
                            data = {"name": name, "files": []}
            except Exception as exc:
                logger.warning("Registry fetch failed for %s: %s", name, exc)
                data = {"name": name, "files": []}

        self._memory_cache[cache_key] = data
        self._cache_timestamps[cache_key] = time.time()

        # Only persist successful responses — don't cache 404/error results so
        # a retry with a corrected URL or fixed server gets a fresh fetch.
        if data.get("files"):
            try:
                async with aiofiles.open(disk_path, "w") as f:
                    await f.write(json.dumps(data))
            except Exception:
                pass

        return data

    async def fetch_many(
        self,
        names: list[str],
        source: RegistrySource = RegistrySource.SHADCN,
    ) -> dict[str, dict]:
        """Concurrent fetch under semaphore. Returns name -> registry JSON."""
        tasks = [self.fetch_component(n, source) for n in names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            name: (r if isinstance(r, dict) else {"name": name, "files": []})
            for name, r in zip(names, results)
        }

    async def fetch_source_code(self, name: str) -> str:
        """Extract TypeScript source from registry JSON files array."""
        data = await self.fetch_component(name, RegistrySource.SHADCN)
        files = data.get("files", [])
        if files:
            return files[0].get("content", "")
        return ""

    async def fetch_from_external(
        self,
        url_template: str,
        component_name: str,
        registry_name: str,
    ) -> dict:
        """Fetch a component from an external registry using its URL template.

        url_template uses {name} as the placeholder e.g.
        "https://bundui.io/r/{name}.json" → "https://bundui.io/r/pagination.json"
        """
        cache_key = f"external:{registry_name}:{component_name}"

        if cache_key in self._memory_cache and self._is_cache_fresh(cache_key):
            return self._memory_cache[cache_key]

        disk_path = self._disk_cache_dir / f"ext_{registry_name}_{component_name}.json"
        if disk_path.exists():
            stat_age = time.time() - disk_path.stat().st_mtime
            if stat_age < self.config.http_cache_ttl_hours * 3600:
                try:
                    async with aiofiles.open(disk_path, "r") as f:
                        data = json.loads(await f.read())
                    self._memory_cache[cache_key] = data
                    self._cache_timestamps[cache_key] = time.time()
                    return data
                except Exception:
                    pass

        url = url_template.replace("{name}", component_name)
        async with self._semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(
                            total=self.config.fetch_timeout_seconds,
                            connect=5,
                            sock_read=self.config.fetch_timeout_seconds,
                        ),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            logger.debug(
                                "Fetched %s/%s from external registry",
                                registry_name, component_name,
                            )
                        else:
                            logger.warning(
                                "External registry %s returned %d for %s",
                                registry_name, resp.status, component_name,
                            )
                            data = {"name": component_name, "files": []}
            except Exception as exc:
                logger.warning(
                    "External registry fetch failed %s/%s: %s",
                    registry_name, component_name, exc,
                )
                data = {"name": component_name, "files": []}

        self._memory_cache[cache_key] = data
        self._cache_timestamps[cache_key] = time.time()
        if data.get("files"):
            try:
                async with aiofiles.open(disk_path, "w") as f:
                    await f.write(json.dumps(data))
            except Exception:
                pass

        return data

    async def fetch_all_external(
        self,
        external_registries: list,
    ) -> dict[str, dict]:
        """Fetch all components from all external registries concurrently.

        Returns dict keyed by "registry_name/component_name".
        """
        tasks = {}
        for reg in external_registries:
            if not reg.open_source:
                continue
            for comp in reg.components:
                key = f"{reg.name}/{comp}"
                tasks[key] = self.fetch_from_external(
                    reg.url_template, comp, reg.name
                )

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            key: (r if isinstance(r, dict) else {"name": key, "files": []})
            for key, r in zip(tasks.keys(), results)
        }

    # ── Registry discovery ────────────────────────────────────────────────────

    # Types that are not UI components — skip when building the index
    _SKIP_TYPES = frozenset({
        "registry:style", "registry:example", "registry:internal",
        "registry:theme", "registry:hook", "registry:lib",
    })

    async def fetch_registry_index(self, reg) -> list[str]:
        """
        Fetch registry.json for a single external registry and return the
        list of UI-compatible component names.

        Works by calling fetch_from_external with component_name="registry",
        which resolves to: url_template.replace("{name}", "registry").
        Reuses all existing disk-caching and timeout logic.
        """
        data = await self.fetch_from_external(
            reg.url_template, "registry", reg.name
        )
        if not data:
            return []

        items: list = (
            data.get("items", []) if isinstance(data, dict)
            else (data if isinstance(data, list) else [])
        )
        return [
            item["name"]
            for item in items
            if isinstance(item, dict)
            and item.get("name")
            and item.get("name") != "registry"
            and item.get("type", "registry:ui") not in self._SKIP_TYPES
        ]

    async def discover_all_external(
        self,
        external_registries: list,
        concurrency: int = 10,
    ) -> dict[str, list[str]]:
        """
        Fetch registry.json for every registry in *external_registries* and
        return a mapping of registry_name → [component_names].

        Uses a semaphore so at most *concurrency* registries are probed at once.
        Registries that 404, timeout, or return no UI components are omitted.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _probe(reg):
            async with sem:
                names = await self.fetch_registry_index(reg)
            return reg.name, names

        results = await asyncio.gather(
            *[_probe(r) for r in external_registries if r.open_source],
            return_exceptions=True,
        )
        return {
            name: comps
            for name, comps in (r for r in results if not isinstance(r, Exception))
            if comps
        }

    async def fetch_all_external_with_discovery(
        self,
        external_registries: list,
    ) -> dict[str, dict]:
        """
        Like fetch_all_external but first auto-discovers component names from
        each registry's registry.json, then fetches every discovered component.

        Hard-coded components in reg.components are always included.
        Registries without a registry.json fall back to reg.components only.

        Returns dict keyed by "registry_name/component_name".
        """
        # Step 1: discover all component names from registry.json files
        discovered = await self.discover_all_external(external_registries)

        # Step 2: merge with hard-coded components lists
        tasks: dict[str, asyncio.coroutine] = {}
        for reg in external_registries:
            if not reg.open_source:
                continue
            all_comps = list({*reg.components, *discovered.get(reg.name, [])})
            for comp in all_comps:
                key = f"{reg.name}/{comp}"
                tasks[key] = self.fetch_from_external(reg.url_template, comp, reg.name)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            key: (r if isinstance(r, dict) else {"name": key, "files": []})
            for key, r in zip(tasks.keys(), results)
        }
