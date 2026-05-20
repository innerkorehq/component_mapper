import asyncio
import json
import logging
import re
from pathlib import Path
from component_mapper.models import (
    ComponentSignature,
    CustomComponentDefinition,
    RegistrySource,
    InteractivityMode,
)

logger = logging.getLogger(__name__)


class CustomRegistry:
    def __init__(self):
        self._store: dict[str, ComponentSignature] = {}
        self._lock = asyncio.Lock()
        self._dirty = False

    async def load(self, path: str) -> None:
        """Load custom registry from disk. Silent if file missing."""
        p = Path(path)
        if not p.exists():
            logger.debug("No custom registry at %s", p)
            return
        try:
            raw = json.loads(p.read_text())
            async with self._lock:
                for name, data in raw.items():
                    try:
                        sig = ComponentSignature.model_validate(data)
                        self._store[name] = sig
                    except Exception as exc:
                        logger.warning(
                            "Skipping malformed custom component %s: %s", name, exc
                        )
            logger.info("Loaded %d custom components", len(self._store))
        except Exception as exc:
            logger.warning("Failed to load custom registry: %s", exc)

    async def persist(self, path: str) -> None:
        """Write custom registry to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with self._lock:
                data = {k: v.model_dump(mode="json") for k, v in self._store.items()}
            p.write_text(json.dumps(data, indent=2))
            self._dirty = False
            logger.debug("Persisted %d custom components", len(data))
        except Exception as exc:
            logger.warning("Failed to persist custom registry: %s", exc)

    def get_all(self) -> list[ComponentSignature]:
        return list(self._store.values())

    def get(self, name: str) -> ComponentSignature | None:
        return self._store.get(name)

    def register(self, defn: CustomComponentDefinition) -> ComponentSignature:
        """Convert CustomComponentDefinition -> ComponentSignature and add to store."""
        sig = self._to_signature(defn)
        self._store[defn.name] = sig
        self._dirty = True
        logger.debug("Registered custom component: %s", defn.name)
        return sig

    def _to_signature(self, defn: CustomComponentDefinition) -> ComponentSignature:
        # Count child tags from skeleton
        child_tag_counts: dict[str, int] = {}
        html_tags = {
            "div",
            "span",
            "p",
            "h1",
            "h2",
            "h3",
            "img",
            "button",
            "a",
            "ul",
            "li",
            "section",
            "article",
            "nav",
            "header",
            "footer",
        }
        for m in re.finditer(r"<([a-z][a-z0-9]*)", defn.dom_skeleton):
            tag = m.group(1)
            if tag in html_tags:
                child_tag_counts[tag] = child_tag_counts.get(tag, 0) + 1
        for tag in re.findall(r"\b([a-z][a-z0-9]*)\b", defn.dom_skeleton):
            if tag in html_tags:
                child_tag_counts[tag] = child_tag_counts.get(tag, 0) + 1

        # Measure nesting depth from skeleton brackets
        depth = 0
        max_depth = 0
        for ch in defn.dom_skeleton:
            if ch == "[":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == "]":
                depth -= 1

        return ComponentSignature(
            component_name=defn.name,
            registry_source=RegistrySource.CUSTOM,
            dom_skeleton=defn.dom_skeleton,
            root_element=defn.dom_skeleton.split(">")[0].split("[")[0].strip() or "div",
            required_children=[],
            optional_children=[],
            structural_class_tokens=defn.structural_class_tokens,
            typical_nesting_depth=max_depth,
            child_tag_counts=child_tag_counts,
            unique_tag_count=len(child_tag_counts),
            compatible_component_types=defn.compatible_component_types,
            interactivity=defn.interactivity,
            description=defn.description,
            props=defn.props,
            astro_import=defn.astro_import or f"@/components/custom/{defn.name}.astro",
            install_command=defn.install_command,
            requires_client_directive=defn.interactivity != InteractivityMode.STATIC,
        )
