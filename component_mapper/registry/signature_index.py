import json
import logging
import time
from pathlib import Path
from typing import Any

from component_mapper.config import MapperSettings
from component_mapper.models import (
    ComponentSignature,
    RankedCandidate,
    RegistrySource,
    InteractivityMode,
    CustomComponentDefinition,
)
from component_mapper.utils.source_parser import parse_source
from component_mapper.utils.similarity import (
    skeleton_similarity,
    jaccard_similarity,
    composite_score,
    rank_candidates,
)
from component_mapper.mcp.official_client import OfficialMCPClient
from component_mapper.mcp.registry_fetcher import RegistryFetcher
from segment_classifier.models import ComponentType
from segment_classifier.utils.html_normalizer import NormalizedSegment

logger = logging.getLogger(__name__)

# Static type map: ComponentType → list of Shadcn component names
TYPE_MAP: dict[ComponentType, list[str]] = {
    ComponentType.LAYOUT_HEADER: ["navigation-menu"],
    ComponentType.LAYOUT_FOOTER: [],
    ComponentType.LAYOUT_NAV: ["navigation-menu", "menubar"],
    ComponentType.LAYOUT_SIDEBAR: ["sidebar"],
    ComponentType.LAYOUT_BREADCRUMB: ["breadcrumb"],
    ComponentType.COLLECTION_PRODUCT_CARD: ["card"],
    ComponentType.COLLECTION_BLOG_CARD: ["card"],
    ComponentType.COLLECTION_NEWS_ITEM: ["card"],
    ComponentType.COLLECTION_PRODUCT_LIST: ["table"],
    ComponentType.COLLECTION_BLOG_LIST: ["card"],
    ComponentType.COLLECTION_NEWS_LIST: ["card"],
    ComponentType.SECTION_HERO: [],
    ComponentType.SECTION_FEATURE_GRID: ["card"],
    ComponentType.SECTION_TESTIMONIAL: ["card"],
    ComponentType.SECTION_CTA: ["card", "button"],
    ComponentType.SECTION_FAQ: ["accordion"],
    ComponentType.SECTION_PRICING: ["card"],
    ComponentType.UI_FORM: ["form", "input", "select"],
    ComponentType.UI_MODAL: ["dialog", "sheet"],
    ComponentType.UI_TABLE: ["table"],
    ComponentType.UI_CAROUSEL: ["carousel"],
    ComponentType.UI_PAGINATION: ["pagination", "bundui/pagination", "hextaui/pagination"],
    ComponentType.UI_SEARCH: ["command"],
    ComponentType.CONTENT_ARTICLE: [],
    ComponentType.CONTENT_RICH_TEXT: [],
    ComponentType.CONTENT_MEDIA: ["aspect-ratio"],
    ComponentType.UNKNOWN: [],
}

# Known Shadcn components (fallback if MCP unavailable)
KNOWN_SHADCN_COMPONENTS = [
    "accordion",
    "alert",
    "alert-dialog",
    "aspect-ratio",
    "avatar",
    "badge",
    "breadcrumb",
    "button",
    "calendar",
    "card",
    "carousel",
    "checkbox",
    "collapsible",
    "command",
    "context-menu",
    "data-table",
    "date-picker",
    "dialog",
    "drawer",
    "dropdown-menu",
    "form",
    "hover-card",
    "input",
    "input-otp",
    "label",
    "menubar",
    "navigation-menu",
    "pagination",
    "popover",
    "progress",
    "radio-group",
    "resizable",
    "scroll-area",
    "select",
    "separator",
    "sheet",
    "sidebar",
    "skeleton",
    "slider",
    "sonner",
    "switch",
    "table",
    "tabs",
    "textarea",
    "toast",
    "toggle",
    "toggle-group",
    "tooltip",
]

# Component type hints for building signatures without source code
COMPONENT_HINTS: dict[str, dict[str, Any]] = {
    "card": {
        "compatible": [
            ComponentType.COLLECTION_PRODUCT_CARD,
            ComponentType.COLLECTION_BLOG_CARD,
            ComponentType.SECTION_FEATURE_GRID,
            ComponentType.SECTION_TESTIMONIAL,
            ComponentType.SECTION_CTA,
            ComponentType.SECTION_PRICING,
        ],
        "skeleton": "div>[div>h3+p, div, div>button]",
        "root": "div",
        "classes": ["card", "content", "header", "footer"],
    },
    "navigation-menu": {
        "compatible": [ComponentType.LAYOUT_HEADER, ComponentType.LAYOUT_NAV],
        "skeleton": "nav>[ul>[li>a]]",
        "root": "nav",
        "classes": ["nav", "menu"],
    },
    "accordion": {
        "compatible": [ComponentType.SECTION_FAQ],
        "skeleton": "div>[div>button+div>p]",
        "root": "div",
        "classes": ["accordion", "item"],
    },
    "dialog": {
        "compatible": [ComponentType.UI_MODAL],
        "skeleton": "div>[div>h2+p+div>button]",
        "root": "div",
        "classes": ["modal", "dialog"],
    },
    "table": {
        "compatible": [ComponentType.UI_TABLE, ComponentType.COLLECTION_PRODUCT_LIST],
        "skeleton": "table>[thead>tr>th, tbody>tr>td]",
        "root": "table",
        "classes": ["table"],
    },
    "carousel": {
        "compatible": [ComponentType.UI_CAROUSEL],
        "skeleton": "div>[div>[div>img], button, button]",
        "root": "div",
        "classes": ["carousel"],
    },
    "form": {
        "compatible": [ComponentType.UI_FORM],
        "skeleton": "form>[div>label+input, button]",
        "root": "form",
        "classes": ["form"],
    },
    "command": {
        "compatible": [ComponentType.UI_SEARCH],
        "skeleton": "div>[input, div>[div>span]]",
        "root": "div",
        "classes": ["search", "command"],
    },
    "pagination": {
        "compatible": [ComponentType.UI_PAGINATION],
        "skeleton": "nav>[ul>[li>a]+li>a+[li>a]]",
        "root": "nav",
        "classes": ["pagination"],
        "install": "npx shadcn@latest add pagination",
    },
    # ── External open-source registries (verified from ui.shadcn.com/docs/directory) ──
    "bundui/pagination": {
        "compatible": [ComponentType.UI_PAGINATION],
        # Same nav>ul>li>a structure; bundui wraps the official pagination with
        # its own styling tokens — skeleton is identical to the official component.
        "skeleton": "nav>[ul>[li>a+li>a+li>a+li>span+li>a]]",
        "root": "nav",
        "classes": ["pagination"],
        "install": "npx shadcn@latest add @bundui/pagination",
        "registry": "bundui",
        "namespace": "@bundui",
    },
    "hextaui/pagination": {
        "compatible": [ComponentType.UI_PAGINATION],
        # HextaUI pagination uses data-slot attributes and buttonVariants;
        # structure matches official but adds touch-manipulation and tabular-nums.
        "skeleton": "nav>[ul>[li>a+li>a+li>a+li>span+li>a]]",
        "root": "nav",
        "classes": ["pagination"],
        "install": "npx shadcn@latest add @hextaui/pagination",
        "registry": "hextaui",
        "namespace": "@hextaui",
    },
    "breadcrumb": {
        "compatible": [ComponentType.LAYOUT_BREADCRUMB],
        "skeleton": "nav>[ol>[li>a+li>a]]",
        "root": "nav",
        "classes": ["breadcrumb"],
    },
    "sidebar": {
        "compatible": [ComponentType.LAYOUT_SIDEBAR],
        "skeleton": "aside>[div>[nav>ul>li>a]]",
        "root": "aside",
        "classes": ["sidebar", "nav"],
    },
    "sheet": {
        "compatible": [ComponentType.UI_MODAL],
        "skeleton": "div>[div>h2+p+div]",
        "root": "div",
        "classes": ["modal", "sheet"],
    },
    "aspect-ratio": {
        "compatible": [ComponentType.CONTENT_MEDIA],
        "skeleton": "div>[img]",
        "root": "div",
        "classes": ["media"],
    },
    "menubar": {
        "compatible": [ComponentType.LAYOUT_NAV],
        "skeleton": "div>[div>[button]]",
        "root": "div",
        "classes": ["menu", "nav"],
    },
    "select": {
        "compatible": [ComponentType.UI_FORM],
        "skeleton": "div>[button>span+span, div>[div>span]]",
        "root": "div",
        "classes": ["form", "select"],
    },
    "input": {
        "compatible": [ComponentType.UI_FORM],
        "skeleton": "input",
        "root": "input",
        "classes": ["form", "input"],
    },
    "button": {
        "compatible": [ComponentType.SECTION_CTA],
        "skeleton": "button>span",
        "root": "button",
        "classes": [],
    },
    "badge": {
        "compatible": [],
        "skeleton": "span",
        "root": "span",
        "classes": ["badge"],
    },
    "avatar": {
        "compatible": [],
        "skeleton": "span>[img, span]",
        "root": "span",
        "classes": [],
    },
    "tabs": {
        "compatible": [],
        "skeleton": "div>[div>[button], div>[div]]",
        "root": "div",
        "classes": [],
    },
    "separator": {"compatible": [], "skeleton": "div", "root": "div", "classes": []},
    "skeleton": {"compatible": [], "skeleton": "div", "root": "div", "classes": []},
    "progress": {
        "compatible": [],
        "skeleton": "div>[div]",
        "root": "div",
        "classes": [],
    },
    "slider": {
        "compatible": [],
        "skeleton": "div>[span+span+span]",
        "root": "div",
        "classes": [],
    },
    "checkbox": {
        "compatible": [],
        "skeleton": "button",
        "root": "button",
        "classes": [],
    },
    "radio-group": {
        "compatible": [],
        "skeleton": "div>[div>[button+label]]",
        "root": "div",
        "classes": ["form"],
    },
    "switch": {
        "compatible": [],
        "skeleton": "button>span",
        "root": "button",
        "classes": [],
    },
    "toggle": {"compatible": [], "skeleton": "button", "root": "button", "classes": []},
    "tooltip": {
        "compatible": [],
        "skeleton": "div>[button, div>p]",
        "root": "div",
        "classes": [],
    },
    "popover": {
        "compatible": [],
        "skeleton": "div>[button, div>div]",
        "root": "div",
        "classes": [],
    },
    "hover-card": {
        "compatible": [],
        "skeleton": "div>[span, div>div]",
        "root": "div",
        "classes": [],
    },
    "alert": {
        "compatible": [],
        "skeleton": "div>[span+h5+p]",
        "root": "div",
        "classes": [],
    },
    "alert-dialog": {
        "compatible": [ComponentType.UI_MODAL],
        "skeleton": "div>[div>[h2+p+div>[button+button]]]",
        "root": "div",
        "classes": ["modal"],
    },
    "calendar": {
        "compatible": [],
        "skeleton": "div>[div>[button+div+button], table>[thead>tr>th, tbody>tr>td]]",
        "root": "div",
        "classes": [],
    },
    "collapsible": {
        "compatible": [],
        "skeleton": "div>[div>button, div]",
        "root": "div",
        "classes": [],
    },
    "context-menu": {
        "compatible": [],
        "skeleton": "div>[div>span]",
        "root": "div",
        "classes": ["menu"],
    },
    "dropdown-menu": {
        "compatible": [],
        "skeleton": "div>[button, div>[div>span]]",
        "root": "div",
        "classes": ["menu"],
    },
    "label": {"compatible": [], "skeleton": "label", "root": "label", "classes": []},
    "scroll-area": {
        "compatible": [],
        "skeleton": "div>[div, div>div]",
        "root": "div",
        "classes": [],
    },
    "sonner": {
        "compatible": [],
        "skeleton": "section>[ol>[li]]",
        "root": "section",
        "classes": [],
    },
    "resizable": {
        "compatible": [],
        "skeleton": "div>[div+div+div]",
        "root": "div",
        "classes": [],
    },
    "drawer": {
        "compatible": [ComponentType.UI_MODAL],
        "skeleton": "div>[div>[div>h2+p+div]]",
        "root": "div",
        "classes": ["modal"],
    },
    "input-otp": {
        "compatible": [],
        "skeleton": "div>[div>[div>input]]",
        "root": "div",
        "classes": ["form"],
    },
    "toggle-group": {
        "compatible": [],
        "skeleton": "div>[button]",
        "root": "div",
        "classes": [],
    },
    "toast": {
        "compatible": [],
        "skeleton": "div>[div>[div>p]]",
        "root": "div",
        "classes": [],
    },
    "date-picker": {
        "compatible": [],
        "skeleton": "div>[button>span+span, div]",
        "root": "div",
        "classes": ["form"],
    },
    "data-table": {
        "compatible": [ComponentType.COLLECTION_PRODUCT_LIST, ComponentType.UI_TABLE],
        "skeleton": "div>[div>input, div>[table>[thead>tr>th, tbody>tr>td]], div>[button]]",
        "root": "div",
        "classes": ["table"],
    },
}


def _build_signature_from_hints(name: str, hints: dict) -> ComponentSignature:
    """Build a ComponentSignature from known hints (no source parsing needed)."""
    classes = hints.get("classes", [])
    # For external registry components (name contains "/"), use the registry namespace
    # for the install command and derive a clean Astro import from the base component name.
    base_name = name.split("/")[-1]  # "bundui/pagination" → "pagination"
    is_external = "/" in name
    astro_import = f"@/components/ui/{base_name}"
    install_cmd = hints.get("install") or (
        f"npx shadcn@latest add {hints.get('namespace', '')}/{base_name}"
        if is_external
        else f"npx shadcn@latest add {name}"
    )
    description = hints.get("description") or (
        f"{hints.get('registry', 'External').capitalize()} {base_name} component"
        if is_external
        else f"Shadcn {name} component"
    )
    return ComponentSignature(
        component_name=name,
        registry_source=RegistrySource.SHADCN,
        dom_skeleton=hints.get("skeleton", "div"),
        root_element=hints.get("root", "div"),
        required_children=[],
        optional_children=[],
        structural_class_tokens=classes,
        typical_nesting_depth=hints.get("skeleton", "").count(">"),
        child_tag_counts={},
        unique_tag_count=0,
        compatible_component_types=hints.get("compatible", []),
        interactivity=InteractivityMode.STATIC,
        description=description,
        props=[],
        astro_import=astro_import,
        install_command=install_cmd,
        requires_client_directive=False,
    )


def _build_signature_from_parsed(name: str, parsed) -> ComponentSignature:
    """Build ComponentSignature from ParsedSource."""
    hints = COMPONENT_HINTS.get(name, {})
    return ComponentSignature(
        component_name=name,
        registry_source=RegistrySource.SHADCN,
        dom_skeleton=parsed.dom_skeleton or hints.get("skeleton", "div"),
        root_element=parsed.root_element or hints.get("root", "div"),
        required_children=parsed.required_children,
        optional_children=parsed.optional_children,
        structural_class_tokens=parsed.structural_class_tokens
        or hints.get("classes", []),
        typical_nesting_depth=parsed.typical_nesting_depth,
        child_tag_counts=parsed.child_tag_counts,
        unique_tag_count=len(parsed.child_tag_counts),
        compatible_component_types=hints.get("compatible", []),
        interactivity=parsed.interactivity,
        description=f"Shadcn {name} component",
        props=parsed.props,
        astro_import=f"@/components/ui/{name}",
        install_command=f"npx shadcn@latest add {name}",
        requires_client_directive=parsed.interactivity != InteractivityMode.STATIC,
    )


class SignatureIndex:
    def __init__(
        self,
        settings: MapperSettings,
        fetcher: RegistryFetcher,
        mcp_client: OfficialMCPClient,
    ):
        self._index: dict[str, ComponentSignature] = {}
        self._settings = settings
        self._fetcher = fetcher
        self._mcp = mcp_client
        self._built = False

    async def build(self) -> None:
        """Build full index. Load from cache if fresh, else rebuild."""
        cache_path = Path(self._settings.signature_index.index_cache_path)
        if await self._load_from_cache(cache_path):
            logger.info(
                "Loaded signature index from cache (%d components)", len(self._index)
            )
            return
        logger.info("Rebuilding signature index — this may take 5-15s on first run")
        await self._rebuild()
        await self._persist(cache_path)
        self._built = True

    async def _rebuild(self) -> None:
        """Fetch all components, parse sources, populate index."""
        # ── Official Shadcn components ────────────────────────────────────────
        mcp_names = await self._mcp.list_components()
        names = mcp_names if mcp_names else KNOWN_SHADCN_COMPONENTS
        logger.info("Building signatures for %d Shadcn components", len(names))

        source_map = await self._fetcher.fetch_many(names, RegistrySource.SHADCN)

        for name in names:
            try:
                reg_data = source_map.get(name, {})
                files = reg_data.get("files", [])
                source_code = files[0].get("content", "") if files else ""

                if source_code:
                    parsed = parse_source(source_code)
                    sig = _build_signature_from_parsed(name, parsed)
                else:
                    hints = COMPONENT_HINTS.get(
                        name,
                        {"skeleton": "div", "root": "div", "compatible": [], "classes": []},
                    )
                    sig = _build_signature_from_hints(name, hints)

                self._index[name] = sig
            except Exception as exc:
                logger.warning("Failed to build signature for %s: %s", name, exc)
                hints = COMPONENT_HINTS.get(
                    name,
                    {"skeleton": "div", "root": "div", "compatible": [], "classes": []},
                )
                self._index[name] = _build_signature_from_hints(name, hints)

        # ── External open-source registries ───────────────────────────────────
        ext_regs = self._settings.registry.external_registries
        if ext_regs:
            logger.info(
                "Fetching components from %d external registries", len(ext_regs)
            )
            ext_source_map = await self._fetcher.fetch_all_external(ext_regs)

            for compound_name, reg_data in ext_source_map.items():
                # compound_name = "bundui/pagination", "hextaui/pagination", etc.
                try:
                    files = reg_data.get("files", [])
                    source_code = files[0].get("content", "") if files else ""

                    if source_code:
                        parsed = parse_source(source_code)
                        sig = _build_signature_from_parsed(compound_name, parsed)
                    else:
                        hints = COMPONENT_HINTS.get(
                            compound_name,
                            {"skeleton": "div", "root": "div", "compatible": [], "classes": []},
                        )
                        sig = _build_signature_from_hints(compound_name, hints)

                    self._index[compound_name] = sig
                    logger.debug("Indexed external component: %s", compound_name)
                except Exception as exc:
                    logger.warning(
                        "Failed to build signature for external %s: %s",
                        compound_name, exc,
                    )
                    hints = COMPONENT_HINTS.get(
                        compound_name,
                        {"skeleton": "div", "root": "div", "compatible": [], "classes": []},
                    )
                    self._index[compound_name] = _build_signature_from_hints(
                        compound_name, hints
                    )

        logger.info("Signature index built: %d components total", len(self._index))

    async def _persist(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        data = {
            "built_at": time.time(),
            "components": {
                k: v.model_dump(mode="json") for k, v in self._index.items()
            },
        }
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(cache_path)
        logger.debug("Persisted signature index to %s", cache_path)

    async def _load_from_cache(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        try:
            raw = json.loads(cache_path.read_text())
            built_at = raw.get("built_at", 0)
            ttl = self._settings.registry.http_cache_ttl_hours * 3600
            if time.time() - built_at > ttl:
                logger.debug("Signature cache expired")
                return False
            components = raw.get("components", {})
            for name, data in components.items():
                try:
                    self._index[name] = ComponentSignature.model_validate(data)
                except Exception:
                    pass
            return bool(self._index)
        except Exception as exc:
            logger.warning("Failed to load signature cache: %s", exc)
            return False

    def get_candidates(
        self,
        component_type: ComponentType,
        normalized: NormalizedSegment,
        fingerprint_hash: str,
    ) -> list[RankedCandidate]:
        """Get ranked candidates for a segment."""
        cfg = self._settings.signature_index

        # Get candidate names from type map
        if component_type == ComponentType.UNKNOWN:
            candidate_names = list(self._index.keys())
        else:
            candidate_names = list(TYPE_MAP.get(component_type, []))
            # Fallback: if no type map entries, use all
            if not candidate_names:
                candidate_names = list(self._index.keys())

        seg_skeleton = normalized.skeleton or ""
        seg_class_tokens = set(normalized.class_tokens or [])

        candidates: list[RankedCandidate] = []
        for name in candidate_names:
            sig = self._index.get(name)
            if sig is None:
                continue

            struct_score = skeleton_similarity(seg_skeleton, sig.dom_skeleton)
            class_score = jaccard_similarity(
                seg_class_tokens, set(sig.structural_class_tokens)
            )
            type_score = (
                1.0 if component_type in sig.compatible_component_types else 0.3
            )
            comp = composite_score(struct_score, class_score, type_score)

            candidates.append(
                RankedCandidate(
                    component_name=name,
                    registry_source=sig.registry_source,
                    signature=sig,
                    structural_score=round(struct_score, 4),
                    type_score=round(type_score, 4),
                    class_token_score=round(class_score, 4),
                    composite_score=round(comp, 4),
                )
            )

        return rank_candidates(
            candidates,
            top_k=cfg.max_candidates_per_segment,
            min_threshold=cfg.candidate_min_threshold,
        )

    def batch_get_candidates(
        self,
        items: list[tuple[ComponentType, NormalizedSegment, str]],
    ) -> dict[str, list[RankedCandidate]]:
        """fingerprint_hash → ranked candidates for all items."""
        result: dict[str, list[RankedCandidate]] = {}
        for component_type, normalized, fingerprint_hash in items:
            result[fingerprint_hash] = self.get_candidates(
                component_type, normalized, fingerprint_hash
            )
        return result

    def get_signature(self, name: str) -> ComponentSignature | None:
        return self._index.get(name)

    def get_all_component_names(self) -> list[str]:
        return list(self._index.keys())

    def register_custom(self, defn: CustomComponentDefinition) -> None:
        """Add a custom component to the live index."""
        from component_mapper.registry.custom_registry import CustomRegistry

        cr = CustomRegistry()
        sig = cr._to_signature(defn)
        self._index[defn.name] = sig
        logger.debug("Registered custom component in index: %s", defn.name)

    def merge_custom(self, signatures: list[ComponentSignature]) -> None:
        """Merge custom registry signatures into the index (priority: custom first)."""
        for sig in signatures:
            self._index[sig.component_name] = sig
        logger.debug("Merged %d custom signatures into index", len(signatures))
