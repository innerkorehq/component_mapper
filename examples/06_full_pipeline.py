"""
06_full_pipeline.py
===================
Full end-to-end pipeline against a live URL:
    page-segmenter (https://getaipage.com)
        → rule-based classification (role → ComponentType)
        → component-mapper (Shadcn + Astro generation)

In production the classification step is handled by segment-classifier's
ClassifierPipeline. This example bridges the two packages by converting
page-segmenter Segment objects to ClassifiedSegments directly, demonstrating
exactly how the three packages connect.

Without LLM_API_KEY:  structural matches only; ambiguous segments stay UNRESOLVED.
With LLM_API_KEY set: all segments get fully resolved.

Run:
    python examples/06_full_pipeline.py
    LLM_API_KEY=sk-ant-... python examples/06_full_pipeline.py
"""

import asyncio
import os
from segment_classifier.models import (
    ClassifiedSegment, ClassificationStage, ComponentType, SegmentPosition,
)
from segment_classifier.utils.html_normalizer import normalize_segment
from bs4 import BeautifulSoup
from page_segmenter import find_segments
from component_mapper import MapperPipeline, MapperSettings
from component_mapper.config import LiteLLMConfig, MappingCacheConfig


TARGET_URL = "https://getaipage.com"


# ---------------------------------------------------------------------------
# Role → ComponentType mapping
# (In production this is handled by ClassifierPipeline)
# ---------------------------------------------------------------------------

ROLE_TO_COMPONENT_TYPE: dict[str, ComponentType] = {
    "header":       ComponentType.LAYOUT_HEADER,
    "footer":       ComponentType.LAYOUT_FOOTER,
    "nav":          ComponentType.LAYOUT_NAV,
    "sidebar":      ComponentType.LAYOUT_SIDEBAR,
    "hero":         ComponentType.SECTION_HERO,
    "card":         ComponentType.COLLECTION_PRODUCT_CARD,
    "grid":         ComponentType.COLLECTION_PRODUCT_LIST,
    "features":     ComponentType.SECTION_FEATURE_GRID,
    "testimonials": ComponentType.SECTION_TESTIMONIAL,
    "cta":          ComponentType.SECTION_CTA,
    "faq":          ComponentType.SECTION_FAQ,
    "pricing":      ComponentType.SECTION_PRICING,
    "form":         ComponentType.UI_FORM,
    "modal":        ComponentType.UI_MODAL,
    "article":      ComponentType.CONTENT_ARTICLE,
    "section":      ComponentType.UNKNOWN,
    "main":         ComponentType.UNKNOWN,
}


# ---------------------------------------------------------------------------
# Step 1: Segment the live page
# ---------------------------------------------------------------------------

async def step1_segment(url: str) -> list[dict]:
    print("=" * 60)
    print(f"Step 1: page-segmenter → {url}")
    print("=" * 60)

    segments = await find_segments(url, page_type="marketing")

    print(f"Found {len(segments)} top-level segments:")
    for seg in segments:
        children = seg.get("children", [])
        print(
            f"  role={seg['role']:12}  depth={seg['depth']}"
            f"  score={seg['identityScore']:2}  signals={seg['identitySignals']}"
            f"  children={len(children)}"
        )
    return segments


# ---------------------------------------------------------------------------
# Step 2: Classify (rule-based role → ComponentType)
# ---------------------------------------------------------------------------

def _flatten(segments: list[dict], result: list[dict] | None = None) -> list[dict]:
    """Recursively flatten nested segment tree into a flat list."""
    if result is None:
        result = []
    for seg in segments:
        result.append(seg)
        _flatten(seg.get("children", []), result)
    return result


def step2_classify(raw_segments: list[dict]) -> list[ClassifiedSegment]:
    print("\n" + "=" * 60)
    print("Step 2: classify — page-segmenter role → ComponentType")
    print("=" * 60)

    all_segs = _flatten(raw_segments)
    classified = []

    for i, seg in enumerate(all_segs):
        raw_html = seg.get("rawHtml", "")
        if not raw_html:
            continue

        role         = seg.get("role", "section")
        ctype        = ROLE_TO_COMPONENT_TYPE.get(role, ComponentType.UNKNOWN)
        text_content = BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
        norm         = normalize_segment(raw_html, text_content)

        classified.append(ClassifiedSegment(
            segment_id=f"seg_{i:03d}_{role}",
            page_url=TARGET_URL,
            page_slug="home",
            raw_html=raw_html,
            text_content=text_content[:300],
            position_hint=(
                SegmentPosition.TOP    if seg.get("depth", 0) == 0 and role in ("header", "hero", "nav")
                else SegmentPosition.BOTTOM if role == "footer"
                else SegmentPosition.MIDDLE
            ),
            component_type=ctype,
            classification_stage=ClassificationStage.RULE_BASED,
            confidence=0.80,
            fingerprint_hash=norm.fingerprint_hash(),
        ))
        print(f"  seg_{i:03d}  role={role:12}  → {ctype.value}")

    return classified


# ---------------------------------------------------------------------------
# Step 3: Map to Shadcn components
# ---------------------------------------------------------------------------

async def step3_map(segments: list[ClassifiedSegment]) -> None:
    print("\n" + "=" * 60)
    print("Step 3: component-mapper — Shadcn + Astro generation")
    print("=" * 60)

    has_key = bool(os.environ.get("LLM_API_KEY"))
    print(f"LLM_API_KEY: {'set ✓' if has_key else 'not set — LLM stage disabled'}\n")

    settings = MapperSettings(
        astro_project_root="./output/getaipage",
        generate_collection_schemas=True,
        litellm=LiteLLMConfig(
            config_path="litellm_config.json",
            api_key_env="LLM_API_KEY",
        ),
        mapping_cache=MappingCacheConfig(
            cache_path=".cache/getaipage_cache.json",
        ),
    )

    pipeline = MapperPipeline(settings)
    await pipeline.initialize()
    result = await pipeline.run(segments)
    await pipeline.shutdown()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print("Results")
    print(f"{'─'*60}")
    print(f"  Total       : {result.total_segments}")
    print(f"  Mapped      : {len(result.mapped)}")
    print(f"  Unresolved  : {len(result.unresolved)}")
    print(f"  Cache hits  : {result.cache_hit_rate:.0%}")
    print(f"  Structural  : {result.structural_match_rate:.0%}")
    print(f"  LLM calls   : {result.llm_calls_made}")

    if result.stage_breakdown:
        print(f"  Stages      :")
        for stage, count in result.stage_breakdown.items():
            if count:
                print(f"    {stage.value:25} {count}")

    # ── Per-component ─────────────────────────────────────────────────────────
    if result.mapped:
        print(f"\n{'─'*60}")
        print("Mapped components")
        print(f"{'─'*60}")
        for comp in result.mapped:
            print(
                f"  {comp.segment_id:22}  {comp.component_name:22}"
                f"  [{comp.registry_source.value:6}]"
                f"  conf={comp.mapping_confidence:.2f}"
                f"  stage={comp.mapping_stage.value}"
            )
            if comp.prop_mapping.mappings:
                for m in comp.prop_mapping.mappings:
                    flag = " ⚠" if m.get("ambiguous") else ""
                    print(f"    {m['segment_field']:30} → {m['component_prop']}{flag}")

    # ── Unresolved ────────────────────────────────────────────────────────────
    if result.unresolved:
        print(f"\n{'─'*60}")
        print(f"Unresolved ({len(result.unresolved)}) — set LLM_API_KEY to resolve")
        print(f"{'─'*60}")
        for seg in result.unresolved:
            print(f"  {seg.segment_id:22}  {seg.component_type.value}")

    # ── Astro files written ───────────────────────────────────────────────────
    if result.mapped and settings.astro_project_root:
        print(f"\n{'─'*60}")
        print(f"Generated Astro files → {settings.astro_project_root}/")
        print(f"{'─'*60}")
        for comp in result.mapped:
            print(f"  {comp.astro_component.file_path}")

    # ── Highest-confidence Astro output ──────────────────────────────────────
    if result.mapped:
        best = max(result.mapped, key=lambda c: c.mapping_confidence)
        print(f"\n{'─'*60}")
        print(f"Highest-confidence output: {best.segment_id} ({best.mapping_confidence:.2f})")
        print(f"{'─'*60}")
        print(best.astro_component.full_file_content)

    # ── Install manifest ──────────────────────────────────────────────────────
    if result.install_commands:
        print(f"\n{'─'*60}")
        print("Shadcn install manifest")
        print(f"{'─'*60}")
        for cmd in result.install_commands:
            print(f"  {cmd}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    raw_segments = await step1_segment(TARGET_URL)

    if not raw_segments:
        print("No segments found — check URL or network connection.")
        return

    classified = step2_classify(raw_segments)

    if not classified:
        print("No segments with HTML — check page-segmenter output.")
        return

    print(f"\nClassified {len(classified)} segments from {TARGET_URL}")
    await step3_map(classified)


if __name__ == "__main__":
    asyncio.run(main())
