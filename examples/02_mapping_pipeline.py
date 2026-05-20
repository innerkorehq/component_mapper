"""
02_mapping_pipeline.py
======================
Demonstrates the MapperPipeline end-to-end.

Accepts pre-built ClassifiedSegments (the output of segment-classifier)
and maps each to a Shadcn UI component + Astro wrapper.

Without LLM_API_KEY:   segments with ambiguous structural matches are UNRESOLVED.
With LLM_API_KEY set:  all stages run fully (cache → structural → LLM → Astro).

Run:
    LLM_API_KEY=sk-ant-... python examples/02_mapping_pipeline.py
    python examples/02_mapping_pipeline.py    # runs without LLM, shows structural path
"""

import asyncio
import os
import textwrap
from segment_classifier.models import (
    ClassifiedSegment, ClassificationStage, ComponentType, SegmentPosition,
)
from segment_classifier.utils.html_normalizer import normalize_segment
from component_mapper import MapperPipeline, MapperSettings
from component_mapper.config import LiteLLMConfig, MappingCacheConfig


# ---------------------------------------------------------------------------
# Sample classified segments
# (in production these come from ClassifierPipeline.run())
# ---------------------------------------------------------------------------

SEGMENTS_HTML = {
    "seg_header": """
        <header id="site-header">
          <nav>
            <a href="/">Home</a><a href="/products">Products</a>
            <a href="/blog">Blog</a><a href="/about">About</a>
          </nav>
        </header>""",

    "seg_product_1": """
        <div class="card product">
          <img src="/shoes/trail-x1.jpg" alt="Trail Runner X1" />
          <h3>Trail Runner X1</h3>
          <p>Grippy outsole for muddy paths.</p>
          <span class="price">₹4,999</span>
          <button>Add to Cart</button>
        </div>""",

    "seg_product_2": """
        <div class="card product">
          <img src="/shoes/road-pro.jpg" alt="Road Sprint Pro" />
          <h3>Road Sprint Pro</h3>
          <p>Lightweight for daily road runs.</p>
          <span class="price">₹3,499</span>
          <button>Add to Cart</button>
        </div>""",

    "seg_faq": """
        <section class="faq accordion">
          <div><button>Do you offer free delivery?</button>
               <p>Yes — free on orders above ₹999.</p></div>
          <div><button>What is your return policy?</button>
               <p>30-day hassle-free returns.</p></div>
        </section>""",

    "seg_hero": """
        <section class="hero">
          <div>
            <h1>Run Further, Run Better</h1>
            <p>Premium running shoes engineered for Indian terrain.</p>
            <a href="/products">Shop Now</a>
          </div>
          <div><img src="/hero-shoes.jpg" alt="Running shoes" /></div>
        </section>""",
}

COMPONENT_TYPES = {
    "seg_header":    ComponentType.LAYOUT_HEADER,
    "seg_product_1": ComponentType.COLLECTION_PRODUCT_CARD,
    "seg_product_2": ComponentType.COLLECTION_PRODUCT_CARD,
    "seg_faq":       ComponentType.SECTION_FAQ,
    "seg_hero":      ComponentType.SECTION_HERO,
}

POSITION_HINTS = {
    "seg_header":    SegmentPosition.TOP,
    "seg_product_1": SegmentPosition.MIDDLE,
    "seg_product_2": SegmentPosition.MIDDLE,
    "seg_faq":       SegmentPosition.MIDDLE,
    "seg_hero":      SegmentPosition.TOP,
}


def make_segments() -> list[ClassifiedSegment]:
    segments = []
    for seg_id, html in SEGMENTS_HTML.items():
        norm = normalize_segment(html.strip(), "")
        segments.append(ClassifiedSegment(
            segment_id=seg_id,
            page_url="https://shopsme.in/products",
            page_slug="products",
            raw_html=html.strip(),
            text_content=" ".join(html.split()),
            position_hint=POSITION_HINTS[seg_id],
            component_type=COMPONENT_TYPES[seg_id],
            classification_stage=ClassificationStage.LLM,
            confidence=0.91,
            fingerprint_hash=norm.fingerprint_hash(),
            sibling_count=2 if "product" in seg_id else 0,
        ))
    return segments


# ---------------------------------------------------------------------------
# Pipeline run
# ---------------------------------------------------------------------------

async def run_pipeline(segments: list[ClassifiedSegment]) -> None:
    has_key = bool(os.environ.get("LLM_API_KEY"))
    print(f"LLM_API_KEY: {'set ✓' if has_key else 'not set — LLM stage will be skipped'}\n")

    settings = MapperSettings(
        astro_project_root="./output/shopsme",
        generate_collection_schemas=True,
        litellm=LiteLLMConfig(
            config_path="litellm_config.json",
            api_key_env="LLM_API_KEY",
        ),
        mapping_cache=MappingCacheConfig(
            cache_path=".cache/example_mapping_cache.json",
        ),
    )

    pipeline = MapperPipeline(settings)
    await pipeline.initialize()
    result = await pipeline.run(segments)
    await pipeline.shutdown()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Pipeline result")
    print("=" * 60)
    print(f"  Total segments    : {result.total_segments}")
    print(f"  Mapped            : {len(result.mapped)}")
    print(f"  Unresolved        : {len(result.unresolved)}")
    print(f"  Cache hit rate    : {result.cache_hit_rate:.0%}")
    print(f"  Structural matches: {result.structural_match_rate:.0%}")
    print(f"  LLM calls made    : {result.llm_calls_made}")
    print(f"  Stage breakdown   :")
    for stage, count in result.stage_breakdown.items():
        if count:
            print(f"    {stage.value:22} {count}")

    # ── Per-component detail ──────────────────────────────────────────────────
    if result.mapped:
        print(f"\n{'─'*60}")
        print("Mapped components")
        print(f"{'─'*60}")
        for comp in result.mapped:
            print(f"\n  {comp.segment_id}")
            print(f"    component : {comp.component_name}  ({comp.registry_source.value})")
            print(f"    stage     : {comp.mapping_stage.value}  confidence={comp.mapping_confidence:.2f}")
            print(f"    file      : {comp.astro_component.file_path}")
            if comp.prop_mapping.mappings:
                print(f"    props     :")
                for m in comp.prop_mapping.mappings:
                    flag = " ⚠ ambiguous" if m.get("ambiguous") else ""
                    print(f"      {m['segment_field']:30} → {m['component_prop']}{flag}")
            if comp.content_collection_schema:
                print(f"    collection: {comp.content_collection_schema.collection_name}")

    # ── Show one Astro file ───────────────────────────────────────────────────
    if result.mapped:
        first = result.mapped[0]
        print(f"\n{'─'*60}")
        print(f"Generated .astro for {first.segment_id}")
        print(f"{'─'*60}")
        print(first.astro_component.full_file_content)

    # ── Unresolved detail ─────────────────────────────────────────────────────
    if result.unresolved:
        print(f"\n{'─'*60}")
        print("Unresolved (need LLM key or manual mapping)")
        print(f"{'─'*60}")
        for seg in result.unresolved:
            print(f"  {seg.segment_id}  ({seg.component_type.value})")

    # ── Install manifest ──────────────────────────────────────────────────────
    if result.install_commands:
        print(f"\n{'─'*60}")
        print("Install manifest (run before building Astro project)")
        print(f"{'─'*60}")
        for cmd in result.install_commands:
            print(f"  {cmd}")


async def main():
    segments = make_segments()
    print(f"Created {len(segments)} ClassifiedSegments\n")
    await run_pipeline(segments)


if __name__ == "__main__":
    asyncio.run(main())
