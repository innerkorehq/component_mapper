"""
03_custom_components.py
=======================
Demonstrates registering custom getaipage components that take priority
over generic Shadcn components when both are candidates for a segment type.

Use case: your platform has branded components (SMEProductHero, RatingWidget,
IndiaMapWidget) that should be preferred over generic Shadcn primitives.

No API key required — custom components bypass the LLM stage entirely when
they match structurally.

Run:
    python examples/03_custom_components.py
"""

import asyncio
from segment_classifier.models import (
    ClassifiedSegment, ClassificationStage, ComponentType, SegmentPosition,
)
from segment_classifier.utils.html_normalizer import normalize_segment
from component_mapper import MapperPipeline, MapperSettings
from component_mapper.models import (
    CustomComponentDefinition, InteractivityMode, PropDefinition,
)
from component_mapper.config import MappingCacheConfig


# ---------------------------------------------------------------------------
# Custom component definitions — getaipage branded components
# ---------------------------------------------------------------------------

SME_PRODUCT_HERO = CustomComponentDefinition(
    name="sme-product-hero",
    dom_skeleton="section>[div>[h1+p+a], div>[img]]",
    structural_class_tokens=["hero", "product"],
    compatible_component_types=[ComponentType.SECTION_HERO],
    props=[
        PropDefinition(name="productName", type="string", required=True),
        PropDefinition(name="tagline",     type="string", required=False),
        PropDefinition(name="heroImage",   type="string", required=True),
        PropDefinition(name="ctaLabel",    type="string", required=False, default_value="Shop Now"),
        PropDefinition(name="ctaHref",     type="string", required=False, default_value="/products"),
    ],
    astro_import="@/components/custom/SmeProductHero.astro",
    interactivity=InteractivityMode.STATIC,
    description="Branded hero section for SME product pages",
)

STAR_RATING_WIDGET = CustomComponentDefinition(
    name="star-rating-widget",
    dom_skeleton="div>[div>[span+span+span+span+span], span]",
    structural_class_tokens=["rating", "widget"],
    compatible_component_types=[ComponentType.COLLECTION_PRODUCT_CARD],
    props=[
        PropDefinition(name="score", type="number", required=True),
        PropDefinition(name="max",   type="number", required=False, default_value="5"),
        PropDefinition(name="label", type="string", required=False),
    ],
    astro_import="@/components/custom/StarRatingWidget.astro",
    interactivity=InteractivityMode.STATIC,
    description="Five-star rating display for product cards",
)

INDIA_MAP_WIDGET = CustomComponentDefinition(
    name="india-map-widget",
    dom_skeleton="div>[svg, div>[h3+ul>[li]]]",
    structural_class_tokens=["media", "widget"],
    compatible_component_types=[ComponentType.CONTENT_MEDIA],
    props=[
        PropDefinition(name="highlightedStates", type="string[]", required=False),
        PropDefinition(name="mapTitle",          type="string",   required=False),
    ],
    astro_import="@/components/custom/IndiaMapWidget.astro",
    interactivity=InteractivityMode.PARTIAL,
    description="SVG India map with state-level highlighting",
)


# ---------------------------------------------------------------------------
# Sample segments targeting the custom components
# ---------------------------------------------------------------------------

HERO_HTML = """
<section class="hero product">
  <div>
    <h1>Trail Runner X1</h1>
    <p>India's most trusted trail shoe, now in wide-fit.</p>
    <a href="/products/trail-x1">Shop Now</a>
  </div>
  <div><img src="/hero-trail-x1.jpg" alt="Trail Runner X1" /></div>
</section>"""

PRODUCT_CARD_HTML = """
<div class="card product">
  <img src="/shoe1.jpg" alt="Trail Runner X1" />
  <h3>Trail Runner X1</h3>
  <p>Grippy outsole for muddy paths.</p>
  <div class="rating widget">
    <div><span>★</span><span>★</span><span>★</span><span>★</span><span>☆</span></div>
    <span>4.2 / 5</span>
  </div>
  <span class="price">₹4,999</span>
  <button>Add to Cart</button>
</div>"""


def make_segments() -> list[ClassifiedSegment]:
    segments = []
    for seg_id, html, ctype, position in [
        ("seg_hero",    HERO_HTML,    ComponentType.SECTION_HERO,           SegmentPosition.TOP),
        ("seg_product", PRODUCT_CARD_HTML, ComponentType.COLLECTION_PRODUCT_CARD, SegmentPosition.MIDDLE),
    ]:
        norm = normalize_segment(html.strip(), "")
        segments.append(ClassifiedSegment(
            segment_id=seg_id,
            page_url="https://shopsme.in/products/trail-x1",
            page_slug="trail-x1",
            raw_html=html.strip(),
            text_content=" ".join(html.split()),
            position_hint=position,
            component_type=ctype,
            classification_stage=ClassificationStage.LLM,
            confidence=0.91,
            fingerprint_hash=norm.fingerprint_hash(),
        ))
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    settings = MapperSettings(
        astro_project_root="./output/custom_example",
        generate_collection_schemas=False,
        mapping_cache=MappingCacheConfig(
            cache_path=".cache/example_custom_cache.json",
        ),
    )

    pipeline = MapperPipeline(settings)
    await pipeline.initialize()

    # Register custom components BEFORE running — they get priority over Shadcn
    pipeline.custom_registry.register(SME_PRODUCT_HERO)
    pipeline.custom_registry.register(STAR_RATING_WIDGET)
    pipeline.custom_registry.register(INDIA_MAP_WIDGET)
    pipeline.signature_index.merge_custom(pipeline.custom_registry.get_all())

    print(f"Registered {len(pipeline.custom_registry.get_all())} custom components")
    for sig in pipeline.custom_registry.get_all():
        compatible = [ct.value for ct in sig.compatible_component_types]
        print(f"  {sig.component_name:28}  compatible_with={compatible}")

    segments = make_segments()
    result = await pipeline.run(segments)
    await pipeline.shutdown()

    print(f"\nMapped: {len(result.mapped)}  Unresolved: {len(result.unresolved)}")
    for comp in result.mapped:
        source_label = "CUSTOM ✓" if comp.registry_source.value == "custom" else "shadcn"
        print(f"\n  {comp.segment_id}")
        print(f"    component : {comp.component_name}  [{source_label}]")
        print(f"    stage     : {comp.mapping_stage.value}  confidence={comp.mapping_confidence:.2f}")
        print(f"    file      : {comp.astro_component.file_path}")
        print(f"\n{comp.astro_component.full_file_content}")


if __name__ == "__main__":
    asyncio.run(main())
