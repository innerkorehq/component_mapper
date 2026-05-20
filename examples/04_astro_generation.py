"""
04_astro_generation.py
======================
Demonstrates direct Astro file generation without running the full pipeline.
Useful for:
  - Previewing what Astro output looks like for a known component
  - Generating Astro wrappers in build tooling or code-gen scripts
  - Writing tests against the Astro generator

No API key or MCP required.

Run:
    python examples/04_astro_generation.py
"""

from pathlib import Path
from segment_classifier.models import ClassifiedSegment, ClassificationStage, ComponentType, SegmentPosition
from segment_classifier.utils.html_normalizer import normalize_segment
from component_mapper.models import (
    ComponentSignature, RegistrySource, InteractivityMode,
    PropDefinition, PropMapping,
)
from component_mapper.registry.astro_generator import (
    generate_astro_component,
    generate_content_collection_schema,
)
from component_mapper.registry.prop_mapper import infer_prop_mapping


# ---------------------------------------------------------------------------
# Helper: build a minimal ComponentSignature by hand
# ---------------------------------------------------------------------------

def card_signature() -> ComponentSignature:
    return ComponentSignature(
        component_name="card",
        registry_source=RegistrySource.SHADCN,
        dom_skeleton="div>[div>img+h3+p, div, div>button]",
        root_element="div",
        required_children=["CardHeader", "CardContent", "CardFooter"],
        optional_children=["CardDescription"],
        structural_class_tokens=["card", "content", "header", "footer"],
        typical_nesting_depth=3,
        child_tag_counts={"div": 4, "h3": 1, "p": 1, "img": 1, "button": 1},
        unique_tag_count=5,
        compatible_component_types=[
            ComponentType.COLLECTION_PRODUCT_CARD,
            ComponentType.COLLECTION_BLOG_CARD,
            ComponentType.SECTION_FEATURE_GRID,
        ],
        interactivity=InteractivityMode.STATIC,
        description="Shadcn Card component with header, content, and footer slots",
        props=[
            PropDefinition(name="title",       type="string",    required=True),
            PropDefinition(name="description", type="string",    required=False),
            PropDefinition(name="image",       type="string",    required=False),
            PropDefinition(name="price",       type="string",    required=False),
            PropDefinition(name="ctaLabel",    type="string",    required=False, default_value="View Details"),
            PropDefinition(name="footer",      type="ReactNode", required=False),
        ],
        astro_import="@/components/ui/card",
        install_command="npx shadcn@latest add card",
        requires_client_directive=False,
    )


def accordion_signature() -> ComponentSignature:
    return ComponentSignature(
        component_name="accordion",
        registry_source=RegistrySource.SHADCN,
        dom_skeleton="div>[div>[button+div>p]]",
        root_element="div",
        required_children=["AccordionItem", "AccordionTrigger", "AccordionContent"],
        optional_children=[],
        structural_class_tokens=["accordion", "faq"],
        typical_nesting_depth=3,
        child_tag_counts={"div": 3, "button": 1, "p": 1},
        unique_tag_count=3,
        compatible_component_types=[ComponentType.SECTION_FAQ],
        interactivity=InteractivityMode.INTERACTIVE,
        description="Shadcn Accordion for FAQ / collapsible content",
        props=[
            PropDefinition(name="type",         type="string",    required=False, default_value="single"),
            PropDefinition(name="collapsible",  type="boolean",   required=False),
            PropDefinition(name="children",     type="ReactNode", required=True),
        ],
        astro_import="@/components/ui/accordion",
        install_command="npx shadcn@latest add accordion",
        requires_client_directive=True,
    )


# ---------------------------------------------------------------------------
# Example 1 — Product card with automatic prop inference
# ---------------------------------------------------------------------------

def example_product_card():
    print("=" * 60)
    print("Example 1: Product card — inferred prop mapping")
    print("=" * 60)

    html = """
    <div class="card product">
      <img src="/shoes/trail-x1.jpg" alt="Trail Runner X1" />
      <h3>Trail Runner X1</h3>
      <p>Grippy outsole for muddy terrain.</p>
      <span class="price">₹4,999</span>
      <button>Add to Cart</button>
    </div>"""

    sig = card_signature()
    norm = normalize_segment(html.strip(), "")

    # Automatically infer which HTML nodes map to which props
    prop_mapping = infer_prop_mapping(html, sig.props)
    print(f"Inferred {len(prop_mapping.mappings)} prop mappings:")
    for m in prop_mapping.mappings:
        flag = "  ⚠ ambiguous" if m.get("ambiguous") else ""
        print(f"  {m['segment_field']:35} → {m['component_prop']:15} ({m['confidence']:.2f}){flag}")
    if prop_mapping.unmapped_props:
        print(f"  Unmapped required props: {prop_mapping.unmapped_props}")

    seg = ClassifiedSegment(
        segment_id="prod_001",
        page_url="https://shopsme.in/products",
        page_slug="products",
        raw_html=html.strip(),
        text_content="Trail Runner X1 Grippy outsole ₹4,999 Add to Cart",
        position_hint=SegmentPosition.MIDDLE,
        component_type=ComponentType.COLLECTION_PRODUCT_CARD,
        classification_stage=ClassificationStage.LLM,
        confidence=0.93,
        fingerprint_hash=norm.fingerprint_hash(),
        sibling_count=3,
    )

    astro = generate_astro_component(seg, sig, prop_mapping, "card")
    print(f"\nGenerated: {astro.file_path}")
    print(f"Install  : {astro.install_commands}")
    print(f"\n{astro.full_file_content}")

    # Content Collection schema (for collection.* types)
    schema = generate_content_collection_schema(
        ComponentType.COLLECTION_PRODUCT_CARD, prop_mapping, sig
    )
    print(f"Content Collection schema ({schema.collection_name}):")
    print(schema.zod_schema)
    print(f"\nExample entry:\n{schema.example_entry}")


# ---------------------------------------------------------------------------
# Example 2 — Interactive component (accordion → client:load directive)
# ---------------------------------------------------------------------------

def example_interactive_accordion():
    print("\n" + "=" * 60)
    print("Example 2: FAQ accordion — interactive (client:load)")
    print("=" * 60)

    html = """
    <section class="faq accordion">
      <div><button>Free delivery?</button><div><p>Yes, above ₹999.</p></div></div>
      <div><button>Return policy?</button><div><p>30-day returns.</p></div></div>
    </section>"""

    sig = accordion_signature()
    norm = normalize_segment(html.strip(), "")

    prop_mapping = infer_prop_mapping(html, sig.props)

    seg = ClassifiedSegment(
        segment_id="faq_001",
        page_url="https://shopsme.in",
        page_slug="home",
        raw_html=html.strip(),
        text_content="Free delivery? Yes. Return policy? 30-day returns.",
        position_hint=SegmentPosition.MIDDLE,
        component_type=ComponentType.SECTION_FAQ,
        classification_stage=ClassificationStage.L1_EXACT_CACHE,
        confidence=0.97,
        fingerprint_hash=norm.fingerprint_hash(),
    )

    astro = generate_astro_component(seg, sig, prop_mapping, "accordion")
    print(f"Client directive : {astro.client_directive!r}   (INTERACTIVE → client:load)")
    print(f"File path        : {astro.file_path}")
    print(f"\n{astro.full_file_content}")


# ---------------------------------------------------------------------------
# Example 3 — Write generated files to disk
# ---------------------------------------------------------------------------

def example_write_to_disk():
    print("\n" + "=" * 60)
    print("Example 3: Write Astro files to disk")
    print("=" * 60)

    output_root = Path("./output/astro_example")
    sig = card_signature()
    norm = normalize_segment("<div class='card'><h3>Title</h3></div>", "")

    seg = ClassifiedSegment(
        segment_id="disk_001",
        page_url="https://shopsme.in",
        page_slug="home",
        raw_html="<div class='card'><h3>Title</h3></div>",
        text_content="Title",
        position_hint=SegmentPosition.MIDDLE,
        component_type=ComponentType.COLLECTION_PRODUCT_CARD,
        classification_stage=ClassificationStage.RULE_BASED,
        confidence=1.0,
        fingerprint_hash=norm.fingerprint_hash(),
    )
    prop_mapping = infer_prop_mapping(seg.raw_html, sig.props)
    astro = generate_astro_component(seg, sig, prop_mapping, "card")

    file_path = output_root / astro.file_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(astro.full_file_content)
    print(f"Wrote {file_path}")
    print(f"Content ({len(astro.full_file_content)} chars):\n{astro.full_file_content}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_product_card()
    example_interactive_accordion()
    example_write_to_disk()
