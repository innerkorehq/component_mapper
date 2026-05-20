"""
05_cache_and_persistence.py
===========================
Demonstrates the L1 mapping cache:
  - Pre-populate: seed the cache with known mappings (simulates a prior pipeline run)
  - Run 1: 100% cache hits — segments resolved instantly from cache, zero LLM calls
  - Run 2: same result (hits increment, but no recomputation)

Also shows:
  - How fingerprint_hash is the shared key between segment-classifier and component-mapper
  - Manual cache inspection (reading the JSON file directly)
  - How to seed the cache programmatically (useful for bootstrapping or testing)

Note: in production the cache is populated automatically when segments are mapped by
the structural-match or LLM stage. Here we pre-populate it directly to demonstrate
cache-hit behaviour without requiring an LLM API key.

Run:
    python examples/05_cache_and_persistence.py
"""

import asyncio
import json
from pathlib import Path
from segment_classifier.models import (
    ClassifiedSegment, ClassificationStage, ComponentType, SegmentPosition,
)
from segment_classifier.utils.html_normalizer import normalize_segment
from component_mapper import MapperPipeline, MapperSettings
from component_mapper.cache.mapping_cache import MappingCache
from component_mapper.models import (
    MappingCacheRecord, MappingStage, RegistrySource, PropMapping,
)
from component_mapper.config import MappingCacheConfig


CACHE_PATH = ".cache/demo_mapping_cache.json"

PRODUCT_HTML = """
<div class="card product">
  <img src="/shoe.jpg" alt="Trail Runner" />
  <h3>Trail Runner X1</h3>
  <p>India's best trail shoe.</p>
  <span class="price">₹4,999</span>
  <button>Add to Cart</button>
</div>"""

FAQ_HTML = """
<section class="faq accordion">
  <div><button>Free delivery?</button><p>Yes, above ₹999.</p></div>
  <div><button>Returns?</button><p>30-day hassle-free.</p></div>
</section>"""


def make_segments() -> list[ClassifiedSegment]:
    segments = []
    for seg_id, html, ctype, position in [
        ("prod_001", PRODUCT_HTML, ComponentType.COLLECTION_PRODUCT_CARD, SegmentPosition.MIDDLE),
        ("faq_001",  FAQ_HTML,     ComponentType.SECTION_FAQ,             SegmentPosition.MIDDLE),
    ]:
        norm = normalize_segment(html.strip(), "")
        segments.append(ClassifiedSegment(
            segment_id=seg_id,
            page_url="https://shopsme.in",
            page_slug="home",
            raw_html=html.strip(),
            text_content=" ".join(html.split()),
            position_hint=position,
            component_type=ctype,
            classification_stage=ClassificationStage.LLM,
            confidence=0.91,
            fingerprint_hash=norm.fingerprint_hash(),
        ))
    return segments


async def run(segments: list[ClassifiedSegment], run_label: str) -> None:
    settings = MapperSettings(
        astro_project_root="",
        generate_collection_schemas=False,
        mapping_cache=MappingCacheConfig(
            cache_path=CACHE_PATH,
            auto_persist_every=1,   # persist after every write for this demo
        ),
    )
    pipeline = MapperPipeline(settings)
    await pipeline.initialize()
    result = await pipeline.run(segments)
    await pipeline.shutdown()

    print(f"\n{'─'*50}")
    print(f"{run_label}")
    print(f"{'─'*50}")
    print(f"  Cache hit rate    : {result.cache_hit_rate:.0%}")
    print(f"  Structural match  : {result.structural_match_rate:.0%}")
    print(f"  LLM calls         : {result.llm_calls_made}")
    print(f"  Mapped            : {len(result.mapped)}")
    print(f"  Unresolved        : {len(result.unresolved)}")
    for comp in result.mapped:
        print(
            f"  {comp.segment_id:12}  {comp.component_name:20}"
            f"  stage={comp.mapping_stage.value}"
        )


def inspect_cache() -> None:
    path = Path(CACHE_PATH)
    if not path.exists():
        print("\n  (cache file does not exist yet)")
        return
    records = json.loads(path.read_text())
    print(f"\n{'─'*50}")
    print(f"Cache contents ({path})  —  {len(records)} records")
    print(f"{'─'*50}")
    for fp_hash, rec in records.items():
        print(
            f"  {fp_hash[:16]}…  →  {rec['component_name']:20}"
            f"  stage={rec['mapping_stage']:20}"
            f"  hits={rec['hit_count']}"
        )


async def seed_cache(segments: list[ClassifiedSegment]) -> None:
    """Pre-populate the cache — simulates a prior successful pipeline run."""
    cache = MappingCache(CACHE_PATH, auto_persist_every=1)
    await cache.load()

    seed_data = {
        "prod_001": ("card",      RegistrySource.SHADCN, MappingStage.STRUCTURAL_MATCH, 0.82),
        "faq_001":  ("accordion", RegistrySource.SHADCN, MappingStage.LLM_MAPPED,       0.91),
    }

    for seg in segments:
        name, source, stage, conf = seed_data[seg.segment_id]
        await cache.set(
            seg.fingerprint_hash,
            MappingCacheRecord(
                fingerprint_hash=seg.fingerprint_hash,
                component_name=name,
                registry_source=source,
                prop_mapping=PropMapping(mappings=[
                    {"segment_field": "h3:0",    "component_prop": "title",       "type": "text",      "confidence": 0.95, "ambiguous": False},
                    {"segment_field": "img[src]", "component_prop": "image",       "type": "image_url", "confidence": 0.99, "ambiguous": False},
                    {"segment_field": "p:0",      "component_prop": "description", "type": "text",      "confidence": 0.88, "ambiguous": False},
                ]),
                mapping_stage=stage,
                confidence=conf,
                hit_count=0,
            ),
        )
    await cache.persist()
    print(f"  Seeded {cache.size} cache records → {CACHE_PATH}")


async def main():
    print("=" * 50)
    print("Cache demo")
    print("=" * 50)

    segments = make_segments()
    print("Segments and their fingerprints:")
    for seg in segments:
        print(f"  {seg.segment_id:12}  fingerprint={seg.fingerprint_hash[:24]}…")

    # Wipe from any previous demo run, then seed
    Path(CACHE_PATH).unlink(missing_ok=True)
    print("\nPre-populating cache (simulating a prior pipeline run):")
    await seed_cache(segments)
    inspect_cache()

    await run(segments, "Run 1 — warm cache (100% hits, zero LLM calls)")
    inspect_cache()

    await run(segments, "Run 2 — still 100% hits (hit_count increments)")
    inspect_cache()

    # ── Fingerprint link demo ─────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Fingerprint hash: shared key between segment-classifier and component-mapper")
    print(f"{'─'*50}")
    for seg in segments:
        norm = normalize_segment(seg.raw_html, seg.text_content)
        computed = norm.fingerprint_hash()
        match = "✓ match" if computed == seg.fingerprint_hash else "✗ mismatch"
        print(f"  {seg.segment_id:12}  stored={seg.fingerprint_hash[:16]}…  recomputed={computed[:16]}…  {match}")


if __name__ == "__main__":
    asyncio.run(main())
