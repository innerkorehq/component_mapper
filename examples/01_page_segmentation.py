"""
01_page_segmentation.py
=======================
Demonstrates page-segmenter: extract logical segments from raw HTML.

No network or API key required — uses inline HTML.
Run:
    python examples/01_page_segmentation.py
"""

import asyncio
import json
from page_segmenter import find_segments_from_html, SegmenterConfig, LogicalSegmenter

# ---------------------------------------------------------------------------
# Sample HTML — a realistic SME product page excerpt
# ---------------------------------------------------------------------------

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>ShopSME — Running Shoes</title></head>
<body>
  <header id="site-header" style="background:#fff;border-bottom:1px solid #eee;padding:16px 32px;">
    <nav>
      <a href="/">Home</a>
      <a href="/products">Products</a>
      <a href="/blog">Blog</a>
      <a href="/contact">Contact</a>
    </nav>
  </header>

  <section id="hero" style="background:#f5f5f5;padding:64px 32px;border-bottom:2px solid #ddd;">
    <h1>Run Further, Run Better</h1>
    <p>Premium running shoes engineered for Indian terrain.</p>
    <a href="/products" style="background:#e63946;color:#fff;padding:12px 24px;border-radius:4px;">
      Shop Now
    </a>
  </section>

  <section id="products" style="padding:48px 32px;">
    <h2>Featured Products</h2>
    <div class="product-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:24px;">
      <div class="product-card" style="border:1px solid #ddd;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:16px;">
        <img src="/shoe1.jpg" alt="Trail Runner X1" style="width:100%;" />
        <h3>Trail Runner X1</h3>
        <p>Grippy outsole for muddy paths.</p>
        <span class="price">₹4,999</span>
        <button style="width:100%;margin-top:12px;">Add to Cart</button>
      </div>
      <div class="product-card" style="border:1px solid #ddd;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:16px;">
        <img src="/shoe2.jpg" alt="Road Sprint Pro" style="width:100%;" />
        <h3>Road Sprint Pro</h3>
        <p>Lightweight for daily road runs.</p>
        <span class="price">₹3,499</span>
        <button style="width:100%;margin-top:12px;">Add to Cart</button>
      </div>
      <div class="product-card" style="border:1px solid #ddd;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:16px;">
        <img src="/shoe3.jpg" alt="Hiker Peak 2" style="width:100%;" />
        <h3>Hiker Peak 2</h3>
        <p>Ankle support for rough trails.</p>
        <span class="price">₹5,999</span>
        <button style="width:100%;margin-top:12px;">Add to Cart</button>
      </div>
    </div>
  </section>

  <section id="faq" style="padding:48px 32px;background:#fafafa;border-top:1px solid #eee;">
    <h2>Frequently Asked Questions</h2>
    <div class="accordion">
      <div style="border:1px solid #ddd;border-radius:4px;margin-bottom:8px;padding:16px;">
        <button>Do you offer free delivery?</button>
        <p>Yes — free delivery on all orders above ₹999.</p>
      </div>
      <div style="border:1px solid #ddd;border-radius:4px;margin-bottom:8px;padding:16px;">
        <button>What is your return policy?</button>
        <p>30-day hassle-free returns on unworn shoes.</p>
      </div>
      <div style="border:1px solid #ddd;border-radius:4px;margin-bottom:8px;padding:16px;">
        <button>Are sizes available for wide feet?</button>
        <p>Yes — all models are available in wide-fit variants.</p>
      </div>
    </div>
  </section>

  <footer id="site-footer" style="background:#1a1a2e;color:#fff;padding:32px;">
    <p>© 2026 ShopSME. All rights reserved.</p>
  </footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Example 1 — find_segments_from_html (the primary public API)
# ---------------------------------------------------------------------------

async def example_basic():
    print("=" * 60)
    print("Example 1: find_segments_from_html")
    print("=" * 60)

    segments = await find_segments_from_html(
        html=SAMPLE_HTML,
        base_url="https://shopsme.in",
        page_type="commerce",   # drives SegmenterConfig tuning
    )

    print(f"Found {len(segments)} top-level segments\n")
    for seg in segments:
        children = seg.get("children", [])
        print(
            f"  [{seg['role']:12}]  depth={seg['depth']}  "
            f"score={seg['identityScore']}  "
            f"signals={seg['identitySignals']}  "
            f"children={len(children)}"
        )
        # Show first 120 chars of raw HTML for context
        raw = seg.get("rawHtml", "")
        print(f"             html={raw[:120].strip()!r}{'…' if len(raw)>120 else ''}")

    return segments


# ---------------------------------------------------------------------------
# Example 2 — page-type config differences
# ---------------------------------------------------------------------------

async def example_config_comparison():
    print("\n" + "=" * 60)
    print("Example 2: page-type config comparison")
    print("=" * 60)

    for page_type in ("commerce", "marketing", "content"):
        segs = await find_segments_from_html(SAMPLE_HTML, page_type=page_type)
        print(f"  page_type={page_type!r:12}  → {len(segs)} segments")


# ---------------------------------------------------------------------------
# Example 3 — access Segment dataclass fields directly via LogicalSegmenter
# ---------------------------------------------------------------------------

async def example_with_playwright_page():
    """
    Uses LogicalSegmenter directly (gives access to the Segment dataclass
    rather than plain dicts).  Useful when you need identity_signals or
    bounding_box in structured form.
    """
    print("\n" + "=" * 60)
    print("Example 3: LogicalSegmenter — Segment dataclass access")
    print("=" * 60)

    from playwright.async_api import async_playwright
    from page_segmenter import SegmenterConfig

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.set_content(SAMPLE_HTML, wait_until="domcontentloaded")

        config = SegmenterConfig(
            min_subtree_depth=2,
            min_subtree_nodes=5,
            component_score_threshold=3,
        )
        segmenter = LogicalSegmenter(page, config)
        segments = await segmenter.segment()
        await browser.close()

    print(f"Found {len(segments)} Segment objects")
    for seg in segments:
        print(
            f"  role={seg.role:10}  depth={seg.depth}"
            f"  score={seg.identity_score}"
            f"  selector={seg.selector!r}"
        )
        print(f"    signals: {seg.identity_signals}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    segments = await example_basic()
    await example_config_comparison()
    await example_with_playwright_page()

    # Show a single segment as JSON
    if segments:
        print("\n--- First segment as JSON ---")
        first = {k: v for k, v in segments[0].items() if k != "rawHtml"}
        print(json.dumps(first, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
