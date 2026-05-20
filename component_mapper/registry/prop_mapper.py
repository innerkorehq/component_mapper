import logging
from bs4 import BeautifulSoup
from component_mapper.models import PropDefinition, PropMapping

logger = logging.getLogger(__name__)

PROP_NAME_TO_TAG_SCORES: dict[str, dict[str, float]] = {
    "title": {"h1": 0.95, "h2": 0.90, "h3": 0.85, "h4": 0.75, "p": 0.40},
    "description": {"p": 0.90, "span": 0.70, "div": 0.50, "h4": 0.55},
    "image": {"img": 0.99, "picture": 0.95},
    "src": {"img": 0.99},
    "href": {"a": 0.99},
    "label": {"span": 0.85, "p": 0.70, "div": 0.60},
    "price": {"span": 0.80, "p": 0.75, "div": 0.60},
    "children": {"div": 0.70, "section": 0.70, "p": 0.65},
    "footer": {"button": 0.80, "div": 0.60, "a": 0.70},
    "action": {"button": 0.95, "a": 0.85},
    "badge": {"span": 0.80},
    "items": {"ul": 0.90, "ol": 0.85},
}

PROP_TYPE_COMPAT: dict[str, set[str]] = {
    "string": {"text", "image_url", "link"},
    "number": {"text"},
    "boolean": set(),  # no content node type maps naturally
    "ReactNode": {"text", "image_url", "link", "action", "list"},
    "any": {"text", "image_url", "link", "action", "list"},
}

CLASS_HINT_SCORES: dict[str, dict[str, float]] = {
    "price": {"price": 0.90, "cost": 0.85, "amount": 0.80},
    "badge": {"badge": 0.90, "tag": 0.80, "label": 0.75},
    "description": {"desc": 0.85, "summary": 0.80, "excerpt": 0.80},
    "title": {"title": 0.85, "heading": 0.80, "name": 0.80},
    "footer": {"footer": 0.85, "actions": 0.80},
}


def infer_prop_mapping(
    segment_html: str,
    props: list[PropDefinition],
) -> PropMapping:
    """Infer how segment content maps to component props. Returns PropMapping."""
    if not props:
        return PropMapping()

    try:
        content_nodes = _extract_content_nodes(segment_html)
    except Exception as exc:
        logger.warning("Failed to parse segment HTML for prop mapping: %s", exc)
        return PropMapping(unmapped_props=[p.name for p in props if p.required])

    if not content_nodes:
        return PropMapping(unmapped_props=[p.name for p in props if p.required])

    # Build cost matrix (we want to maximize, so we negate for scipy)
    n_props = len(props)
    n_nodes = len(content_nodes)

    score_matrix = []
    for prop in props:
        row = []
        for node in content_nodes:
            row.append(_score_assignment(node, prop))
        score_matrix.append(row)

    # Greedy assignment: highest score first
    assignments = _greedy_assign(score_matrix, n_props, n_nodes)

    mappings = []
    unmapped = []
    has_ambiguous = False

    for prop_idx, prop in enumerate(props):
        node_idx = assignments.get(prop_idx)
        if node_idx is None:
            if prop.required:
                unmapped.append(prop.name)
            continue

        node = content_nodes[node_idx]
        confidence = score_matrix[prop_idx][node_idx]

        if confidence < 0.10:
            if prop.required:
                unmapped.append(prop.name)
            continue

        ambiguous = confidence < 0.70
        if ambiguous:
            has_ambiguous = True

        # Determine content type
        content_type = _infer_content_type(node, prop)

        mappings.append(
            {
                "segment_field": node["selector"],
                "component_prop": prop.name,
                "type": content_type,
                "confidence": round(confidence, 4),
                "ambiguous": ambiguous,
            }
        )

    return PropMapping(
        mappings=mappings,
        has_ambiguous=has_ambiguous,
        unmapped_props=unmapped,
    )


def _extract_content_nodes(html: str) -> list[dict]:
    """Extract content nodes from HTML. Returns list of node descriptors."""
    soup = BeautifulSoup(html, "html.parser")
    nodes = []
    seen_selectors: set[str] = set()

    def make_selector(el, idx: int) -> str:
        tag = el.name
        classes = el.get("class", [])
        semantic_classes = [
            c
            for c in classes
            if any(
                kw in c.lower()
                for kw in (
                    "price",
                    "title",
                    "desc",
                    "badge",
                    "name",
                    "label",
                    "summary",
                    "heading",
                    "action",
                    "cta",
                    "footer",
                    "content",
                )
            )
        ]
        if semantic_classes:
            return f"{tag}.{semantic_classes[0]}"

        # Use position-based selector
        parent = el.parent
        if parent and parent.name:
            siblings = [
                s for s in parent.children if hasattr(s, "name") and s.name == tag
            ]
            if len(siblings) > 1:
                pos = siblings.index(el) + 1
                return f"{tag}:nth-of-type({pos})"
        return f"{tag}:{idx}"

    # Images
    for i, img in enumerate(soup.find_all("img")):
        sel = img.get("src") and "img[src]" or f"img:{i}"
        if sel not in seen_selectors:
            nodes.append(
                {
                    "tag": "img",
                    "selector": sel,
                    "type": "image_url",
                    "class_tokens": _get_class_tokens(img),
                    "content_preview": img.get("src", "") or img.get("alt", ""),
                    "attrs": {"src": img.get("src", ""), "alt": img.get("alt", "")},
                }
            )
            seen_selectors.add(sel)

    # Headings (by priority)
    for tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        for i, el in enumerate(soup.find_all(tag)):
            sel = make_selector(el, i)
            if sel not in seen_selectors:
                nodes.append(
                    {
                        "tag": tag,
                        "selector": sel,
                        "type": "text",
                        "class_tokens": _get_class_tokens(el),
                        "content_preview": el.get_text(strip=True)[:80],
                    }
                )
                seen_selectors.add(sel)

    # Paragraphs and spans
    for tag in ["p", "span"]:
        for i, el in enumerate(soup.find_all(tag)):
            sel = make_selector(el, i)
            if sel not in seen_selectors:
                text = el.get_text(strip=True)
                if text:
                    nodes.append(
                        {
                            "tag": tag,
                            "selector": sel,
                            "type": "text",
                            "class_tokens": _get_class_tokens(el),
                            "content_preview": text[:80],
                        }
                    )
                    seen_selectors.add(sel)

    # Links
    for i, a in enumerate(soup.find_all("a")):
        sel = make_selector(a, i)
        if sel not in seen_selectors:
            nodes.append(
                {
                    "tag": "a",
                    "selector": sel,
                    "type": "link",
                    "class_tokens": _get_class_tokens(a),
                    "content_preview": a.get("href", ""),
                    "attrs": {"href": a.get("href", "")},
                }
            )
            seen_selectors.add(sel)

    # Buttons
    for i, btn in enumerate(soup.find_all("button")):
        sel = make_selector(btn, i)
        if sel not in seen_selectors:
            nodes.append(
                {
                    "tag": "button",
                    "selector": sel,
                    "type": "action",
                    "class_tokens": _get_class_tokens(btn),
                    "content_preview": btn.get_text(strip=True)[:80],
                }
            )
            seen_selectors.add(sel)

    # Lists
    for i, ul in enumerate(soup.find_all(["ul", "ol"])):
        sel = make_selector(ul, i)
        if sel not in seen_selectors:
            items = [li.get_text(strip=True) for li in ul.find_all("li")]
            nodes.append(
                {
                    "tag": ul.name,
                    "selector": sel,
                    "type": "list",
                    "class_tokens": _get_class_tokens(ul),
                    "content_preview": str(items[:3]),
                }
            )
            seen_selectors.add(sel)

    return nodes


def _get_class_tokens(el) -> list[str]:
    classes = el.get("class", [])
    return [c.lower() for c in classes]


def _score_assignment(content_node: dict, prop: PropDefinition) -> float:
    """Score how well a content node matches a prop. Returns 0.0-1.0."""
    tag = content_node["tag"]
    prop_name_lower = prop.name.lower()
    node_type = content_node["type"]
    class_tokens = content_node.get("class_tokens", [])

    # Type compatibility gate
    prop_type = prop.type.strip()
    compat_types = PROP_TYPE_COMPAT.get(
        prop_type, {"text", "image_url", "link", "action", "list"}
    )
    if compat_types and node_type not in compat_types:
        return 0.0

    score = 0.0

    # Direct prop name → tag lookup
    tag_scores = PROP_NAME_TO_TAG_SCORES.get(prop_name_lower, {})
    if tag in tag_scores:
        score = max(score, tag_scores[tag])

    # Prop name substring match in class tokens
    for cls_token in class_tokens:
        if prop_name_lower in cls_token or cls_token in prop_name_lower:
            score = max(score, 0.80)

    # Class hint scoring
    hint_map = CLASS_HINT_SCORES.get(prop_name_lower, {})
    for cls_token in class_tokens:
        for hint, hint_score in hint_map.items():
            if hint in cls_token:
                score = max(score, hint_score)

    # ReactNode props accept anything at medium confidence
    if prop_type == "ReactNode" and score == 0.0:
        score = 0.65

    # Partial prop name match against tag
    if score == 0.0:
        if prop_name_lower in tag or tag in prop_name_lower:
            score = 0.50

    return min(1.0, score)


def _infer_content_type(node: dict, prop: PropDefinition) -> str:
    """Infer content type string for the mapping entry."""
    node_type = node["type"]
    prop_type = prop.type.strip()

    if node_type == "image_url":
        return "image_url"
    if node_type == "action":
        return "node"
    if prop_type == "ReactNode":
        return "node"
    if node_type == "list":
        return "node"
    if node_type == "link":
        return "text"
    return "text"


def _greedy_assign(
    score_matrix: list[list[float]],
    n_props: int,
    n_nodes: int,
) -> dict[int, int]:
    """Greedy 1:1 assignment: highest score first."""
    # Try scipy first
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        mat = np.zeros((n_props, max(n_props, n_nodes)))
        for i in range(n_props):
            for j in range(n_nodes):
                mat[i, j] = score_matrix[i][j]

        row_ind, col_ind = linear_sum_assignment(-mat)
        result = {}
        for r, c in zip(row_ind, col_ind):
            if c < n_nodes and mat[r, c] > 0.05:
                result[r] = c
        return result
    except ImportError:
        pass

    # Greedy fallback
    pairs = []
    for i in range(n_props):
        for j in range(n_nodes):
            pairs.append((score_matrix[i][j], i, j))
    pairs.sort(reverse=True)

    used_props: set[int] = set()
    used_nodes: set[int] = set()
    result: dict[int, int] = {}

    for score, prop_idx, node_idx in pairs:
        if prop_idx in used_props or node_idx in used_nodes:
            continue
        if score > 0.05:
            result[prop_idx] = node_idx
            used_props.add(prop_idx)
            used_nodes.add(node_idx)

    return result
