import re
from dataclasses import dataclass, field
from component_mapper.models import PropDefinition, InteractivityMode


STRUCTURAL_CLASS_PATTERN = re.compile(
    r"\b(card|grid|list|item|hero|nav|menu|header|footer|sidebar|"
    r"form|modal|badge|price|rating|carousel|pagination|search|"
    r"feature|testimonial|cta|faq|pricing|article|media|table|"
    r"product|blog|news|collection|section|widget)\b",
    re.IGNORECASE,
)

PROPS_INTERFACE_PATTERN = re.compile(
    r"(?:interface\s+\w*Props\w*|type\s+\w*Props\w*\s*=)\s*\{([^}]+)\}", re.DOTALL
)
INTERACTIVE_HOOKS_PATTERN = re.compile(
    r"\b(useState|useEffect|onClick|onChange|useRef|useCallback|useReducer)\b"
)
PARTIAL_INTERACTIVE_PATTERN = re.compile(r"\b(onMouseEnter|onFocus|onBlur|onHover)\b")
RETURN_BLOCK_PATTERN = re.compile(r"\breturn\s*\((.*?)\)\s*[;]?\s*\}", re.DOTALL)
RETURN_BLOCK_ALT_PATTERN = re.compile(
    r"\breturn\s*(<[A-Za-z].*?)(?=\n\s*\})", re.DOTALL
)
CONDITIONAL_COMPONENT_PATTERN = re.compile(r"\{[^}]*&&\s*<([A-Z][A-Za-z0-9]*)")
TERNARY_COMPONENT_PATTERN = re.compile(r"\?[^:]*<([A-Z][A-Za-z0-9]*)[^:]*:")
CLASSNAME_PATTERN = re.compile(r'className=["\']([^"\']+)["\']')


@dataclass
class ParsedSource:
    dom_skeleton: str = ""
    root_element: str = "div"
    required_children: list[str] = field(default_factory=list)
    optional_children: list[str] = field(default_factory=list)
    structural_class_tokens: list[str] = field(default_factory=list)
    typical_nesting_depth: int = 0
    child_tag_counts: dict[str, int] = field(default_factory=dict)
    interactivity: InteractivityMode = InteractivityMode.STATIC
    props: list[PropDefinition] = field(default_factory=list)


def parse_source(source_code: str) -> ParsedSource:
    """Parse TypeScript/TSX source into ParsedSource. Gracefully handles malformed input."""
    result = ParsedSource()
    try:
        jsx_block = _extract_return_block(source_code)
        if jsx_block:
            result.dom_skeleton = _parse_jsx_skeleton(jsx_block, max_depth=5)
            result.root_element = _extract_root_element(jsx_block)
            result.typical_nesting_depth = _measure_nesting_depth(jsx_block)
            result.child_tag_counts = _count_child_tags(jsx_block)
            result.required_children, result.optional_children = _extract_children(
                jsx_block
            )

        all_classnames = CLASSNAME_PATTERN.findall(source_code)
        tokens = set()
        for cls_string in all_classnames:
            for token in STRUCTURAL_CLASS_PATTERN.findall(cls_string):
                tokens.add(token.lower())
        result.structural_class_tokens = sorted(tokens)

        result.interactivity = _detect_interactivity(source_code)
        result.props = _extract_props(source_code)
    except Exception:
        pass
    return result


def _extract_return_block(source: str) -> str:
    """Extract the JSX return() block from a component function."""
    m = RETURN_BLOCK_PATTERN.search(source)
    if m:
        return m.group(1).strip()
    m = RETURN_BLOCK_ALT_PATTERN.search(source)
    if m:
        return m.group(1).strip()
    # Fallback: find any JSX-looking block
    jsx_start = re.search(r"<[A-Z][A-Za-z]*|<[a-z]+[\s>/]", source)
    if jsx_start:
        return source[jsx_start.start() :]
    return ""


def _extract_root_element(jsx_block: str) -> str:
    """Extract outermost JSX element tag name."""
    m = re.match(r"<([A-Za-z][A-Za-z0-9]*)", jsx_block.strip())
    if m:
        return m.group(1).lower()
    return "div"


def _parse_jsx_skeleton(jsx: str, max_depth: int) -> str:
    """Recursively parse JSX into skeleton string."""
    # Build token stream
    tokens = []
    # Use (?:[^>/]|/(?!>))* so the slash in /> is captured by group 4, not group 3
    for m in re.finditer(r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:[^>/]|/(?!>))*)(/?)>", jsx):
        is_close = m.group(1) == "/"
        tag = m.group(2)
        # attrs = m.group(3) - Unused

        is_self_close = m.group(4) == "/" or tag.lower() in (
            "input",
            "img",
            "br",
            "hr",
            "meta",
            "link",
        )

        if is_close:
            tokens.append(("close", tag, m.start()))
        elif is_self_close:
            tokens.append(("self", tag, m.start()))
        else:
            tokens.append(("open", tag, m.start()))

    if not tokens:
        return ""

    def build_tree(pos: int, depth: int) -> tuple[str, int]:
        if pos >= len(tokens) or depth > max_depth:
            return "", pos

        kind, tag, _ = tokens[pos]
        tag_lower = tag.lower()

        if kind == "close":
            return "", pos

        if kind == "self":
            return tag_lower, pos + 1

        # kind == 'open'
        children = []
        i = pos + 1
        while i < len(tokens):
            k, t, _ = tokens[i]
            if k == "close" and t.lower() == tag_lower:
                i += 1
                break
            prev_i = i
            child_str, i = build_tree(i, depth + 1)
            if i == prev_i:      # build_tree didn't advance (mismatched close tag)
                i += 1           # skip token to prevent infinite loop
            if child_str:
                children.append(child_str)

        if not children:
            return tag_lower, i
        elif len(children) == 1:
            return f"{tag_lower}>{children[0]}", i
        else:
            return f"{tag_lower}>[{'+'.join(children)}]", i

    result, _ = build_tree(0, 0)
    return result


def _measure_nesting_depth(jsx: str) -> int:
    """Count max nesting depth of JSX tags."""
    depth = 0
    max_depth = 0
    for m in re.finditer(r"<(/?)([A-Za-z][A-Za-z0-9]*)([^>]*)(/?)>", jsx):
        is_close = m.group(1) == "/"
        is_self = m.group(4) == "/" or m.group(2).lower() in (
            "input",
            "img",
            "br",
            "hr",
        )
        if is_close:
            depth = max(0, depth - 1)
        elif not is_self:
            depth += 1
            max_depth = max(max_depth, depth)
    return max_depth


def _count_child_tags(jsx: str) -> dict[str, int]:
    """Count occurrences of each lowercase HTML tag."""
    counts: dict[str, int] = {}
    html_tags = {
        "div",
        "span",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "a",
        "img",
        "button",
        "input",
        "form",
        "ul",
        "ol",
        "li",
        "table",
        "tr",
        "td",
        "th",
        "section",
        "article",
        "nav",
        "header",
        "footer",
        "main",
        "aside",
        "figure",
        "figcaption",
        "label",
        "select",
        "textarea",
        "picture",
        "source",
    }
    for m in re.finditer(r"<([A-Za-z][A-Za-z0-9]*)", jsx):
        tag = m.group(1).lower()
        if tag in html_tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def _extract_children(jsx: str) -> tuple[list[str], list[str]]:
    """Extract required and optional sub-component children."""
    # All capitalized component refs
    all_components = set(re.findall(r"<([A-Z][A-Za-z0-9]+)", jsx))

    # Optional: inside conditional expressions
    optional_set = set()
    for m in CONDITIONAL_COMPONENT_PATTERN.finditer(jsx):
        optional_set.add(m.group(1))
    for m in TERNARY_COMPONENT_PATTERN.finditer(jsx):
        optional_set.add(m.group(1))

    required = sorted(all_components - optional_set)
    optional = sorted(optional_set & all_components)
    return required, optional


def _extract_props(source: str) -> list[PropDefinition]:
    """Parse Props interface/type into PropDefinition list."""
    m = PROPS_INTERFACE_PATTERN.search(source)
    if not m:
        return []

    body = m.group(1)
    props = []

    # Split on both newlines and semicolons to handle single-line interfaces
    raw_fields = re.split(r"[;\n]", body)
    for raw_field in raw_fields:
        raw_field = raw_field.strip()
        if not raw_field or raw_field.startswith("//") or raw_field.startswith("*"):
            continue

        # Match: propName?: Type  or  propName: Type
        pm = re.match(r"(\w+)(\?)?\s*:\s*(.+?)(?:,)?\s*$", raw_field)
        if not pm:
            continue

        name = pm.group(1)
        optional = pm.group(2) == "?"
        type_str = pm.group(3).strip().rstrip(",")

        # Extract default from JSDoc or inline comment
        default_val = None
        default_m = re.search(r"@default\s+(\S+)", raw_field)
        if default_m:
            default_val = default_m.group(1)

        props.append(
            PropDefinition(
                name=name,
                type=type_str,
                required=not optional,
                default_value=default_val,
                description="",
            )
        )

    return props


def _detect_interactivity(source: str) -> InteractivityMode:
    """Check for React hooks/event handlers to determine interactivity."""
    if INTERACTIVE_HOOKS_PATTERN.search(source):
        return InteractivityMode.INTERACTIVE
    if PARTIAL_INTERACTIVE_PATTERN.search(source):
        return InteractivityMode.PARTIAL
    return InteractivityMode.STATIC
