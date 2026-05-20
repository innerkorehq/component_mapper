import re
import logging
from component_mapper.models import (
    AstroComponent,
    ContentCollectionSchema,
    AstroImport,
    PropMapping,
    ComponentSignature,
    RegistrySource,
    InteractivityMode,
    PropDefinition,
)
from segment_classifier.models import ClassifiedSegment, ComponentType

logger = logging.getLogger(__name__)

COLLECTION_TYPE_TO_NAME: dict[ComponentType, str] = {
    ComponentType.COLLECTION_PRODUCT_CARD: "products",
    ComponentType.COLLECTION_PRODUCT_LIST: "products",
    ComponentType.COLLECTION_BLOG_CARD: "blog",
    ComponentType.COLLECTION_BLOG_LIST: "blog",
    ComponentType.COLLECTION_NEWS_ITEM: "news",
    ComponentType.COLLECTION_NEWS_LIST: "news",
}

FILE_PATH_PREFIX: dict[str, str] = {
    "layout": "src/components/layout",
    "collection": "src/components/collection",
    "section": "src/components/sections",
    "ui": "src/components/ui",
    "content": "src/components/content",
}

ZOD_TYPE_MAP: dict[str, str] = {
    "string": "z.string()",
    "number": "z.number()",
    "boolean": "z.boolean()",
    "ReactNode": "z.string()",
    "any": "z.any()",
}


def generate_astro_component(
    segment: ClassifiedSegment,
    signature: ComponentSignature,
    prop_mapping: PropMapping,
    component_name: str,
) -> AstroComponent:
    """Generate a complete .astro wrapper file."""
    pascal_name = _to_pascal_case(component_name)
    file_path = _compute_file_path(segment.component_type, pascal_name)

    is_collection = segment.component_type.value.startswith("collection.")

    # Client directive
    client_directive: str | None = None
    if signature.interactivity == InteractivityMode.INTERACTIVE:
        client_directive = "load"
    elif signature.interactivity == InteractivityMode.PARTIAL:
        client_directive = "visible"

    # Build imports
    imports, import_stmts = _build_imports(signature, component_name)

    # Build Props interface
    props_interface = _build_props_interface(signature.props)

    # Build props destructure
    props_destructure = _build_props_destructure(signature.props)

    # Build template
    template = _build_template(
        signature=signature,
        prop_mapping=prop_mapping,
        component_name=component_name,
        client_directive=client_directive,
    )

    # Assemble frontmatter
    frontmatter_parts = []
    for stmt in import_stmts:
        frontmatter_parts.append(stmt)
    if import_stmts:
        frontmatter_parts.append("")
    if props_interface:
        frontmatter_parts.append(props_interface)
        frontmatter_parts.append("")
    if props_destructure:
        frontmatter_parts.append(props_destructure)
    frontmatter = "\n".join(frontmatter_parts)

    # Assemble full file
    full_content = f"---\n{frontmatter}\n---\n\n{template}\n"

    install_commands = [signature.install_command] if signature.install_command else []

    return AstroComponent(
        component_name=pascal_name,
        file_path=file_path,
        frontmatter=frontmatter,
        template=template,
        imports=imports,
        full_file_content=full_content,
        install_commands=install_commands,
        client_directive=client_directive,
        is_collection_item=is_collection,
    )


def _compute_file_path(component_type: ComponentType, pascal_name: str) -> str:
    prefix_key = component_type.value.split(".")[0]
    prefix = FILE_PATH_PREFIX.get(prefix_key, "src/components")
    return f"{prefix}/{pascal_name}.astro"


def _build_imports(
    signature: ComponentSignature,
    component_name: str,
) -> tuple[list[AstroImport], list[str]]:
    imports: list[AstroImport] = []
    stmts: list[str] = []

    if signature.registry_source == RegistrySource.CUSTOM:
        # Custom: single default import from .astro file
        astro_import_path = signature.astro_import
        pascal = _to_pascal_case(component_name)
        imports.append(
            AstroImport(identifier=pascal, source=astro_import_path, is_default=True)
        )
        stmts.append(f'import {pascal} from "{astro_import_path}";')
    else:
        # Shadcn: named imports of component + required children
        source_path = f"@/components/ui/{component_name}"
        all_names = [_to_pascal_case(component_name)] + [
            _to_pascal_case(c) for c in signature.required_children
        ]
        # Deduplicate, preserve order
        seen: set[str] = set()
        unique_names = []
        for n in all_names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)

        for name in unique_names:
            imports.append(AstroImport(identifier=name, source=source_path))

        if unique_names:
            names_str = ", ".join(unique_names)
            stmts.append(f'import {{ {names_str} }} from "{source_path}";')

    return imports, stmts


def _build_props_interface(props: list[PropDefinition]) -> str:
    if not props:
        return ""
    lines = ["interface Props {"]
    for prop in props:
        optional = "" if prop.required else "?"
        ts_type = _to_ts_type(prop.type)
        lines.append(f"  {prop.name}{optional}: {ts_type};")
    lines.append("}")
    return "\n".join(lines)


def _build_props_destructure(props: list[PropDefinition]) -> str:
    if not props:
        return ""
    parts = []
    for prop in props:
        if prop.default_value is not None:
            default = prop.default_value
            # Quote strings
            if prop.type == "string" and not default.startswith('"'):
                default = f'"{default}"'
            parts.append(f"{prop.name} = {default}")
        else:
            parts.append(prop.name)
    return f"const {{ {', '.join(parts)} }} = Astro.props;"


def _build_template(
    signature: ComponentSignature,
    prop_mapping: PropMapping,
    component_name: str,
    client_directive: str | None,
) -> str:
    pascal = _to_pascal_case(component_name)

    # Build a prop → segment_field lookup for template rendering
    prop_to_field: dict[str, dict] = {}
    for mapping in prop_mapping.mappings:
        prop_name = mapping.get("component_prop", "")
        prop_to_field[prop_name] = mapping

    directive_attr = f" client:{client_directive}" if client_directive else ""

    if signature.registry_source == RegistrySource.CUSTOM:
        return _build_custom_template(pascal, signature, prop_to_field, directive_attr)

    return _build_shadcn_template(
        pascal, signature, prop_to_field, directive_attr, prop_mapping
    )


def _build_shadcn_template(
    pascal: str,
    signature: ComponentSignature,
    prop_to_field: dict[str, dict],
    directive_attr: str,
    prop_mapping: PropMapping,
) -> str:
    lines = [f"<{pascal}{directive_attr}>"]

    # Use required children to structure the template
    children = signature.required_children
    if not children:
        # Flat: render props directly
        for prop_name, mapping in prop_to_field.items():
            ambiguous_comment = (
                " {/* TODO: verify mapping */}" if mapping.get("ambiguous") else ""
            )
            content_type = mapping.get("type", "text")
            if content_type == "image_url":
                lines.append(
                    f'  <img src={{{prop_name}}} alt="" class="w-full object-cover" />'
                )
            else:
                lines.append(f"  <span>{{{prop_name}}}{ambiguous_comment}</span>")
        for unmapped in prop_mapping.unmapped_props:
            lines.append(f"  {{/* {unmapped} */}}")
    else:
        # Structured: distribute props across sub-components
        prop_names = list(prop_to_field.keys())
        chunk_size = max(1, len(prop_names) // max(1, len(children)))

        for i, child_name in enumerate(children):
            child_pascal = _to_pascal_case(child_name)
            chunk = prop_names[i * chunk_size : (i + 1) * chunk_size]
            if i == len(children) - 1:
                chunk = prop_names[i * chunk_size :]

            lines.append(f"  <{child_pascal}>")
            for prop_name in chunk:
                mapping = prop_to_field[prop_name]
                ambiguous_comment = (
                    " {/* TODO: verify mapping */}" if mapping.get("ambiguous") else ""
                )
                content_type = mapping.get("type", "text")
                if content_type == "image_url":
                    lines.append(
                        f'    <img src={{{prop_name}}} alt="" class="w-full h-48 object-cover rounded-t-lg" />'
                    )
                elif prop_name in ("footer", "action"):
                    lines.append(
                        f'    <button class="w-full">{{{prop_name}}}{ambiguous_comment}</button>'
                    )
                else:
                    lines.append(
                        f'    <span class="block">{{{prop_name}}}{ambiguous_comment}</span>'
                    )
            for unmapped in (
                prop_mapping.unmapped_props if i == len(children) - 1 else []
            ):
                lines.append(f"    {{/* {unmapped} */}}")
            lines.append(f"  </{child_pascal}>")

    lines.append(f"</{pascal}>")
    return "\n".join(lines)


def _build_custom_template(
    pascal: str,
    signature: ComponentSignature,
    prop_to_field: dict[str, dict],
    directive_attr: str,
) -> str:
    lines = [f"<{pascal}{directive_attr}>"]
    for prop_name, mapping in prop_to_field.items():
        content_type = mapping.get("type", "text")
        if content_type == "image_url":
            lines.append(f'  <img slot="{prop_name}" src={{{prop_name}}} alt="" />')
        else:
            lines.append(f'  <span slot="{prop_name}">{{{prop_name}}}</span>')
    lines.append(f"</{pascal}>")
    return "\n".join(lines)


def generate_content_collection_schema(
    component_type: ComponentType,
    prop_mapping: PropMapping,
    signature: ComponentSignature,
) -> ContentCollectionSchema:
    """Generate Zod schema for Astro Content Collections."""
    collection_name = COLLECTION_TYPE_TO_NAME.get(component_type, "items")

    zod_fields = []
    for prop in signature.props:
        field_str = _prop_to_zod(prop)
        zod_fields.append(f"    {prop.name}: {field_str},")

    fields_block = "\n".join(zod_fields) if zod_fields else "    // no props defined"

    zod_schema = f"""import {{ defineCollection, z }} from 'astro:content';

const {collection_name} = defineCollection({{
  type: 'data',
  schema: z.object({{
{fields_block}
  }}),
}});

export const collections = {{ {collection_name} }};"""

    # Build example entry
    example_lines = ["{"]
    for prop in signature.props:
        example_val = _example_value(prop)
        example_lines.append(f'  "{prop.name}": {example_val},')
    example_lines.append("}")
    example_entry = "\n".join(example_lines)

    return ContentCollectionSchema(
        collection_name=collection_name,
        zod_schema=zod_schema,
        example_entry=example_entry,
    )


def _prop_to_zod(prop: PropDefinition) -> str:
    """Convert PropDefinition to Zod schema string."""
    base = ZOD_TYPE_MAP.get(prop.type, "z.string()")

    # URL hint
    if (
        "image" in prop.name.lower()
        or "src" in prop.name.lower()
        or "url" in prop.name.lower()
    ):
        if "string" in prop.type:
            base = "z.string().url()"

    if not prop.required:
        if prop.default_value is not None:
            dv = prop.default_value
            if prop.type == "string":
                dv = f'"{dv}"'
            base = f"{base}.default({dv})"
        else:
            base = f"{base}.optional()"

    return base


def _example_value(prop: PropDefinition) -> str:
    if prop.default_value:
        v = prop.default_value
        if prop.type == "string":
            return f'"{v}"'
        return v
    if "image" in prop.name.lower() or "src" in prop.name.lower():
        return '"https://example.com/image.jpg"'
    if "url" in prop.name.lower() or "href" in prop.name.lower():
        return '"https://example.com"'
    if prop.type == "number":
        return "0"
    if prop.type == "boolean":
        return "false"
    return f'"{prop.name} value"'


def _to_pascal_case(name: str) -> str:
    """'product-card' -> 'ProductCard'. 'CardHeader' -> 'CardHeader'. 'card' -> 'Card'."""
    # Already PascalCase (starts uppercase and no delimiters) — preserve as-is
    if name and name[0].isupper() and not re.search(r"[-_\s]", name):
        return name
    return "".join(part.capitalize() for part in re.split(r"[-_\s]+", name) if part)


def _to_ts_type(prop_type: str) -> str:
    """Map Pydantic type strings to TypeScript types."""
    mapping = {
        "string": "string",
        "number": "number",
        "boolean": "boolean",
        "ReactNode": "astro.ComponentProps<any>",
        "any": "unknown",
    }
    return mapping.get(prop_type.strip(), prop_type.strip())
