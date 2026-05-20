from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from segment_classifier.models import ClassifiedSegment, ComponentType


class MappingStage(str, Enum):
    CACHE_HIT = "cache_hit"
    STRUCTURAL_MATCH = "structural_match"
    LLM_MAPPED = "llm_mapped"
    LLM_NOVEL = "llm_novel"
    UNRESOLVED = "unresolved"


class InteractivityMode(str, Enum):
    STATIC = "static"
    INTERACTIVE = "interactive"
    PARTIAL = "partial"


class RegistrySource(str, Enum):
    SHADCN = "shadcn"
    CUSTOM = "custom"
    NOVEL = "novel"


class PropDefinition(BaseModel):
    name: str
    type: str
    required: bool = False
    default_value: str | None = None
    description: str = ""


class ComponentSignature(BaseModel):
    component_name: str
    registry_source: RegistrySource

    dom_skeleton: str
    root_element: str
    required_children: list[str]
    optional_children: list[str]
    structural_class_tokens: list[str]
    typical_nesting_depth: int
    child_tag_counts: dict[str, int]
    unique_tag_count: int

    compatible_component_types: list[ComponentType]
    interactivity: InteractivityMode = InteractivityMode.STATIC
    description: str = ""

    props: list[PropDefinition] = Field(default_factory=list)

    astro_import: str
    install_command: str
    requires_client_directive: bool = False


class CustomComponentDefinition(BaseModel):
    name: str
    dom_skeleton: str
    structural_class_tokens: list[str]
    compatible_component_types: list[ComponentType]
    props: list[PropDefinition]
    astro_import: str
    install_command: str = ""
    interactivity: InteractivityMode = InteractivityMode.STATIC
    description: str = ""
    source: str = "manual"
    confidence: float = 1.0


class RankedCandidate(BaseModel):
    component_name: str
    registry_source: RegistrySource
    signature: ComponentSignature

    structural_score: float = Field(ge=0.0, le=1.0)
    type_score: float = Field(ge=0.0, le=1.0)
    class_token_score: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)


class PropMapping(BaseModel):
    mappings: list[dict[str, Any]] = Field(default_factory=list)
    has_ambiguous: bool = False
    unmapped_props: list[str] = Field(default_factory=list)


class AstroImport(BaseModel):
    identifier: str
    source: str
    is_default: bool = False


class AstroComponent(BaseModel):
    component_name: str
    file_path: str
    frontmatter: str
    template: str
    imports: list[AstroImport]
    full_file_content: str
    install_commands: list[str]
    client_directive: str | None = None
    is_collection_item: bool = False


class ContentCollectionSchema(BaseModel):
    collection_name: str
    zod_schema: str
    example_entry: str


class MappingCacheRecord(BaseModel):
    fingerprint_hash: str
    component_name: str
    registry_source: RegistrySource
    prop_mapping: PropMapping
    mapping_stage: MappingStage
    confidence: float
    hit_count: int = 1


class MappedComponent(BaseModel):
    segment_id: str
    page_url: str
    component_type: ComponentType
    classification_stage: str

    component_name: str
    registry_source: RegistrySource
    mapping_stage: MappingStage
    mapping_confidence: float

    prop_mapping: PropMapping

    astro_component: AstroComponent
    content_collection_schema: ContentCollectionSchema | None = None

    llm_model_used: str | None = None
    llm_reasoning: str | None = None


class PipelineRunResult(BaseModel):
    total_segments: int
    mapped: list[MappedComponent]
    unresolved: list[ClassifiedSegment]

    stage_breakdown: dict[MappingStage, int]

    llm_calls_made: int
    llm_model_usage: dict[str, int]
    mcp_calls_made: int

    cache_hit_rate: float
    structural_match_rate: float

    install_commands: list[str]
    unique_components_used: list[str]
