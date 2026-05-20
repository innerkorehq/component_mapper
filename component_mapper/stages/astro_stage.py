import logging
from pathlib import Path
from component_mapper.config import MapperSettings
from component_mapper.models import MappedComponent, AstroComponent
from component_mapper.registry.signature_index import SignatureIndex
from component_mapper.registry.astro_generator import (
    generate_astro_component,
    generate_content_collection_schema,
    COLLECTION_TYPE_TO_NAME,
)

logger = logging.getLogger(__name__)


class AstroStage:
    def __init__(self, settings: MapperSettings, index: SignatureIndex):
        self._settings = settings
        self._index = index

    async def process(
        self,
        mapped: list[MappedComponent],
    ) -> list[MappedComponent]:
        """Enrich all MappedComponents with astro_component, write files to disk."""
        if not mapped:
            return []

        seen_astro: dict[str, AstroComponent] = {}
        enriched: list[MappedComponent] = []

        for component in mapped:
            if component.component_name in seen_astro:
                enriched.append(
                    component.model_copy(
                        update={"astro_component": seen_astro[component.component_name]}
                    )
                )
                continue

            sig = self._index.get_signature(component.component_name)
            if sig is not None:
                try:
                    from segment_classifier.models import ClassifiedSegment

                    seg = ClassifiedSegment(
                        segment_id=component.segment_id,
                        page_url=component.page_url,
                        component_type=component.component_type,
                        classification_stage=component.classification_stage,
                        fingerprint_hash=component.segment_id,
                        raw_html="",
                    )
                    astro = generate_astro_component(
                        seg, sig, component.prop_mapping, component.component_name
                    )
                    updated = component.model_copy(update={"astro_component": astro})
                    seen_astro[component.component_name] = astro
                    enriched.append(updated)
                    continue
                except Exception as exc:
                    logger.debug(
                        "Astro enrich failed for %s: %s", component.segment_id, exc
                    )

            enriched.append(component)
            seen_astro[component.component_name] = component.astro_component

        # Write to disk
        astro_root = self._settings.astro_project_root
        if astro_root:
            await self._write_files(enriched)

        if self._settings.generate_collection_schemas:
            await self._attach_collection_schemas(enriched)

        return enriched

    async def _write_files(self, mapped: list[MappedComponent]) -> None:
        astro_root = Path(self._settings.astro_project_root)
        written: set[str] = set()
        for component in mapped:
            if component.astro_component is None:
                continue
            file_path = astro_root / component.astro_component.file_path
            if str(file_path) in written:
                continue
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(component.astro_component.full_file_content)
                written.add(str(file_path))
                logger.debug("Wrote %s", file_path)
            except Exception as exc:
                logger.warning("Failed to write %s: %s", file_path, exc)

    async def _attach_collection_schemas(self, mapped: list[MappedComponent]) -> None:
        seen_collections: dict[str, any] = {}
        for component in mapped:
            ct = component.component_type
            if ct not in COLLECTION_TYPE_TO_NAME:
                continue
            collection_name = COLLECTION_TYPE_TO_NAME[ct]
            if collection_name in seen_collections:
                # Reuse existing schema
                try:
                    object.__setattr__(
                        component,
                        "content_collection_schema",
                        seen_collections[collection_name],
                    )
                except Exception:
                    pass
                continue
            sig = self._index.get_signature(component.component_name)
            if sig:
                try:
                    schema = generate_content_collection_schema(
                        ct, component.prop_mapping, sig
                    )
                    seen_collections[collection_name] = schema
                    object.__setattr__(component, "content_collection_schema", schema)
                except Exception as exc:
                    logger.debug("Schema gen failed: %s", exc)
