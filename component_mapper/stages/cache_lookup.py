import asyncio
import logging
from component_mapper.models import (
    MappedComponent,
    MappingStage,
    AstroComponent,
)
from component_mapper.cache.mapping_cache import MappingCache
from component_mapper.registry.signature_index import SignatureIndex
from component_mapper.registry.astro_generator import generate_astro_component
from segment_classifier.models import ClassifiedSegment

logger = logging.getLogger(__name__)


class CacheLookupStage:
    def __init__(self, cache: MappingCache, index: SignatureIndex):
        self._cache = cache
        self._index = index

    async def process(
        self,
        segments: list[ClassifiedSegment],
    ) -> tuple[list[MappedComponent], list[ClassifiedSegment]]:
        """Returns (cache_hits, cache_misses)."""
        tasks = [self._lookup(seg) for seg in segments]
        results = await asyncio.gather(*tasks)

        hits: list[MappedComponent] = []
        misses: list[ClassifiedSegment] = []
        for seg, result in zip(segments, results):
            if result is not None:
                hits.append(result)
            else:
                misses.append(seg)

        logger.info("Cache lookup: %d hits, %d misses", len(hits), len(misses))
        return hits, misses

    async def _lookup(self, segment: ClassifiedSegment) -> MappedComponent | None:
        record = await self._cache.get(segment.fingerprint_hash)
        if record is None:
            return None

        await self._cache.increment_hit(segment.fingerprint_hash)

        # Try to rebuild AstroComponent from cached signature
        sig = self._index.get_signature(record.component_name)

        if sig is None:
            # Build a minimal AstroComponent placeholder
            astro = _minimal_astro(record.component_name)
        else:
            try:
                astro = generate_astro_component(
                    segment, sig, record.prop_mapping, record.component_name
                )
            except Exception as exc:
                logger.debug(
                    "Failed to regenerate astro for cache hit %s: %s",
                    segment.segment_id,
                    exc,
                )
                astro = _minimal_astro(record.component_name)

        return MappedComponent(
            segment_id=segment.segment_id,
            page_url=segment.page_url,
            component_type=segment.component_type,
            classification_stage=segment.classification_stage,
            component_name=record.component_name,
            registry_source=record.registry_source,
            mapping_stage=MappingStage.CACHE_HIT,
            mapping_confidence=record.confidence,
            prop_mapping=record.prop_mapping,
            astro_component=astro,
        )


def _minimal_astro(component_name: str) -> AstroComponent:
    pascal = "".join(
        p.capitalize() for p in component_name.replace("-", "_").split("_")
    )
    content = f"---\n// {pascal} (from cache)\n---\n\n<{pascal} />\n"
    return AstroComponent(
        component_name=pascal,
        file_path=f"src/components/{pascal}.astro",
        frontmatter=f"// {pascal}",
        template=f"<{pascal} />",
        imports=[],
        full_file_content=content,
        install_commands=[],
    )
