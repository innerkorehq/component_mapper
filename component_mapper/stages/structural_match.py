import asyncio
import logging
from component_mapper.models import (
    MappedComponent,
    MappingStage,
    MappingCacheRecord,
    RankedCandidate,
)
from component_mapper.cache.mapping_cache import MappingCache
from component_mapper.registry.signature_index import SignatureIndex
from component_mapper.registry.prop_mapper import infer_prop_mapping
from component_mapper.registry.astro_generator import generate_astro_component
from segment_classifier.models import ClassifiedSegment
from segment_classifier.utils.html_normalizer import normalize_segment

logger = logging.getLogger(__name__)


class StructuralMatchStage:
    def __init__(self, index: SignatureIndex, cache: MappingCache):
        self._index = index
        self._cache = cache

    async def process(
        self,
        segments: list[ClassifiedSegment],
    ) -> tuple[
        list[MappedComponent],
        list[tuple[ClassifiedSegment, list[RankedCandidate]]],
        list[ClassifiedSegment],
    ]:
        """Returns (direct_matches, ambiguous_with_candidates, novel)."""
        if not segments:
            return [], [], []

        cfg = self._index._settings.signature_index

        # Build batch items for index query
        items = []
        for seg in segments:
            normalized = normalize_segment(seg.raw_html, seg.text_content)
            items.append((seg.component_type, normalized, seg.fingerprint_hash))

        candidates_map = self._index.batch_get_candidates(items)

        direct: list[MappedComponent] = []
        ambiguous: list[tuple[ClassifiedSegment, list[RankedCandidate]]] = []
        novel: list[ClassifiedSegment] = []

        tasks = []
        task_segs = []
        for seg in segments:
            cands = candidates_map.get(seg.fingerprint_hash, [])
            tasks.append(self._classify_segment(seg, cands, cfg))
            task_segs.append(seg)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for seg, result in zip(task_segs, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Structural match error for %s: %s", seg.segment_id, result
                )
                novel.append(seg)
                continue

            outcome, data = result
            if outcome == "direct":
                direct.append(data)
            elif outcome == "ambiguous":
                ambiguous.append(data)
            else:
                novel.append(seg)

        logger.info(
            "Structural match: %d direct, %d ambiguous, %d novel",
            len(direct),
            len(ambiguous),
            len(novel),
        )
        return direct, ambiguous, novel

    async def _classify_segment(
        self,
        seg: ClassifiedSegment,
        candidates: list[RankedCandidate],
        cfg,
    ) -> tuple[str, any]:
        if not candidates:
            return "novel", None

        top = candidates[0]

        if top.composite_score >= cfg.direct_match_threshold:
            # Try clean direct match
            sig = top.signature
            prop_mapping = infer_prop_mapping(seg.raw_html, sig.props)

            if not prop_mapping.has_ambiguous:
                # Clean match — generate Astro and cache
                try:
                    astro = generate_astro_component(
                        seg, sig, prop_mapping, top.component_name
                    )
                except Exception as exc:
                    logger.debug(
                        "Astro gen failed for direct match %s: %s",
                        seg.segment_id,
                        exc,
                    )
                    return "ambiguous", (seg, candidates)

                mapped = MappedComponent(
                    segment_id=seg.segment_id,
                    page_url=seg.page_url,
                    component_type=seg.component_type,
                    classification_stage=seg.classification_stage,
                    component_name=top.component_name,
                    registry_source=top.registry_source,
                    mapping_stage=MappingStage.STRUCTURAL_MATCH,
                    mapping_confidence=top.composite_score,
                    prop_mapping=prop_mapping,
                    astro_component=astro,
                )

                await self._cache.set(
                    seg.fingerprint_hash,
                    MappingCacheRecord(
                        fingerprint_hash=seg.fingerprint_hash,
                        component_name=top.component_name,
                        registry_source=top.registry_source,
                        prop_mapping=prop_mapping,
                        mapping_stage=MappingStage.STRUCTURAL_MATCH,
                        confidence=top.composite_score,
                    ),
                )
                return "direct", mapped
            else:
                return "ambiguous", (seg, candidates)

        elif top.composite_score >= cfg.candidate_min_threshold:
            return "ambiguous", (seg, candidates)

        else:
            return "novel", None
