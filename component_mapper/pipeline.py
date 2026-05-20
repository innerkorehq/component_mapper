import asyncio
import logging
from component_mapper.config import MapperSettings
from component_mapper.models import (
    PipelineRunResult,
    MappingStage,
)
from component_mapper.mcp.official_client import OfficialMCPClient
from component_mapper.mcp.registry_fetcher import RegistryFetcher
from component_mapper.registry.signature_index import SignatureIndex
from component_mapper.registry.custom_registry import CustomRegistry
from component_mapper.cache.mapping_cache import MappingCache
from component_mapper.stages.cache_lookup import CacheLookupStage
from component_mapper.stages.structural_match import StructuralMatchStage
from component_mapper.stages.llm_mapper import LLMMapperStage
from component_mapper.stages.astro_stage import AstroStage
from segment_classifier.models import ClassifiedSegment

logger = logging.getLogger(__name__)


class MapperPipeline:
    def __init__(self, settings: MapperSettings):
        self.settings = settings
        self.mcp_client = OfficialMCPClient(settings.mcp)
        self.fetcher = RegistryFetcher(settings.registry)
        self.custom_registry = CustomRegistry()
        self.mapping_cache = MappingCache(
            settings.mapping_cache.cache_path,
            settings.mapping_cache.auto_persist_every,
        )
        self.signature_index = SignatureIndex(settings, self.fetcher, self.mcp_client)

        self.cache_stage = CacheLookupStage(self.mapping_cache, self.signature_index)
        self.structural_stage = StructuralMatchStage(
            self.signature_index, self.mapping_cache
        )
        self.llm_stage = LLMMapperStage(
            settings, self.signature_index, self.custom_registry, self.mapping_cache
        )
        self.astro_stage = AstroStage(settings, self.signature_index)

    async def initialize(self) -> None:
        """One-time setup. Call before run()."""
        logger.info("Initializing MapperPipeline")

        # Connect MCP (non-fatal if unavailable)
        await self.mcp_client.connect()

        # Load caches
        await asyncio.gather(
            self.mapping_cache.load(),
            self.custom_registry.load(
                self.settings.signature_index.custom_registry_path
            ),
        )

        # Build signature index (uses cache if fresh, else rebuilds)
        await self.signature_index.build()

        # Merge custom components into index (priority: custom first)
        self.signature_index.merge_custom(self.custom_registry.get_all())

        logger.info(
            "Pipeline initialized: %d components in index, %d mapping cache records",
            len(self.signature_index.get_all_component_names()),
            self.mapping_cache.size,
        )

    async def run(self, segments: list[ClassifiedSegment]) -> PipelineRunResult:
        """Map all segments. Returns PipelineRunResult."""
        if not segments:
            return PipelineRunResult(
                total_segments=0,
                mapped=[],
                unresolved=[],
                stage_breakdown={s: 0 for s in MappingStage},
                llm_calls_made=0,
                llm_model_usage={},
                mcp_calls_made=self.mcp_client.calls_made,
                cache_hit_rate=0.0,
                structural_match_rate=0.0,
                install_commands=[],
                unique_components_used=[],
            )

        total = len(segments)
        logger.info("Starting pipeline run: %d segments", total)

        # Stage 1: Cache lookup
        hits, misses = await self.cache_stage.process(segments)

        # Stage 2: Structural match
        direct_matches, ambiguous, novel = await self.structural_stage.process(misses)

        # Stage 3: LLM mapping
        llm_mapped, unresolved = await self.llm_stage.process(ambiguous, novel)

        # Stage 4: Astro generation
        all_mapped = hits + direct_matches + llm_mapped
        all_mapped = await self.astro_stage.process(all_mapped)

        # Collect install manifest
        install_set: set[str] = set()
        component_names: set[str] = set()
        for comp in all_mapped:
            if comp.astro_component:
                for cmd in comp.astro_component.install_commands:
                    if cmd:
                        install_set.add(cmd)
            component_names.add(comp.component_name)

        # Stage breakdown
        breakdown: dict[MappingStage, int] = {s: 0 for s in MappingStage}
        for comp in all_mapped:
            breakdown[comp.mapping_stage] = breakdown.get(comp.mapping_stage, 0) + 1
        breakdown[MappingStage.UNRESOLVED] = len(unresolved)

        miss_count = len(misses)
        structural_match_rate = len(direct_matches) / miss_count if miss_count else 0.0

        logger.info(
            "Pipeline complete: %d mapped, %d unresolved | "
            "cache=%.0f%% structural=%.0f%% llm=%d calls",
            len(all_mapped),
            len(unresolved),
            (len(hits) / total * 100) if total else 0,
            structural_match_rate * 100,
            self.llm_stage.calls_made,
        )

        return PipelineRunResult(
            total_segments=total,
            mapped=all_mapped,
            unresolved=unresolved,
            stage_breakdown=breakdown,
            llm_calls_made=self.llm_stage.calls_made,
            llm_model_usage=self.llm_stage.get_model_usage(),
            mcp_calls_made=self.mcp_client.calls_made,
            cache_hit_rate=len(hits) / total if total else 0.0,
            structural_match_rate=structural_match_rate,
            install_commands=sorted(install_set),
            unique_components_used=sorted(component_names),
        )

    async def shutdown(self) -> None:
        """Persist state and install components. Call after run()."""
        logger.info("Shutting down pipeline")

        # Persist caches
        await asyncio.gather(
            self.mapping_cache.persist(),
            self.custom_registry.persist(
                self.settings.signature_index.custom_registry_path
            ),
        )

        # Install unique shadcn components via MCP
        unique_shadcn = [
            name
            for name in self.signature_index.get_all_component_names()
            if (
                self.signature_index.get_signature(name) is not None
                and self.signature_index.get_signature(name).registry_source.value
                == "shadcn"
            )
        ]

        if unique_shadcn and self.mcp_client._connected:
            try:
                results = await self.mcp_client.install_components(unique_shadcn)
                success_count = sum(1 for v in results.values() if v)
                logger.info(
                    "Installed %d/%d shadcn components",
                    success_count,
                    len(unique_shadcn),
                )
            except Exception as exc:
                logger.warning("Component install failed: %s", exc)

        await self.mcp_client.disconnect()
        logger.info("Pipeline shutdown complete")
