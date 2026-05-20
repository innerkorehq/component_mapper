import asyncio
import json
import logging
import os
import litellm
from component_mapper.config import MapperSettings
from component_mapper.models import (
    MappedComponent,
    MappingStage,
    MappingCacheRecord,
    RankedCandidate,
    PropMapping,
    CustomComponentDefinition,
    InteractivityMode,
    PropDefinition,
)
from component_mapper.cache.mapping_cache import MappingCache
from component_mapper.registry.signature_index import SignatureIndex
from component_mapper.registry.custom_registry import CustomRegistry
from component_mapper.registry.prop_mapper import (
    infer_prop_mapping,
    _extract_content_nodes,
)
from component_mapper.registry.astro_generator import generate_astro_component
from segment_classifier.models import ClassifiedSegment, ComponentType
from segment_classifier.utils.html_normalizer import normalize_segment

logger = logging.getLogger(__name__)


class LLMMapperStage:
    def __init__(
        self,
        settings: MapperSettings,
        index: SignatureIndex,
        custom_registry: CustomRegistry,
        cache: MappingCache,
    ):
        self._settings = settings
        self._index = index
        self._custom_registry = custom_registry
        self._cache = cache
        self.calls_made = 0
        self._model_usage: dict[str, int] = {}
        self._sem = asyncio.Semaphore(settings.litellm.max_concurrent_batches)
        self._llm_defaults: dict = {}
        self._setup_litellm()

    def _setup_litellm(self) -> None:
        cfg = self._settings.litellm

        # Universal API key: read once, set on litellm module so all providers can use it
        api_key = os.environ.get(cfg.api_key_env, "")
        if api_key:
            litellm.api_key = api_key
            logger.debug("LiteLLM api_key set from env %s", cfg.api_key_env)
        else:
            logger.debug(
                "Env %s not set — relying on provider-specific env vars",
                cfg.api_key_env,
            )

        # Load config file defaults (JSON, optional)
        if cfg.config_path:
            try:
                self._llm_defaults = litellm.read_config_args(cfg.config_path)
                logger.info(
                    "Loaded LiteLLM config from %s: %s",
                    cfg.config_path,
                    list(self._llm_defaults),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load LiteLLM config %s: %s", cfg.config_path, exc
                )

    def get_model_usage(self) -> dict[str, int]:
        return dict(self._model_usage)

    def _select_model(
        self,
        candidates: list[RankedCandidate],
        is_novel: bool,
        max_prop_count: int,
    ) -> str:
        cfg = self._settings.model_routing
        if is_novel:
            return cfg.standard_model
        if len(candidates) > cfg.complex_candidate_threshold:
            return cfg.complex_model
        if (
            len(candidates) <= cfg.fast_max_candidates
            and max_prop_count <= cfg.fast_max_props
        ):
            return cfg.fast_model
        return cfg.standard_model

    async def process(
        self,
        ambiguous: list[tuple[ClassifiedSegment, list[RankedCandidate]]],
        novel: list[ClassifiedSegment],
    ) -> tuple[list[MappedComponent], list[ClassifiedSegment]]:
        """Returns (mapped, still_unresolved)."""
        mapped: list[MappedComponent] = []
        unresolved: list[ClassifiedSegment] = []

        # Process ambiguous
        if ambiguous:
            batch_size = self._settings.litellm.batch_size
            batches = [
                ambiguous[i : i + batch_size]
                for i in range(0, len(ambiguous), batch_size)
            ]
            for batch in batches:
                results = await asyncio.gather(
                    *[self._process_ambiguous(seg, cands) for seg, cands in batch],
                    return_exceptions=True,
                )
                for (seg, _), result in zip(batch, results):
                    if isinstance(result, Exception) or result is None:
                        logger.warning(
                            "LLM mapping failed for %s: %s", seg.segment_id, result
                        )
                        unresolved.append(seg)
                    else:
                        mapped.append(result)

        # Process novel
        if novel:
            batch_size = self._settings.litellm.batch_size
            batches = [
                novel[i : i + batch_size] for i in range(0, len(novel), batch_size)
            ]
            for batch in batches:
                results = await asyncio.gather(
                    *[self._process_novel(seg) for seg in batch],
                    return_exceptions=True,
                )
                for seg, result in zip(batch, results):
                    if isinstance(result, Exception) or result is None:
                        logger.warning(
                            "Novel LLM mapping failed for %s: %s",
                            seg.segment_id,
                            result,
                        )
                        unresolved.append(seg)
                    else:
                        mapped.append(result)

        logger.info(
            "LLM stage: %d mapped, %d unresolved, %d LLM calls",
            len(mapped),
            len(unresolved),
            self.calls_made,
        )
        return mapped, unresolved

    async def _process_ambiguous(
        self,
        seg: ClassifiedSegment,
        candidates: list[RankedCandidate],
    ) -> MappedComponent | None:
        normalized = normalize_segment(seg.raw_html, seg.text_content)

        try:
            content_nodes = _extract_content_nodes(seg.raw_html)
        except Exception:
            content_nodes = []

        # Find ambiguous prop mappings from best candidate
        ambiguous_mappings = []
        if candidates:
            best_sig = candidates[0].signature
            pm = infer_prop_mapping(seg.raw_html, best_sig.props)
            ambiguous_mappings = [
                {
                    "segment_field": m["segment_field"],
                    "candidates": [m["component_prop"]],
                }
                for m in pm.mappings
                if m.get("ambiguous")
            ]

        max_prop_count = max((len(c.signature.props) for c in candidates), default=0)
        model = self._select_model(candidates, False, max_prop_count)

        prompt_user = json.dumps(
            {
                "segment_id": seg.segment_id,
                "component_type": seg.component_type.value,
                "dom_skeleton": normalized.skeleton,
                "class_tokens": normalized.class_tokens,
                "sibling_count": getattr(seg, "sibling_count", 0),
                "candidates": [
                    {
                        "name": c.component_name,
                        "score": c.composite_score,
                        "props": [p.model_dump() for p in c.signature.props],
                        "dom_skeleton": c.signature.dom_skeleton,
                    }
                    for c in candidates
                ],
                "content_nodes": [
                    {
                        "tag": n["tag"],
                        "selector": n["selector"],
                        "type": n["type"],
                    }
                    for n in content_nodes[:10]
                ],
                "ambiguous_mappings": ambiguous_mappings,
            },
            indent=2,
        )

        response = await self._call_llm(
            model=model,
            system=(
                "You are a UI component mapper. Given an HTML segment and candidate "
                "Shadcn components, select the best match and provide prop mappings. "
                "Output ONLY valid JSON. No markdown, no explanation."
            ),
            user=prompt_user,
        )

        if response is None:
            return None

        return self._build_mapped_ambiguous(seg, response, candidates, model)

    async def _process_novel(self, seg: ClassifiedSegment) -> MappedComponent | None:
        normalized = normalize_segment(seg.raw_html, seg.text_content)

        try:
            content_nodes = _extract_content_nodes(seg.raw_html)
        except Exception:
            content_nodes = []

        model = self._settings.model_routing.standard_model

        prompt_user = json.dumps(
            {
                "segment_id": seg.segment_id,
                "component_type": seg.component_type.value,
                "dom_skeleton": normalized.skeleton,
                "class_tokens": normalized.class_tokens,
                "sibling_count": getattr(seg, "sibling_count", 0),
                "content_nodes": [
                    {
                        "tag": n["tag"],
                        "selector": n["selector"],
                        "type": n["type"],
                    }
                    for n in content_nodes[:10]
                ],
                "task": (
                    "Define a new custom component for this segment. "
                    "Return a JSON object with keys: segment_id, custom_component "
                    "(name, dom_skeleton, structural_class_tokens, compatible_component_types, "
                    "props, interactivity, description), prop_mapping, confidence, reasoning."
                ),
            },
            indent=2,
        )

        response = await self._call_llm(
            model=model,
            system=(
                "You are a UI component designer. Given an HTML segment, define a new "
                "custom component and its prop mapping. "
                "Output ONLY valid JSON. No markdown, no explanation."
            ),
            user=prompt_user,
        )

        if response is None:
            return None

        return self._build_mapped_novel(seg, response, model)

    async def _call_llm(self, model: str, system: str, user: str) -> dict | None:
        # Config file model takes priority; routing model is the fallback
        resolved_model = self._llm_defaults.get("model", model)
        async with self._sem:
            try:
                self.calls_made += 1
                self._model_usage[resolved_model] = self._model_usage.get(resolved_model, 0) + 1

                call_kwargs: dict = {
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    **self._llm_defaults,
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }

                resp = await asyncio.wait_for(
                    litellm.acompletion(**call_kwargs),
                    timeout=self._settings.litellm.timeout_seconds,
                )

                content = resp.choices[0].message.content or ""
                content = content.strip()
                if content.startswith("```"):
                    content = "\n".join(content.split("\n")[1:])
                if content.endswith("```"):
                    content = "\n".join(content.split("\n")[:-1])

                return json.loads(content)

            except asyncio.TimeoutError:
                logger.warning("LLM call timed out for model %s", model)
                return None
            except json.JSONDecodeError as exc:
                logger.warning("LLM returned invalid JSON from %s: %s", model, exc)
                return None
            except Exception as exc:
                logger.warning("LLM call failed (%s): %s", model, exc)
                return None

    def _build_mapped_ambiguous(
        self,
        seg: ClassifiedSegment,
        response: dict,
        candidates: list[RankedCandidate],
        model: str,
    ) -> MappedComponent | None:
        selected = response.get("selected_component", "")
        confidence = float(response.get("confidence", 0.5))
        prop_mapping_raw = response.get("prop_mapping", [])
        reasoning = response.get("reasoning", "")

        # Find signature
        sig = self._index.get_signature(selected)
        if sig is None and candidates:
            sig = candidates[0].signature
            selected = candidates[0].component_name

        if sig is None:
            return None

        # Build PropMapping from LLM response
        mappings = []
        for m in prop_mapping_raw:
            mappings.append(
                {
                    "segment_field": m.get("segment_field", ""),
                    "component_prop": m.get("component_prop", ""),
                    "type": "text",
                    "confidence": float(m.get("confidence", 0.7)),
                    "ambiguous": float(m.get("confidence", 0.7)) < 0.70,
                }
            )
        prop_mapping = PropMapping(
            mappings=mappings,
            has_ambiguous=any(m["ambiguous"] for m in mappings),
        )

        try:
            astro = generate_astro_component(seg, sig, prop_mapping, selected)
        except Exception as exc:
            logger.debug("Astro gen failed for LLM-mapped %s: %s", seg.segment_id, exc)
            from component_mapper.stages.cache_lookup import _minimal_astro

            astro = _minimal_astro(selected)

        # Cache result
        asyncio.create_task(
            self._cache.set(
                seg.fingerprint_hash,
                MappingCacheRecord(
                    fingerprint_hash=seg.fingerprint_hash,
                    component_name=selected,
                    registry_source=sig.registry_source,
                    prop_mapping=prop_mapping,
                    mapping_stage=MappingStage.LLM_MAPPED,
                    confidence=confidence,
                ),
            )
        )

        return MappedComponent(
            segment_id=seg.segment_id,
            page_url=seg.page_url,
            component_type=seg.component_type,
            classification_stage=seg.classification_stage,
            component_name=selected,
            registry_source=sig.registry_source,
            mapping_stage=MappingStage.LLM_MAPPED,
            mapping_confidence=confidence,
            prop_mapping=prop_mapping,
            astro_component=astro,
            llm_model_used=model,
            llm_reasoning=reasoning,
        )

    def _build_mapped_novel(
        self,
        seg: ClassifiedSegment,
        response: dict,
        model: str,
    ) -> MappedComponent | None:
        custom_raw = response.get("custom_component", {})
        if not custom_raw:
            return None

        confidence = float(response.get("confidence", 0.6))
        prop_mapping_raw = response.get("prop_mapping", [])
        reasoning = response.get("reasoning", "")

        # Build props
        props = []
        for p in custom_raw.get("props", []):
            props.append(
                PropDefinition(
                    name=p.get("name", "prop"),
                    type=p.get("type", "string"),
                    required=p.get("required", False),
                    default_value=p.get("default_value"),
                )
            )

        # Parse compatible types
        compat_types = []
        for ct_str in custom_raw.get("compatible_component_types", []):
            try:
                compat_types.append(ComponentType(ct_str))
            except ValueError:
                pass

        interactivity = InteractivityMode.STATIC
        try:
            interactivity = InteractivityMode(custom_raw.get("interactivity", "static"))
        except ValueError:
            pass

        defn = CustomComponentDefinition(
            name=custom_raw.get("name", f"custom-{seg.segment_id[:8]}"),
            dom_skeleton=custom_raw.get("dom_skeleton", "div"),
            structural_class_tokens=custom_raw.get("structural_class_tokens", []),
            compatible_component_types=compat_types,
            props=props,
            astro_import=f"@/components/custom/{custom_raw.get('name', 'custom')}.astro",
            interactivity=interactivity,
            description=custom_raw.get("description", ""),
            source="llm_generated",
            confidence=confidence,
        )

        sig = self._custom_registry.register(defn)
        self._index.register_custom(defn)

        # Build PropMapping
        mappings = []
        for m in prop_mapping_raw:
            mappings.append(
                {
                    "segment_field": m.get("segment_field", ""),
                    "component_prop": m.get("component_prop", ""),
                    "type": "text",
                    "confidence": float(m.get("confidence", 0.6)),
                    "ambiguous": float(m.get("confidence", 0.6)) < 0.70,
                }
            )
        prop_mapping = PropMapping(
            mappings=mappings,
            has_ambiguous=any(m["ambiguous"] for m in mappings),
        )

        try:
            astro = generate_astro_component(seg, sig, prop_mapping, defn.name)
        except Exception as exc:
            logger.debug("Astro gen failed for novel %s: %s", seg.segment_id, exc)
            from component_mapper.stages.cache_lookup import _minimal_astro

            astro = _minimal_astro(defn.name)

        # Cache
        asyncio.create_task(
            self._cache.set(
                seg.fingerprint_hash,
                MappingCacheRecord(
                    fingerprint_hash=seg.fingerprint_hash,
                    component_name=defn.name,
                    registry_source=sig.registry_source,
                    prop_mapping=prop_mapping,
                    mapping_stage=MappingStage.LLM_NOVEL,
                    confidence=confidence,
                ),
            )
        )

        return MappedComponent(
            segment_id=seg.segment_id,
            page_url=seg.page_url,
            component_type=seg.component_type,
            classification_stage=seg.classification_stage,
            component_name=defn.name,
            registry_source=sig.registry_source,
            mapping_stage=MappingStage.LLM_NOVEL,
            mapping_confidence=confidence,
            prop_mapping=prop_mapping,
            astro_component=astro,
            llm_model_used=model,
            llm_reasoning=reasoning,
        )
