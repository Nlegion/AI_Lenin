import logging
from pathlib import Path
from typing import List

import aiohttp

from src.core.analysis.context_orchestrator import AnalysisContextOrchestrator
from src.core.generation.degrade_policy import CircuitBreaker, template_degrade
from src.core.generation.errors import ErrorKind, classify_exception
from src.core.generation.pipeline import AnalysisGenerationPipeline
from src.core.retrieval.provider_factory import build_provider
from src.core.safety.news_guard import NewsGuard
from src.core.settings.analysis_defaults import ANALYSIS_CACHE_LIMIT
from src.core.settings.config import Settings
from src.core.generation.postprocess_clean.passthrough import passthrough_pipeline_text
from src.core.settings.generation_config import (
    PersonaModel,
    default_generation_config_path,
    load_generation_config,
)
from src.core.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class LeninAnalyzer:
    def __init__(
        self, vector_db_path: str = None, persona_model: PersonaModel | None = None
    ):
        logger.info("Инициализация EnhancedLeninAnalyzer")
        _ = vector_db_path  # compatibility with legacy initializer signature
        self.config = Settings()
        self.base_dir = Path(self.config.BASE_DIR)
        self.session = None
        self.analysis_cache = {}
        self.text_cleaner = TextCleaner()
        self.persona_model = persona_model
        self.generation_config = load_generation_config(
            path=default_generation_config_path(self.base_dir)
        )
        if persona_model is not None:
            self.generation_config = self.generation_config.with_persona_model(
                persona_model
            )
        self.retrieval_provider = self._init_retrieval_provider()
        retrieval_config_path = self.base_dir / "config" / "retrieval_pipeline.yaml"
        self.context_orchestrator = AnalysisContextOrchestrator(
            retrieval_provider=self.retrieval_provider,
            config_path=retrieval_config_path,
            taxonomy_path=self.base_dir / "config" / "ontology_taxonomy.yaml",
        )
        self.news_guard = self._init_news_guard()
        self._pipeline: AnalysisGenerationPipeline | None = None
        self.last_pipeline_metadata: dict = {}
        self.circuit_breaker = CircuitBreaker()

    def _init_news_guard(self) -> NewsGuard | None:
        config_path = self.base_dir / "config" / "news_guard.yaml"
        if not config_path.exists():
            return None
        try:
            return NewsGuard.from_file(path=config_path)
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to initialize NewsGuard in analyzer: %s", error)
            return None

    def _init_retrieval_provider(self):
        config_path = self.base_dir / "config" / "retrieval_pipeline.yaml"
        try:
            provider = build_provider(
                config_path=config_path,
                base_dir=self.base_dir,
            )
            if provider is None:
                logger.info("Retrieval provider disabled or unavailable.")
                return None
            logger.info("Retrieval provider initialized from pipeline config.")
            return provider
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to initialize retrieval provider: %s", error)
            return None

    async def initialize_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300, sock_connect=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        if self._pipeline is not None:
            await self._pipeline.close()
            self._pipeline = None
        if self.session:
            await self.session.close()
            self.session = None

    def _get_pipeline(self) -> AnalysisGenerationPipeline:
        if self._pipeline is None:
            dialectical_enabled = bool(
                self.context_orchestrator.dialectical_config.enabled
            )
            self._pipeline = AnalysisGenerationPipeline(
                base_dir=self.base_dir,
                context_builder=self.context_orchestrator.build_context,
                evidence_builder=self.context_orchestrator.build_evidence_brief,
                dialectical_enabled=dialectical_enabled,
                news_guard=self.news_guard,
                text_cleaner=self.text_cleaner,
                generation_config=self.generation_config,
                persona_model=self.persona_model,
                session=self.session,
                apply_fallback_recommendation=True,
            )
        return self._pipeline

    def extract_key_concepts(self, text: str) -> List[str]:
        """Извлечение ключевых концепций с акцентом на международную политэкономию"""
        concepts = []
        political_economy_terms = [
            "капитал",
            "пролетариат",
            "буржуазия",
            "эксплуатация",
            "революция",
            "диалектика",
            "материализм",
            "идеализм",
            "классовая борьба",
            "прибавочная стоимость",
            "средства производства",
            "империализм",
            "монополия",
            "государство",
            "диктатура пролетариата",
            "международная торговля",
            "валютный кризис",
            "рынок",
            "капитализм",
            "социализм",
            "коммунизм",
            "колониализм",
            "неоколониализм",
            "глобализация",
            "национальный вопрос",
            "санкции",
            "экономические санкции",
            "международные отношения",
            "дипломатия",
            "гегемония",
            "мировой рынок",
            "транснациональные корпорации",
            "международное разделение труда",
            "внешняя политика",
            "экономическая зависимость",
            "сырьевая экономика",
            "финансовый капитал",
            "долговая зависимость",
            "неравномерное развитие",
        ]
        text_lower = text.lower()
        for term in political_economy_terms:
            if term in text_lower:
                concepts.append(term)
        if any(
            word in text_lower
            for word in ["экономик", "финанс", "деньг", "рынок", "банк", "валюта"]
        ):
            concepts.extend(["экономика", "капитал", "прибыль", "политэкономия"])
        if any(
            word in text_lower
            for word in ["политик", "власт", "правительств", "государств", "партия"]
        ):
            concepts.extend(
                ["политика", "государство", "власть", "диктатура пролетариата"]
            )
        if any(
            word in text_lower
            for word in ["международн", "дипломати", "санкц", "договор", "ООН", "НАТО"]
        ):
            concepts.extend(
                [
                    "империализм",
                    "международные отношения",
                    "колониализм",
                    "международная политэкономия",
                ]
            )
        if any(word in text_lower for word in ["войн", "военн", "конфликт", "оруж"]):
            concepts.extend(
                [
                    "империализм",
                    "война",
                    "мирное сосуществование",
                    "военно-промышленный комплекс",
                ]
            )
        return list(set(concepts))[:5]

    async def generate_analysis(
        self,
        news_title: str,
        news_content: str,
        feedback: List[str] = None,
        risk_tier: str = "green",
        context_hints: list[str] | None = None,
        needs_yellow_warning: bool = False,
    ) -> str:
        try:
            await self.initialize_session()
            from src.core.settings.runtime_knobs import (
                load_reasoning_config_with_generation_sot,
            )

            if not self.circuit_breaker.allow_request():
                degraded = template_degrade(reason="circuit_open")
                self.last_pipeline_metadata = dict(degraded.metadata)
                return degraded.text

            reasoning_cfg = load_reasoning_config_with_generation_sot(
                base_dir=self.base_dir
            )
            cache_key = (
                f"{news_title}_{hash(news_content[:200])}_"
                f"{reasoning_cfg.mode.value}_{reasoning_cfg.schema_version}"
            )
            if cache_key in self.analysis_cache and not feedback:
                self.last_pipeline_metadata = {"cache_hit": True}
                return self.analysis_cache[cache_key]

            key_concepts = self.extract_key_concepts(news_content)
            enhanced_query = (
                f"{news_title} {news_content[:200]} {' '.join(key_concepts)}"
            )
            pipeline = self._get_pipeline()
            result = await pipeline.generate(
                news_title=news_title,
                news_content=news_content,
                enhanced_query=enhanced_query,
                key_concepts=key_concepts,
                feedback=feedback,
                warn_only_guard=False,
                risk_tier=risk_tier,
                context_hints=context_hints,
                needs_yellow_warning=needs_yellow_warning,
            )
            self.circuit_breaker.record_success()
            self.last_pipeline_metadata = dict(result.metadata or {})
            self.last_pipeline_metadata["latency_ms"] = int(result.latency_ms)
            # Terminal post-guard already ran in the pipeline; do not re-mutate.
            cleaned_content = self.clean_analysis(result.analysis)
            suffix = result.metadata.get("cache_suffix")
            if suffix:
                cache_key = f"{news_title}_{hash(news_content[:200])}_{suffix}"
            if not feedback and result.metadata.get("orchestration_mode") != "error":
                if result.metadata.get("dialectical_outcome") != "suppress":
                    self.analysis_cache[cache_key] = cleaned_content
                    if len(self.analysis_cache) > ANALYSIS_CACHE_LIMIT:
                        self.analysis_cache.clear()
            return cleaned_content
        except Exception as error:  # noqa: BLE001
            kind = classify_exception(error)
            logger.exception("Ошибка генерации kind=%s: %s", kind.value, error)
            if kind == ErrorKind.TRANSIENT:
                self.circuit_breaker.record_timeout()
            degraded = template_degrade(reason=f"generation_{kind.value}")
            self.last_pipeline_metadata = {
                **degraded.metadata,
                "error_kind": kind.value,
                "error": str(error)[:300],
            }
            # Do not raise: production cycle continues; result is non-publishable.
            return degraded.text

    def clean_analysis(self, text: str) -> str:
        """Identity for pipeline output. Placeholders stay non-publishable."""
        return passthrough_pipeline_text(text)
