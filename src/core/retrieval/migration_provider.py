"""Migration provider for controlled Chroma -> Qdrant cutover."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path

from src.core.retrieval.base_provider import RetrievalProvider, RetrievalResult

logger = logging.getLogger(__name__)


MigrationMode = str


@dataclass(frozen=True)
class MigrationConfig:
    mode: MigrationMode
    parity_min_shared_ratio: float = 0.25
    audit_log_path: Path | None = None


@dataclass(frozen=True)
class ParitySnapshot:
    shared_ratio: float
    primary_non_empty: bool
    shadow_non_empty: bool


class MigrationRetrievalProvider(RetrievalProvider):
    """Runs one provider as primary and optionally compares with shadow provider."""

    def __init__(
        self,
        primary: RetrievalProvider,
        shadow: RetrievalProvider | None,
        config: MigrationConfig,
        primary_name: str,
        shadow_name: str | None = None,
    ):
        self.primary = primary
        self.shadow = shadow
        self.config = config
        self.primary_name = primary_name
        self.shadow_name = shadow_name or "none"

    def retrieve_context(self, query_text: str, author_filter: str | None = None) -> RetrievalResult:
        mode = self.config.mode
        if mode in {"qdrant_only", "chroma_only"}:
            return self.primary.retrieve_context(query_text=query_text, author_filter=author_filter)

        if mode not in {"ab_shadow"}:
            raise ValueError(f"Unsupported migration mode: {mode}")

        primary_result = self.primary.retrieve_context(query_text=query_text, author_filter=author_filter)
        if self.shadow is None:
            return primary_result

        shadow_result = self.shadow.retrieve_context(query_text=query_text, author_filter=author_filter)
        parity = self._compute_parity(primary_result=primary_result, shadow_result=shadow_result)

        if parity.shared_ratio < self.config.parity_min_shared_ratio:
            logger.warning(
                "Retrieval parity below threshold. shared_ratio=%.3f threshold=%.3f",
                parity.shared_ratio,
                self.config.parity_min_shared_ratio,
            )
        self._write_audit(
            query_text=query_text,
            primary_result=primary_result,
            shadow_result=shadow_result,
            parity=parity,
        )
        merged_metadata = dict(primary_result.metadata)
        merged_metadata.update(
            {
                "mode": mode,
                "primary": self.primary_name,
                "shadow": self.shadow_name,
                "parity_shared_ratio": f"{parity.shared_ratio:.4f}",
                "parity_primary_non_empty": str(parity.primary_non_empty).lower(),
                "parity_shadow_non_empty": str(parity.shadow_non_empty).lower(),
            }
        )
        return RetrievalResult(
            context=primary_result.context,
            candidates_count=primary_result.candidates_count,
            metadata=merged_metadata,
        )

    @staticmethod
    def _tokenize(context: str) -> set[str]:
        return {token for token in context.lower().split() if token}

    def _compute_parity(self, primary_result: RetrievalResult, shadow_result: RetrievalResult) -> ParitySnapshot:
        primary_tokens = self._tokenize(primary_result.context)
        shadow_tokens = self._tokenize(shadow_result.context)
        union = primary_tokens | shadow_tokens
        if not union:
            shared_ratio = 1.0
        else:
            shared_ratio = len(primary_tokens & shadow_tokens) / len(union)
        return ParitySnapshot(
            shared_ratio=shared_ratio,
            primary_non_empty=bool(primary_result.context.strip()),
            shadow_non_empty=bool(shadow_result.context.strip()),
        )

    def _write_audit(
        self,
        query_text: str,
        primary_result: RetrievalResult,
        shadow_result: RetrievalResult,
        parity: ParitySnapshot,
    ) -> None:
        if self.config.audit_log_path is None:
            return
        payload = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": self.config.mode,
            "primary": self.primary_name,
            "shadow": self.shadow_name,
            "query_hash": f"sha256:{hashlib.sha256(query_text.encode('utf-8')).hexdigest()}",
            "primary_candidates": primary_result.candidates_count,
            "shadow_candidates": shadow_result.candidates_count,
            "primary_non_empty": parity.primary_non_empty,
            "shadow_non_empty": parity.shadow_non_empty,
            "shared_ratio": parity.shared_ratio,
            "parity_min_shared_ratio": self.config.parity_min_shared_ratio,
        }
        path = self.config.audit_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
