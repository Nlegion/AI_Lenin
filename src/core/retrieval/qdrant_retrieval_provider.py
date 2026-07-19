"""Qdrant retrieval provider with query rewriting and arbitration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

from qdrant_client import QdrantClient, models

from src.core.settings.device import (
    GIGA_EMBEDDING_DIM,
    ensure_exclusive_gpu_for_embeddings,
    load_sentence_transformer,
    release_embedding_model,
)
from src.core.retrieval.arbiter import (
    RetrievalCandidate,
    apply_stance_boost,
    enforce_core_self_presence,
    weighted_rrf,
)
from src.core.retrieval.query_transform import (
    build_hyde_query,
    decompose_query,
    rewrite_query_to_philosophical_register,
)
from src.core.retrieval.base_provider import RetrievalResult
from src.core.vector.bm25_sparse import Bm25SparseEncoder


@dataclass(frozen=True)
class RetrievalProviderConfig:
    collection_name: str
    qdrant_path: Path
    dense_model: str
    sparse_state_path: Path
    ontology_tags_path: Path
    trust_remote_code: bool
    device: str
    top_k: int
    rrf_k: int
    retriever_weights: dict[str, float]
    source_boosts: dict[str, float]
    max_context_chunks: int
    hyde_enabled: bool
    query_rewrite_enabled: bool
    query_decomposition_enabled: bool
    judge_enabled: bool = True
    judge_alpha: float = 0.2
    fallback_to_cpu: bool = True
    expected_dim: int | None = GIGA_EMBEDDING_DIM
    server_url: str = "http://127.0.0.1:8080"
    interactive: bool = True


class QdrantRetrievalProvider:
    def __init__(self, config: RetrievalProviderConfig):
        self.config = config
        self.client = QdrantClient(path=str(config.qdrant_path))
        device = ensure_exclusive_gpu_for_embeddings(
            preferred=config.device,
            fallback_to_cpu=config.fallback_to_cpu,
            server_url=config.server_url,
            interactive=config.interactive,
        )
        local_only = Path(config.dense_model).exists()
        self.model = load_sentence_transformer(
            model_path=config.dense_model,
            preferred_device=device,
            trust_remote_code=config.trust_remote_code,
            fallback_to_cpu=config.fallback_to_cpu,
            expected_dim=config.expected_dim,
            local_files_only=local_only,
        )
        self.resolved_device = device
        self.sparse_encoder = Bm25SparseEncoder.load(path=config.sparse_state_path)
        self.ontology_rows = self._read_ontology(path=config.ontology_tags_path)

    def close(self) -> None:
        release_embedding_model(self.model)
        self.model = None

    @staticmethod
    def _read_ontology(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            return list(csv.DictReader(file_handle, delimiter="\t"))

    def _dense_search(self, query_text: str, retriever_name: str, limit: int) -> list[RetrievalCandidate]:
        vector = self.model.encode([query_text], normalize_embeddings=True)[0].tolist()
        points = self.client.query_points(
            collection_name=self.config.collection_name,
            query=vector,
            using="dense",
            limit=limit,
            with_payload=True,
        ).points
        return [
            RetrievalCandidate(
                source_id=str(point.payload.get("source_id")),
                chunk_id=str(point.payload.get("chunk_id")),
                text=str(point.payload.get("text", "")),
                stance_type=str(point.payload.get("stance_type", "contextual")),
                source_path=str(point.payload.get("source_path", "")),
                retriever=retriever_name,
                rank=index,
                score=float(point.score),
            )
            for index, point in enumerate(points, start=1)
        ]

    def _sparse_search(self, query_text: str, limit: int) -> list[RetrievalCandidate]:
        sparse = self.sparse_encoder.encode_query(text=query_text)
        points = self.client.query_points(
            collection_name=self.config.collection_name,
            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using="sparse",
            limit=limit,
            with_payload=True,
        ).points
        return [
            RetrievalCandidate(
                source_id=str(point.payload.get("source_id")),
                chunk_id=str(point.payload.get("chunk_id")),
                text=str(point.payload.get("text", "")),
                stance_type=str(point.payload.get("stance_type", "contextual")),
                source_path=str(point.payload.get("source_path", "")),
                retriever="sparse",
                rank=index,
                score=float(point.score),
            )
            for index, point in enumerate(points, start=1)
        ]

    def _ontology_search(self, query_text: str, limit: int) -> list[RetrievalCandidate]:
        terms = set(re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", query_text.lower()))
        results: list[RetrievalCandidate] = []
        rank = 1
        for row in self.ontology_rows:
            concepts = [item for item in row.get("concepts", "").split("|") if item]
            overlap = sum(1 for concept in concepts if concept.lower() in terms)
            if overlap <= 0:
                continue
            results.append(
                RetrievalCandidate(
                    source_id=str(row.get("source_id", "")),
                    chunk_id=str(row.get("source_id", "")),
                    text="",
                    stance_type="contextual",
                    source_path=str(row.get("source_path", "")),
                    retriever="onto",
                    rank=rank,
                    score=float(overlap),
                )
            )
            rank += 1
            if len(results) >= limit:
                break
        return results

    def _prepare_queries(self, query_text: str) -> list[str]:
        base = rewrite_query_to_philosophical_register(query_text) if self.config.query_rewrite_enabled else query_text
        queries = [base]
        if self.config.query_decomposition_enabled:
            factual, evaluative = decompose_query(base)
            queries.extend([factual, evaluative])
        if self.config.hyde_enabled:
            queries.append(build_hyde_query(base))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in queries:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped

    @staticmethod
    def _judge_scores(query_text: str, candidates: list[RetrievalCandidate]) -> dict[str, float]:
        terms = {token for token in re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", query_text.lower()) if token}
        scores: dict[str, float] = {}
        for candidate in candidates:
            payload_terms = {token for token in re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", candidate.text.lower()) if token}
            if not terms:
                scores[candidate.chunk_id] = 0.0
                continue
            overlap = len(terms & payload_terms)
            scores[candidate.chunk_id] = overlap / max(len(terms), 1)
        return scores

    def retrieve_with_trace(self, query_text: str, apply_judge: bool | None = None) -> tuple[list[RetrievalCandidate], dict]:
        effective_apply_judge = self.config.judge_enabled if apply_judge is None else apply_judge
        queries = self._prepare_queries(query_text=query_text)
        candidates: list[RetrievalCandidate] = []
        dense_trace: list[dict[str, str | float | int]] = []
        sparse_trace: list[dict[str, str | float | int]] = []
        onto_trace: list[dict[str, str | float | int]] = []
        for query in queries:
            dense = self._dense_search(query_text=query, retriever_name="dense", limit=self.config.top_k)
            sparse = self._sparse_search(query_text=query, limit=self.config.top_k)
            onto = self._ontology_search(query_text=query, limit=self.config.top_k)
            candidates.extend(dense)
            candidates.extend(sparse)
            candidates.extend(onto)
            dense_trace.extend(self._serialize_candidates(candidates=dense, query=query))
            sparse_trace.extend(self._serialize_candidates(candidates=sparse, query=query))
            onto_trace.extend(self._serialize_candidates(candidates=onto, query=query))

        if not candidates:
            return [], {
                "query_variants": queries,
                "dense": dense_trace,
                "sparse": sparse_trace,
                "onto": onto_trace,
                "merged_scores": {},
                "boosted_scores": {},
                "judge_scores": {},
                "judge_enabled": effective_apply_judge,
            }

        chunk_to_stance = {candidate.chunk_id: candidate.stance_type for candidate in candidates}
        merged_scores = weighted_rrf(
            candidates=candidates,
            retriever_weights=self.config.retriever_weights,
            rrf_k=self.config.rrf_k,
        )
        boosted = apply_stance_boost(
            scores=merged_scores,
            chunk_to_stance=chunk_to_stance,
            source_boosts=self.config.source_boosts,
        )
        judge_scores: dict[str, float] = {}
        final_scores = boosted
        if effective_apply_judge:
            judge_scores = self._judge_scores(query_text=query_text, candidates=candidates)
            final_scores = {
                chunk_id: (1 - self.config.judge_alpha) * boosted.get(chunk_id, 0.0)
                + self.config.judge_alpha * judge_scores.get(chunk_id, 0.0)
                for chunk_id in boosted
            }
        ordered_chunk_ids = [chunk_id for chunk_id, _ in sorted(final_scores.items(), key=lambda item: item[1], reverse=True)]
        ordered_chunk_ids = enforce_core_self_presence(
            ordered_chunk_ids=ordered_chunk_ids,
            chunk_to_stance=chunk_to_stance,
        )

        best_by_chunk: dict[str, RetrievalCandidate] = {}
        for candidate in candidates:
            current = best_by_chunk.get(candidate.chunk_id)
            if current is None or candidate.score > current.score:
                best_by_chunk[candidate.chunk_id] = candidate

        final: list[RetrievalCandidate] = []
        for chunk_id in ordered_chunk_ids:
            if chunk_id in best_by_chunk:
                final.append(best_by_chunk[chunk_id])
            if len(final) >= self.config.max_context_chunks:
                break
        trace = {
            "query_variants": queries,
            "dense": dense_trace,
            "sparse": sparse_trace,
            "onto": onto_trace,
            "merged_scores": {key: round(value, 6) for key, value in merged_scores.items()},
            "boosted_scores": {key: round(value, 6) for key, value in boosted.items()},
            "judge_scores": {key: round(value, 6) for key, value in judge_scores.items()},
            "final_scores": {key: round(value, 6) for key, value in final_scores.items()},
            "judge_enabled": effective_apply_judge,
        }
        return final, trace

    @staticmethod
    def _serialize_candidates(candidates: list[RetrievalCandidate], query: str) -> list[dict[str, str | float | int]]:
        return [
            {
                "query": query,
                "chunk_id": candidate.chunk_id,
                "source_id": candidate.source_id,
                "stance_type": candidate.stance_type,
                "retriever": candidate.retriever,
                "rank": candidate.rank,
                "score": round(candidate.score, 6),
                "source_path": candidate.source_path,
            }
            for candidate in candidates
        ]

    def retrieve(self, query_text: str) -> list[RetrievalCandidate]:
        final, _ = self.retrieve_with_trace(query_text=query_text)
        return final

    def render_context(self, candidates: list[RetrievalCandidate]) -> str:
        parts: list[str] = []
        for candidate in candidates:
            source = candidate.source_path or candidate.source_id
            if candidate.text.strip():
                parts.append(
                    f"[source: {source}; stance: {candidate.stance_type}; retriever: {candidate.retriever}] {candidate.text}"
                )
            else:
                parts.append(
                    f"[source: {source}; stance: {candidate.stance_type}; retriever: {candidate.retriever}]"
                )
        return "\n\n".join(parts)

    def retrieve_context(self, query_text: str, author_filter: str | None = None) -> RetrievalResult:
        _ = author_filter  # kept for contract parity with legacy provider
        candidates, trace = self.retrieve_with_trace(query_text=query_text)
        return RetrievalResult(
            context=self.render_context(candidates=candidates),
            candidates_count=len(candidates),
            metadata={
                "provider": "qdrant",
                "query_variants": " || ".join(trace.get("query_variants", [])),
                "judge_enabled": str(trace.get("judge_enabled", False)).lower(),
            },
        )
