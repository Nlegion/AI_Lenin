"""Qdrant retrieval provider with query rewriting and arbitration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

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


class QdrantRetrievalProvider:
    def __init__(self, config: RetrievalProviderConfig):
        self.config = config
        self.client = QdrantClient(path=str(config.qdrant_path))
        self.model = SentenceTransformer(
            model_name_or_path=config.dense_model,
            trust_remote_code=config.trust_remote_code,
            device=config.device,
        )
        self.sparse_encoder = Bm25SparseEncoder.load(path=config.sparse_state_path)
        self.ontology_rows = self._read_ontology(path=config.ontology_tags_path)

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

    def retrieve(self, query_text: str) -> list[RetrievalCandidate]:
        queries = self._prepare_queries(query_text=query_text)
        candidates: list[RetrievalCandidate] = []
        for query in queries:
            candidates.extend(self._dense_search(query_text=query, retriever_name="dense", limit=self.config.top_k))
            candidates.extend(self._sparse_search(query_text=query, limit=self.config.top_k))
            candidates.extend(self._ontology_search(query_text=query, limit=self.config.top_k))

        if not candidates:
            return []

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
        ordered_chunk_ids = [chunk_id for chunk_id, _ in sorted(boosted.items(), key=lambda item: item[1], reverse=True)]
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
        candidates = self.retrieve(query_text=query_text)
        return RetrievalResult(
            context=self.render_context(candidates=candidates),
            candidates_count=len(candidates),
            metadata={"provider": "qdrant"},
        )
