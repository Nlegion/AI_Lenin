"""Qdrant ingestion pipeline for dense+sparse chunk indexing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
import sys

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.core.vector.bm25_sparse import Bm25SparseEncoder


@dataclass
class IngestionConfig:
    collection_name: str
    dense_model: str
    trust_remote_code: bool
    device: str
    batch_size: int
    retries: int
    checkpoint_path: Path
    qdrant_path: Path
    sparse_state_path: Path
    prewarm_core_limit: int


class CheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> int:
        if not self.path.exists():
            return 0
        try:
            return int(self.path.read_text(encoding="utf-8").strip())
        except ValueError:
            return 0

    def save(self, offset: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(offset), encoding="utf-8")


class QdrantIngestionPipeline:
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.client = QdrantClient(path=str(config.qdrant_path))
        self.model = SentenceTransformer(
            model_name_or_path=config.dense_model,
            trust_remote_code=config.trust_remote_code,
            device=config.device,
        )
        self.sparse_encoder = Bm25SparseEncoder()
        self.checkpoint = CheckpointStore(path=config.checkpoint_path)

    def _read_rows(self, chunks_tsv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
        csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
        with chunks_tsv_path.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle, delimiter="\t")
            rows = list(reader)
        return rows[:limit] if limit else rows

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if self.config.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False)),
            },
        )

    def _upsert_with_retries(self, points: list[models.PointStruct]) -> None:
        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.config.retries:
            try:
                self.client.upsert(collection_name=self.config.collection_name, points=points)
                return
            except Exception as error:  # noqa: BLE001
                last_error = error
                attempt += 1
                time.sleep(min(2.0, 0.2 * attempt))
        raise RuntimeError(f"Upsert failed after retries: {last_error}") from last_error

    def _build_payload(self, row: dict[str, str]) -> dict[str, str | int | bool]:
        return {
            "chunk_id": row["chunk_id"],
            "source_id": row["source_id"],
            "source_path": row["source_path"],
            "author": row["author"],
            "work": row["work"],
            "stance_type": row["stance_type"],
            "chapter": row["chapter"],
            "section": row["section"],
            "paragraph_index": int(row["paragraph_index"]),
            "thesis_index": int(row["thesis_index"]),
            "chunk_index": int(row["chunk_index"]),
            "token_count": int(row["token_count"]),
            "char_start": int(row["char_start"]),
            "char_end": int(row["char_end"]),
            "boundary_ok": str(row["boundary_ok"]).lower() == "true",
            "text": row["text"],
        }

    def _to_point_id(self, chunk_id: str) -> int:
        digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:15]
        return int(digest, 16)

    def _prewarm_cache(self) -> int:
        points, _ = self.client.scroll(
            collection_name=self.config.collection_name,
            limit=self.config.prewarm_core_limit,
            with_payload=True,
            with_vectors=False,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="stance_type",
                        match=models.MatchValue(value="core_self"),
                    )
                ]
            ),
        )
        if not points:
            points, _ = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=self.config.prewarm_core_limit,
                with_payload=True,
                with_vectors=False,
            )
        if points:
            self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[item.id for item in points],
                with_payload=True,
                with_vectors=False,
            )
        return len(points)

    def run(self, chunks_tsv_path: Path, limit: int | None = None) -> dict[str, float | int]:
        rows = self._read_rows(chunks_tsv_path=chunks_tsv_path, limit=limit)
        documents = [row["text"] for row in rows]
        self.sparse_encoder.fit(documents=documents)
        self.sparse_encoder.save(path=self.config.sparse_state_path)
        vector_size = len(self.model.encode(["warmup"], normalize_embeddings=True)[0])
        self._ensure_collection(vector_size=vector_size)

        start_offset = min(self.checkpoint.load(), len(rows))
        processed = 0
        for offset in range(start_offset, len(rows), self.config.batch_size):
            batch = rows[offset : offset + self.config.batch_size]
            texts = [row["text"] for row in batch]
            dense_vectors = self.model.encode(texts, normalize_embeddings=True).tolist()
            points: list[models.PointStruct] = []
            for row, dense_vector in zip(batch, dense_vectors):
                sparse_vector = self.sparse_encoder.encode_document(text=row["text"])
                point = models.PointStruct(
                    id=self._to_point_id(row["chunk_id"]),
                    vector={
                        "dense": dense_vector,
                        "sparse": models.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                    },
                    payload=self._build_payload(row=row),
                )
                points.append(point)
            self._upsert_with_retries(points=points)
            processed += len(batch)
            self.checkpoint.save(offset + len(batch))

        warmed = self._prewarm_cache()
        return {
            "rows_total": len(rows),
            "rows_processed": processed,
            "checkpoint_offset": self.checkpoint.load(),
            "prewarmed_points": warmed,
        }
