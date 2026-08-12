"""Qdrant ingestion pipeline for dense+sparse chunk indexing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

from qdrant_client import QdrantClient, models
import torch
import logging

from src.core.settings.device import (
    GIGA_EMBEDDING_DIM,
    load_sentence_transformer,
    log_gpu_memory,
    release_embedding_model,
    resolve_torch_device,
)
from src.core.vector.bm25_sparse import Bm25SparseEncoder
from src.core.vector.ingest_fingerprint import validate_fingerprint_or_raise

logger = logging.getLogger(__name__)


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
    fallback_to_cpu: bool = True
    adaptive_batch: bool = True
    min_batch_size: int = 4
    expected_dim: int = GIGA_EMBEDDING_DIM
    model_dir: Path | None = None
    reset_checkpoint: bool = False


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
        model_dir = config.model_dir or Path(config.dense_model)
        validate_fingerprint_or_raise(
            checkpoint_path=config.checkpoint_path,
            model_dir=model_dir,
            dense_model=config.dense_model,
            collection_name=config.collection_name,
            expected_dim=config.expected_dim,
            reset_checkpoint=config.reset_checkpoint,
        )
        preferred = resolve_torch_device(
            preferred=config.device,
            fallback_to_cpu=config.fallback_to_cpu,
        )
        local_only = bool(model_dir and model_dir.exists())
        self.model = load_sentence_transformer(
            model_path=config.dense_model,
            preferred_device=preferred,
            trust_remote_code=config.trust_remote_code,
            fallback_to_cpu=config.fallback_to_cpu,
            expected_dim=config.expected_dim,
            local_files_only=local_only,
        )
        self.sparse_encoder = Bm25SparseEncoder()
        self.checkpoint = CheckpointStore(path=config.checkpoint_path)
        self.batch_size = max(config.min_batch_size, config.batch_size)

    def close(self) -> None:
        release_embedding_model(self.model)
        self.model = None

    def _read_rows(self, chunks_tsv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
        csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
        with chunks_tsv_path.open("r", encoding="utf-8", newline="") as file_handle:
            rows = list(csv.DictReader(file_handle, delimiter="\t"))
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

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        while True:
            try:
                log_gpu_memory(tag=f"encode_batch_size_{len(texts)}")
                return self.model.encode(texts, normalize_embeddings=True).tolist()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                if not self.config.adaptive_batch or self.batch_size <= self.config.min_batch_size:
                    if self.config.fallback_to_cpu and str(getattr(self.model, "device", "")) != "cpu":
                        self.model = load_sentence_transformer(
                            model_path=self.config.dense_model,
                            preferred_device="cpu",
                            trust_remote_code=self.config.trust_remote_code,
                            fallback_to_cpu=False,
                            expected_dim=self.config.expected_dim,
                            local_files_only=True,
                        )
                        continue
                    raise
                self.batch_size = max(self.config.min_batch_size, self.batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(texts) > self.batch_size:
                    raise RuntimeError(
                        f"OOM at batch; retry with smaller batch_size={self.batch_size}"
                    ) from error

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
        total = len(rows)
        processed = 0
        offset = start_offset
        started = time.perf_counter()
        self._print_progress(
            offset=offset,
            total=total,
            batch_size=self.batch_size,
            elapsed_sec=0.0,
            prefix="ingest_start",
        )
        while offset < total:
            batch = rows[offset : offset + self.batch_size]
            texts = [row["text"] for row in batch]
            try:
                dense_vectors = self._encode_batch(texts=texts)
            except RuntimeError as error:
                if "retry with smaller batch_size" in str(error):
                    continue
                raise
            points: list[models.PointStruct] = []
            for row, dense_vector in zip(batch, dense_vectors):
                sparse_vector = self.sparse_encoder.encode_document(text=row["text"])
                points.append(
                    models.PointStruct(
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
                )
            self._upsert_with_retries(points=points)
            processed += len(batch)
            offset += len(batch)
            self.checkpoint.save(offset)
            self._print_progress(
                offset=offset,
                total=total,
                batch_size=self.batch_size,
                elapsed_sec=time.perf_counter() - started,
                prefix="ingest",
            )

        warmed = self._prewarm_cache()
        mem = log_gpu_memory(tag="ingest_complete") or {}
        self._print_progress(
            offset=offset,
            total=total,
            batch_size=self.batch_size,
            elapsed_sec=time.perf_counter() - started,
            prefix="ingest_done",
        )
        return {
            "rows_total": total,
            "rows_processed": processed,
            "checkpoint_offset": self.checkpoint.load(),
            "prewarmed_points": warmed,
            "final_batch_size": self.batch_size,
            **{f"gpu_{key}": value for key, value in mem.items()},
        }

    @staticmethod
    def _print_progress(
        *,
        offset: int,
        total: int,
        batch_size: int,
        elapsed_sec: float,
        prefix: str,
    ) -> None:
        pct = (100.0 * offset / total) if total else 100.0
        rate = (offset / elapsed_sec) if elapsed_sec > 0 else 0.0
        remaining = max(total - offset, 0)
        eta_sec = (remaining / rate) if rate > 0 else 0.0
        line = (
            f"[{prefix}] {offset}/{total} ({pct:5.1f}%) "
            f"batch={batch_size} rate={rate:5.2f} rows/s "
            f"elapsed={elapsed_sec/60:5.1f}m eta={eta_sec/60:5.1f}m"
        )
        print(line, flush=True)
        logger.info(line)
