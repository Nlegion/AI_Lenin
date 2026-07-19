"""Model fingerprint helpers for safe ingest checkpoint resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_model_fingerprint(model_dir: Path) -> str:
    parts: list[str] = []
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "modules.json",
        "config_sentence_transformers.json",
    ):
        path = model_dir / name
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            parts.append(f"{name}:{digest}:{path.stat().st_size}")
    for path in sorted(model_dir.glob("model-*.safetensors")):
        stat = path.stat()
        parts.append(f"{path.name}:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def meta_path_for_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")


def load_fingerprint_meta(checkpoint_path: Path) -> dict[str, Any] | None:
    path = meta_path_for_checkpoint(checkpoint_path=checkpoint_path)
    if not path.exists():
        # Support plan naming: ingestion_giga_v1.meta.json beside .offset
        alt = checkpoint_path.with_name(checkpoint_path.name.replace(".offset", ".meta.json"))
        if alt.exists():
            path = alt
        else:
            return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_fingerprint_meta(checkpoint_path: Path, payload: dict[str, Any]) -> Path:
    path = checkpoint_path.with_name(checkpoint_path.name.replace(".offset", ".meta.json"))
    if not str(checkpoint_path).endswith(".offset"):
        path = meta_path_for_checkpoint(checkpoint_path=checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_fingerprint_or_raise(
    *,
    checkpoint_path: Path,
    model_dir: Path,
    dense_model: str,
    collection_name: str,
    expected_dim: int,
    reset_checkpoint: bool,
) -> str:
    current = compute_model_fingerprint(model_dir=model_dir)
    meta = load_fingerprint_meta(checkpoint_path=checkpoint_path)
    if meta is None:
        write_fingerprint_meta(
            checkpoint_path=checkpoint_path,
            payload={
                "dense_model": dense_model,
                "model_fingerprint": current,
                "expected_dim": expected_dim,
                "collection_name": collection_name,
            },
        )
        return current
    stored = str(meta.get("model_fingerprint", ""))
    if stored and stored != current:
        if not reset_checkpoint:
            raise RuntimeError(
                "Ingest checkpoint fingerprint mismatch. "
                "Re-run with --reset-checkpoint if the model/chunks intentionally changed. "
                f"stored={stored[:16]}... current={current[:16]}..."
            )
    if reset_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("0", encoding="utf-8")
    write_fingerprint_meta(
        checkpoint_path=checkpoint_path,
        payload={
            "dense_model": dense_model,
            "model_fingerprint": current,
            "expected_dim": expected_dim,
            "collection_name": collection_name,
        },
    )
    return current
