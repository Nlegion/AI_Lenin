"""IO helpers for quality QA batch (load, checkpoint, artifact paths, txt)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.lib._quality_qa_txt import (  # noqa: F401 - re-export public API
    format_answer_for_display,
    format_txt_block,
    format_txt_header,
)


REQUIRED_FIELDS = ("id", "title", "content", "question")


@dataclass(frozen=True)
class QaItem:
    id: str
    title: str
    content: str
    question: str
    topic: str = ""
    source: str = ""

    def input_hash(self) -> str:
        payload = f"{self.title}\0{self.content}\0{self.question}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ArtifactPaths:
    checkpoint: Path
    results: Path
    txt: Path


def load_qa_items(path: Path) -> list[QaItem]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    items: list[QaItem] = []
    seen: dict[str, int] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        for field in REQUIRED_FIELDS:
            value = str(row.get(field, "")).strip()
            if not value:
                raise ValueError(f"Line {line_no}: missing/empty required field '{field}'")
        item_id = str(row["id"]).strip()
        if item_id in seen:
            raise ValueError(f"Duplicate id '{item_id}' at lines {seen[item_id]} and {line_no}")
        seen[item_id] = line_no
        items.append(
            QaItem(
                id=item_id,
                title=str(row["title"]).strip(),
                content=str(row["content"]).strip(),
                question=str(row["question"]).strip(),
                topic=str(row.get("topic", "") or "").strip(),
                source=str(row.get("source", "") or "").strip(),
            )
        )
    if not items:
        raise ValueError(f"No items loaded from {path}")
    return items


def load_checkpoint_last_wins(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    last: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        item_id = str(row.get("id", "")).strip()
        if item_id:
            last[item_id] = row
    return last


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_artifact_paths(
    *,
    input_path: Path,
    output_dir: Path,
    checkpoint: Path | None,
    stamp: str | None = None,
) -> ArtifactPaths:
    if checkpoint is not None:
        ckpt = checkpoint
        name = ckpt.name
        if name.endswith(".checkpoint.jsonl"):
            stem = name[: -len(".checkpoint.jsonl")]
            results = ckpt.with_name(f"{stem}.jsonl")
            txt = ckpt.with_name(f"{stem}.txt")
        else:
            results = ckpt.with_name(f"{ckpt.name}.results.jsonl")
            txt = ckpt.with_name(f"{ckpt.name}.txt")
        return ArtifactPaths(checkpoint=ckpt, results=results, txt=txt)

    stamp_value = stamp or datetime.now().strftime("%Y%m%d-%H%M")
    stem = f"{input_path.stem}_{stamp_value}"
    base = output_dir
    base.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        checkpoint=base / f"{stem}.checkpoint.jsonl",
        results=base / f"{stem}.jsonl",
        txt=base / f"{stem}.txt",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def should_skip_checkpoint_row(*, row: dict[str, Any] | None, input_hash: str, force: bool) -> bool:
    if force or row is None:
        return False
    status = str(row.get("status", ""))
    if status not in {"done", "blocked"}:
        return False
    return str(row.get("input_hash", "")) == input_hash
