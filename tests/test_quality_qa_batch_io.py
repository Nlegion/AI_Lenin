"""Unit tests for quality QA batch IO helpers (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._quality_qa_io import (
    QaItem,
    format_txt_block,
    format_txt_header,
    load_checkpoint_last_wins,
    load_qa_items,
    resolve_artifact_paths,
    should_skip_checkpoint_row,
)


def test_load_qa_items_requires_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "a", "title": "t", "content": "c"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_qa_items(path=path)


def test_load_qa_items_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    row = {"id": "a", "title": "t", "content": "c", "question": "q?"}
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_qa_items(path=path)


def test_checkpoint_last_wins_and_skip(tmp_path: Path) -> None:
    path = tmp_path / "ck.jsonl"
    rows = [
        {"id": "a", "input_hash": "h1", "status": "done", "answer": "old"},
        {"id": "a", "input_hash": "h2", "status": "done", "answer": "new"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    last = load_checkpoint_last_wins(path=path)
    assert last["a"]["answer"] == "new"
    assert should_skip_checkpoint_row(row=last["a"], input_hash="h2", force=False) is True
    assert should_skip_checkpoint_row(row=last["a"], input_hash="h1", force=False) is False


def test_artifact_paths_and_txt_format(tmp_path: Path) -> None:
    ckpt = tmp_path / "batch_20260101.checkpoint.jsonl"
    paths = resolve_artifact_paths(
        input_path=tmp_path / "quality_qa_batch.jsonl",
        output_dir=tmp_path,
        checkpoint=ckpt,
    )
    assert paths.results.name == "batch_20260101.jsonl"
    assert paths.txt.name == "batch_20260101.txt"
    item = QaItem(id="eco_01", title="T", content="C", question="Q?", topic="economy")
    header = format_txt_header()
    assert "только для удобства" in header
    block = format_txt_block(index=1, item=item, answer="A", txt_max_chars=0)
    assert "Вопрос: Q?" in block
    assert "Ответ:\nA" in block


def test_dataset_file_has_fifty_unique_ids() -> None:
    path = Path("data/eval/quality_qa_batch.jsonl")
    items = load_qa_items(path=path)
    assert len(items) == 50
    assert len({item.id for item in items}) == 50
