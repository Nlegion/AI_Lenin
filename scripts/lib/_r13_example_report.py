"""Format R1/R2/R3 example-trace reports (markdown + JSONL)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.lib._quality_qa_io import QaItem

SLOT_SPECS: tuple[tuple[str, str], ...] = (
    ("r1_items", "R1 — Ленин"),
    ("r2_items", "R2 — Опоры"),
    ("r3_items", "R3 — Критика"),
)


def format_slot_markdown(
    title: str, items: list[dict[str, Any]] | None
) -> str:
    rows = list(items or [])
    lines = [f"### {title} — {len(rows)}"]
    if not rows:
        lines.append("")
        lines.append("(пусто)")
        return "\n".join(lines)
    for index, item in enumerate(rows, start=1):
        source = str(item.get("source_path") or item.get("chunk_id") or "?")
        score = item.get("score")
        score_bit = f" score={float(score):.3f}" if score is not None else ""
        text = str(item.get("text") or "").strip() or "(без текста)"
        lines.append("")
        lines.append(f"{index}. `{source}`{score_bit}")
        lines.append(f"   {text}")
    return "\n".join(lines)


def format_report(rows: list[dict[str, Any]]) -> str:
    parts = ["# R1/R2/R3 example trace", ""]
    for index, row in enumerate(rows, start=1):
        item_id = str(row.get("id") or index)
        title = str(row.get("title") or "").strip() or "(без заголовка)"
        content = str(row.get("content") or "").strip() or "(без текста)"
        source = str(row.get("source") or "").strip() or "unknown"
        answer = str(row.get("answer") or "").strip() or "(нет ответа)"
        parts.append(f"## Пример {index} (`{item_id}`)")
        parts.append("")
        parts.append("### Новость")
        parts.append(f"- source: {source}")
        parts.append(f"- title: {title}")
        parts.append("")
        parts.append(content)
        parts.append("")
        for key, slot_title in SLOT_SPECS:
            raw = row.get(key)
            items = raw if isinstance(raw, list) else []
            parts.append(format_slot_markdown(title=slot_title, items=items))
            parts.append("")
        parts.append("### Ответ модели")
        parts.append("")
        parts.append(answer)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def load_fixture_qa_items(*, path: Path, limit: int = 0) -> list[QaItem]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("fixtures", {})
    items: list[QaItem] = []
    for name, row in section.items():
        title = str((row or {}).get("title") or "").strip()
        content = str((row or {}).get("content") or title).strip()
        source = str((row or {}).get("source") or "fixture").strip()
        if not title or not content:
            continue
        items.append(
            QaItem(
                id=f"fixture:{name}",
                title=title,
                content=content,
                question=f"Прокомментируйте с позиций Ленина: {title}",
                topic=str(name),
                source=source,
            )
        )
        if limit > 0 and len(items) >= limit:
            break
    return items


def load_jsonl_qa_items(*, path: Path, limit: int = 0) -> list[QaItem]:
    items: list[QaItem] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        item_id = str(row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        content = str(row.get("content") or title).strip()
        question = str(row.get("question") or "").strip() or (
            f"Прокомментируйте с позиций Ленина: {title}"
        )
        if not item_id or not title or not content:
            continue
        items.append(
            QaItem(
                id=item_id,
                title=title,
                content=content,
                question=question,
                topic=str(row.get("topic") or "").strip(),
                source=str(row.get("source") or "").strip(),
            )
        )
        if limit > 0 and len(items) >= limit:
            break
    return items


def write_report_files(
    *,
    output_dir: Path,
    stem: str,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    md_path = output_dir / f"{stem}_{stamp}.md"
    jsonl_path = output_dir / f"{stem}_{stamp}.jsonl"
    md_path.write_text(format_report(rows), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return md_path, jsonl_path
