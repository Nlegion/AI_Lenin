"""Convert live NewsFetcher rows into QaItem list."""

from __future__ import annotations

from scripts.lib._quality_qa_io import QaItem
from src.modules.news_system.fetcher import NewsFetcher


def news_row_to_qa_item(row: dict) -> QaItem:
    title = str(row.get("title") or "").strip()
    content = str(row.get("content") or title).strip()
    item_id = str(row.get("id") or "").strip()
    source = str(row.get("source") or "").strip()
    if not item_id or not title or not content:
        raise ValueError(f"Invalid news row: id/title/content required, got keys={list(row)}")
    question = f"Прокомментируйте с позиций Ленина: {title}"
    return QaItem(
        id=item_id,
        title=title,
        content=content,
        question=question,
        topic="live",
        source=source,
    )


def fetch_live_qa_items(*, fetch_limit: int = 0) -> list[QaItem]:
    raw = NewsFetcher().fetch_all()
    items: list[QaItem] = []
    seen: set[str] = set()
    for row in raw:
        item = news_row_to_qa_item(row)
        if item.id in seen:
            continue
        seen.add(item.id)
        items.append(item)
        if fetch_limit > 0 and len(items) >= fetch_limit:
            break
    return items
