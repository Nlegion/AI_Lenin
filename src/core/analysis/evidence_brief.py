"""Evidence brief data structures for dialectical R1–R3 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.settings.dialectical_constants import TRACE_QUERY_MAX_CHARS


@dataclass(frozen=True)
class EvidenceItem:
    stance_type: str
    source_id: str
    source_path: str
    chunk_id: str
    text: str
    score: float
    retriever: str
    query_used: str
    multi_stance: bool = False


@dataclass
class EvidenceBrief:
    news_title: str
    news_content: str
    axes: list[str]
    key_concepts: list[str]
    r1_core_self: list[EvidenceItem] = field(default_factory=list)
    r2_influence_agree: list[EvidenceItem] = field(default_factory=list)
    r3_influence_critical: list[EvidenceItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    legacy_context: str | None = None

    def mark_multi_stance(self) -> None:
        counts: dict[str, int] = {}
        for item in [
            *self.r1_core_self,
            *self.r2_influence_agree,
            *self.r3_influence_critical,
        ]:
            counts[item.chunk_id] = counts.get(item.chunk_id, 0) + 1
        multi_ids = {chunk_id for chunk_id, count in counts.items() if count > 1}
        self.trace["multi_slot_chunk_ids"] = sorted(multi_ids)

        def _remap(items: list[EvidenceItem]) -> list[EvidenceItem]:
            return [
                EvidenceItem(
                    stance_type=item.stance_type,
                    source_id=item.source_id,
                    source_path=item.source_path,
                    chunk_id=item.chunk_id,
                    text=item.text,
                    score=item.score,
                    retriever=item.retriever,
                    query_used=item.query_used,
                    multi_stance=item.chunk_id in multi_ids,
                )
                for item in items
            ]

        self.r1_core_self = _remap(self.r1_core_self)
        self.r2_influence_agree = _remap(self.r2_influence_agree)
        self.r3_influence_critical = _remap(self.r3_influence_critical)

    def render_for_prompt(self) -> str:
        sections = [
            ("## R1 — Ленин (core_self)", self.r1_core_self),
            ("## R2 — Опоры (influence_agree)", self.r2_influence_agree),
            (
                "## R3 — Критика / оппозиция (influence_critical)",
                self.r3_influence_critical,
            ),
        ]
        parts: list[str] = []
        if self.axes:
            parts.append("## Оси\n" + "\n".join(f"- {axis}" for axis in self.axes))
        for header, items in sections:
            parts.append(header)
            if not items:
                parts.append("(пусто)")
                continue
            for index, item in enumerate(items, start=1):
                marker = "[multi-stance] " if item.multi_stance else ""
                source = item.source_path or item.source_id
                quote = item.text.strip() or "(без текста)"
                parts.append(f'{marker}[{index}] ({source}) "{quote}"')
        return "\n\n".join(parts)


def truncate_query_for_trace(query: str) -> dict[str, str | bool]:
    cleaned = query.strip()
    if len(cleaned) <= TRACE_QUERY_MAX_CHARS:
        return {"query": cleaned, "truncated": False}
    return {"query": cleaned[:TRACE_QUERY_MAX_CHARS], "truncated": True}
