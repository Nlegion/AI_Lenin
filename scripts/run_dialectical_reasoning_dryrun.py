#!/usr/bin/env python
"""Dry-run dialectical reasoning engine on a fixture brief (no Telegram)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.analysis.evidence_brief import EvidenceBrief, EvidenceItem
from src.core.dialectics.config import DialecticalReasoningConfig
from src.core.dialectics.engine import DialecticalEngine
from src.core.dialectics.rag_brief import build_principle_cards
from src.core.dialectics.schemas import DialecticalRequest
from tests.helpers.dialectics_mocks import MockBackend

QUOTE = (
    "Монополии срастаются с государственным аппаратом и перекладывают "
    "издержки кризиса на трудящихся через регулирование."
)


def _demo_brief() -> EvidenceBrief:
    def item(cid: str, stance: str) -> EvidenceItem:
        return EvidenceItem(
            stance_type=stance,
            source_id=cid,
            source_path=f"demo/{cid}",
            chunk_id=cid,
            text=QUOTE,
            score=0.9,
            retriever="dense",
            query_used="demo",
        )

    return EvidenceBrief(
        news_title="demo",
        news_content="demo",
        axes=[],
        key_concepts=[],
        r1_core_self=[item("c1", "core_self")],
        r2_influence_agree=[item("c2", "influence_agree")],
        r3_influence_critical=[],
    )


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="neftegaz")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    fixture_path = ROOT / "tests" / "fixtures" / "dialectics" / f"{args.fixture}.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8")) if fixture_path.is_file() else {}
    title = str(payload.get("news_title") or "Демо новость")
    content = str(payload.get("news_content") or "Демо содержание")
    brief = _demo_brief()
    cfg = DialecticalReasoningConfig(fixture_mode=True)
    cards = build_principle_cards(brief, config=cfg)
    backend = MockBackend(
        mode="valid",
        principle_id=cards[0].principle_id if cards else "pc-x",
        chunk_id="c1",
    )
    engine = DialecticalEngine(backend=backend, config=cfg)
    result = await engine.analyze(
        request=DialecticalRequest(news_title=title, news_content=content, fixture_mode=True),
        brief=brief,
        enable_repair=bool(args.repair),
    )
    print(json.dumps(
        {
            "outcome": result.outcome,
            "reason_codes": result.reason_codes,
            "rendered_text": result.rendered_text,
            "timings_ms": result.pass_timings_ms,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
