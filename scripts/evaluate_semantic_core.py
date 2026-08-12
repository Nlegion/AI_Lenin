"""Evaluate semantic-core router accuracy and lacuna-hedge patterns (shadow)."""

from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.analysis.author_normalize import is_lenin_author  # noqa: E402
from src.core.analysis.semantic_core_config import load_semantic_core_config  # noqa: E402
from src.core.analysis.semantic_integration import (  # noqa: E402
    cliche_gate_blocks_enable,
    legacy_enable_decision,
)
from src.core.analysis.topic_router import route_topics  # noqa: E402
from src.core.safety.lacuna_hedge_gate import lacuna_hedge_gate  # noqa: E402

logger = logging.getLogger(__name__)


def _load_fixtures(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def evaluate_router(fixtures: list[dict], config) -> dict:
    enabled = replace(config, enabled=True)
    total = 0
    correct = 0
    details: list[dict] = []
    for row in fixtures:
        if "lacuna_sample" in row and row.get("expect_route") is False and not row.get("expected_topic"):
            if row.get("title", "").startswith("Проверка lacuna"):
                continue
        total += 1
        result = route_topics(
            news_title=row["title"],
            news_content=row["content"],
            config=enabled,
        )
        expected = row.get("expected_topic")
        expect_route = bool(row.get("expect_route", True))
        ok = (
            (result.dominant_topic_id == expected)
            if expect_route
            else (result.dominant_topic_id is None)
        )
        correct += int(ok)
        details.append(
            {
                "id": row.get("id"),
                "ok": ok,
                "got": result.dominant_topic_id,
                "expected": expected,
            }
        )
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "details": details,
    }


def evaluate_lacuna(fixtures: list[dict]) -> dict:
    samples = [row["lacuna_sample"] for row in fixtures if row.get("lacuna_sample")]
    hits = 0
    for sample in samples:
        result = lacuna_hedge_gate(analysis=sample)
        hits += int(bool(result.reason_codes))
    return {
        "samples": len(samples),
        "detected": hits,
        "recall": (hits / len(samples)) if samples else 0.0,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    fixtures_path = ROOT / "data" / "eval" / "semantic_core_fixtures.jsonl"
    config = load_semantic_core_config(path=ROOT / "config" / "semantic_core.yaml")
    fixtures = _load_fixtures(fixtures_path)
    router_report = evaluate_router(fixtures=fixtures, config=config)
    lacuna_report = evaluate_lacuna(fixtures=fixtures)
    report = {
        "router": router_report,
        "lacuna": lacuna_report,
        "legacy_policy_example": {
            "author_known_rate_min": config.author_known_rate_min,
            "enable_when_known_rate_low_without_human": legacy_enable_decision(
                author_known_rate=0.4,
                author_known_rate_min=config.author_known_rate_min,
                human_scores_available=False,
            ),
            "is_lenin_author_vi": is_lenin_author("Ленин ВИ"),
        },
        "cliche_compound_gate_example": {
            "blocks": cliche_gate_blocks_enable(
                warn_rate_off=0.01,
                warn_rate_on=0.02,
                max_ratio=config.cliche_warn_rate_max_ratio,
                min_delta_pp=config.cliche_warn_rate_min_delta_pp,
            ),
        },
    }
    out_dir = ROOT / ".cursor" / "artifacts" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "semantic_core_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if router_report["accuracy"] < 1.0:
        logger.warning("router_accuracy_below_one accuracy=%s", router_report["accuracy"])
        return 1
    if lacuna_report["recall"] < 1.0:
        logger.warning("lacuna_recall_below_one recall=%s", lacuna_report["recall"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
