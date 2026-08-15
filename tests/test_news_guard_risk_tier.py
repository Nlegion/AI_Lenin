"""Risk-tier / economy yellow / soft-skip regressions (Quality QA plan)."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.safety.news_guard import NewsGuard

ROOT = Path(__file__).resolve().parents[1]
GUARD = NewsGuard.from_file(ROOT / "config" / "news_guard.yaml")


def test_insurance_militarized_economy_allow_or_yellow() -> None:
    result = GUARD.evaluate_input(
        title="Страховка транспорта от военных угроз",
        content="Страховые компании расширяют покрытие грузов от военных рисков на маршрутах.",
        source="TASS",
    )
    assert result.decision == "allow"
    assert result.risk_tier in {"green", "yellow"}


def test_railway_economy_not_hard_deny_without_combat() -> None:
    result = GUARD.evaluate_input(
        title="Армения о требованиях по ЮКЖД",
        content="Ереван выдвинул экономические требования по ЮКЖД и транзиту грузов.",
        source="TASS",
    )
    assert result.decision == "allow"
    assert result.risk_tier in {"green", "yellow"}


def test_true_combat_stays_red() -> None:
    result = GUARD.evaluate_input(
        title="Удар по позициям",
        content="Артиллерийский удар ВСУ по позициям; войска ведут обстрел населённого пункта.",
        source="TASS",
    )
    assert result.decision == "deny"
    assert result.risk_tier == "red"


def test_svo_stays_red() -> None:
    result = GUARD.evaluate_input(
        title="Встреча для участников СВО",
        content="В регионе прошла встреча участников специальной военной операции.",
        source="TASS",
    )
    assert result.decision == "deny"
    assert result.risk_tier == "red"


def test_sport_soft_skip_template() -> None:
    result = GUARD.evaluate_input(
        title="Марафон и футбольный матч",
        content="Сборная сыграла товарищеский матч чемпионата.",
        source="TASS",
    )
    assert result.decision == "skip"
    assert "коммерциализации" in result.message or "вне сферы" in result.message.lower()


def test_sport_policy_exception_labor() -> None:
    result = GUARD.evaluate_input(
        title="Забастовка футболистов",
        content="Профсоюз игроков объявил забастовку из-за зарплат и контрактов.",
        source="TASS",
    )
    assert result.decision == "allow"


def test_mixed_economy_combat_stays_red() -> None:
    result = GUARD.evaluate_input(
        title="Страховые выплаты после удара",
        content="Страховые выплаты после ракетного удара по складу войск на фронте.",
        source="TASS",
    )
    assert result.decision == "deny"
    assert result.risk_tier == "red"


def test_yellow_output_blocks_tactics() -> None:
    out = GUARD.guard_output(
        analysis="Нужно разобрать перемещение войск и тактику удара на фронте.",
        risk_tier="yellow",
    )
    assert out.blocked is True


def test_red_gold_no_false_accept() -> None:
    path = ROOT / "data" / "eval" / "red_gold_combat.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 50
    false_accept = 0
    for row in rows:
        result = GUARD.evaluate_input(
            title=row["title"], content=row["content"], source="TASS"
        )
        if result.decision != "deny":
            false_accept += 1
    assert false_accept == 0
