"""Shadow dual-run parity and rollback flag behavior."""

from __future__ import annotations

from pathlib import Path

from src.core.safety.news_guard import NewsGuard
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.safety_gate_metrics import alert_levels, aggregate_gate_shares
from src.core.safety.safety_gate_types import GateContext

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    (
        "Над регионами сбили БПЛА",
        "В Белгородской области силы ПВО сбили несколько БПЛА, объявлена опасность атаки.",
        "deny",
    ),
    (
        "Футбольный матч чемпионата",
        "Сборная сыграла товарищеский матч чемпионата без политических заявлений.",
        "skip",
    ),
    (
        "Инфляция и бюджет",
        "Минфин сообщил о исполнении бюджета и росте тарифов для населения.",
        None,
    ),
    (
        "Силовики отразили обстрел",
        "Российские силовики отразили обстрел на границе; войска ведут ответный огонь.",
        "deny",
    ),
]


def test_shadow_parity_on_stratified_sample() -> None:
    gate = SafetyGate.from_base_dir(ROOT)
    legacy = NewsGuard.from_file(ROOT / "config" / "news_guard.yaml")
    matches = 0
    for title, content, _expected in CASES:
        compare = gate.evaluate_with_shadow(
            GateContext(title=title, content=content, source="TASS"),
            legacy_guard=legacy,
        )
        if compare.decision_match:
            matches += 1
        # Shadow mode must enforce legacy path.
        assert compare.enforced.decision == compare.old_decision.decision
    assert matches / len(CASES) >= 0.95


def test_red_suite_zero_leakage() -> None:
    """Known red/borderline cases must never be allow."""
    gate = SafetyGate.from_base_dir(ROOT)
    red_cases = [
        (
            "Над регионами сбили БПЛА",
            "В Белгородской области силы ПВО сбили несколько БПЛА.",
        ),
        (
            "Силовики отразили обстрел",
            "Российские силовики отразили обстрел на границе; войска ведут ответный огонь.",
        ),
        (
            "Опасность атаки БПЛА",
            "В Курской области объявлена опасность атаки БПЛА и воздушная тревога.",
        ),
    ]
    for title, content in red_cases:
        decision = gate.evaluate(
            GateContext(title=title, content=content, source="TASS")
        )
        assert decision.decision != "allow"
        assert decision.risk_tier == "red" or decision.decision == "deny"


def test_enforce_mode_old_rollback() -> None:
    gate = SafetyGate.from_base_dir(ROOT)
    assert gate.config.flags.enforce_mode == "old"
    assert gate.config.flags.shadow_mode is True
    compare = gate.evaluate_with_shadow(
        GateContext(
            title="Инфляция и тарифы",
            content="Рост тарифов и инфляции в экономике.",
            source="TASS",
        )
    )
    assert compare.enforced.decision == compare.old_decision.decision


def test_alert_levels_and_shares() -> None:
    rows = [
        {"decision": "allow", "risk_tier": "green"},
        {"decision": "deny", "risk_tier": "red"},
        {"decision": "skip", "risk_tier": "green"},
        {"decision": "allow", "risk_tier": "yellow"},
    ]
    shares = aggregate_gate_shares(rows)
    assert abs(shares["gate_allow_share"] - 0.5) < 1e-9
    alerts = alert_levels(
        mismatch_rate=0.02,
        red_allow_leak_rate=0.0,
        yellow_share_delta=0.01,
        mean_output_chars_delta=-0.05,
        template_share=0.03,
        deny_rate_delta=0.0,
    )
    assert alerts["level"] == "ok"
