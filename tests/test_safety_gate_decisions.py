"""SafetyGate decision contract and yellow hint propagation."""

from __future__ import annotations

from pathlib import Path

from src.core.generation.prompt_adapter import build_chat_request
from src.core.safety.safety_gate import SafetyGate, apply_yellow_warning
from src.core.safety.safety_gate_types import GateContext, GateDecision, SafetyHint
from src.core.settings.safety_gate_config import load_safety_gate_config

ROOT = Path(__file__).resolve().parents[1]


def _gate() -> SafetyGate:
    return SafetyGate.from_base_dir(ROOT)


def test_safety_gate_drone_deny() -> None:
    gate = _gate()
    decision = gate.evaluate(
        GateContext(
            title="Над регионами сбили БПЛА",
            content="В Белгородской области силы ПВО сбили несколько БПЛА.",
            source="TASS",
        )
    )
    assert decision.decision == "deny"
    assert decision.risk_tier == "red"


def test_safety_gate_sport_skip() -> None:
    gate = _gate()
    decision = gate.evaluate(
        GateContext(
            title="Футбольный матч чемпионата",
            content="Сборная сыграла товарищеский матч чемпионата без политических заявлений.",
            source="TASS",
        )
    )
    assert decision.decision == "skip"


def test_safety_gate_economy_allow_or_yellow() -> None:
    gate = _gate()
    decision = gate.evaluate(
        GateContext(
            title="Инфляция и тарифы ЖКХ",
            content="Рост тарифов и инфляции обсуждают в правительстве и Госдуме.",
            source="TASS",
        )
    )
    assert decision.decision in {"allow", "quarantine"}
    if decision.risk_tier == "yellow" and decision.decision == "allow":
        assert SafetyHint.YELLOW_CONSTRAINED_ANALYSIS in decision.context_hints


def test_context_hints_in_prompt() -> None:
    req = build_chat_request(
        news_title="Санкции и торговля",
        news_content="Новые тарифы на экспорт.",
        context="Контекст о монополиях.",
        max_context_chars=2000,
        risk_tier="green",
        context_hints=[SafetyHint.YELLOW_CONSTRAINED_ANALYSIS.value],
    )
    assert "Режим ограниченного анализа" in req.system_prompt


def test_yellow_warning_injected_upstream() -> None:
    decision = GateDecision(
        decision="allow",
        risk_tier="yellow",
        reason="yellow",
        reason_codes=["risk_tier:yellow"],
        needs_yellow_warning=True,
    )
    text = apply_yellow_warning(
        analysis="Краткий анализ экономики.",
        decision=decision,
        warning_text="Ограниченный режим анализа.",
    )
    assert "Ограниченный режим анализа" in text
    assert text.startswith("Краткий")


def test_config_fallback_keys_tracked() -> None:
    cfg = load_safety_gate_config(
        path=ROOT / "config" / "safety_gate_config.yaml",
        news_guard_path=ROOT / "config" / "news_guard.yaml",
    )
    assert cfg.flags.enabled is True
    assert cfg.flags.shadow_mode is True
    assert cfg.flags.enforce_mode == "old"


def test_quality_postcheck_does_not_duplicate_yellow_policy() -> None:
    """When SafetyGate owns yellow, post-gen yellow filter stays off."""
    from src.core.settings.quality_postcheck_config import load_quality_postcheck_config

    qc = load_quality_postcheck_config(path=ROOT / "config" / "quality_postcheck.yaml")
    assert qc.yellow_output_filter_enabled is False
    assert qc.quote_postcheck_enforce_mode == "soft"
    assert qc.artifact_enforce_mode == "soft"
