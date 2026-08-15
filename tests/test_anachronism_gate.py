"""Anti-anachronism gate unit tests."""

from __future__ import annotations

from src.core.safety.anachronism_gate import anachronism_gate
from src.core.settings.gate_constants import ANACHRONISM_CODE_FIRST_PERSON_TECH


def test_bare_tech_mention_passes() -> None:
    result = anachronism_gate(
        analysis="В новости говорится о росте продаж смартфонов в регионе."
    )
    assert result.reason_codes == []
    assert result.blocked is False


def test_first_person_tech_warns() -> None:
    result = anachronism_gate(
        analysis="Я пользовался TikTok и видел там обсуждение стачки."
    )
    assert ANACHRONISM_CODE_FIRST_PERSON_TECH in result.reason_codes


def test_quoted_expert_passes() -> None:
    result = anachronism_gate(
        analysis="Эксперт сказал: «Я пользуюсь TikTok каждый день», — сообщает агентство."
    )
    assert result.reason_codes == []


def test_attribution_near_cue_passes() -> None:
    result = anachronism_gate(
        analysis="По словам журналиста, я пользовался смартфоном на митинге — так пересказали очевидцы."
    )
    # Attribution cue before first-person should exempt
    assert result.reason_codes == []


def test_fail_open(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise RuntimeError("x")

    monkeypatch.setattr("src.core.safety.anachronism_gate._evaluate", _boom)
    result = anachronism_gate(analysis="я пользовался tiktok")
    assert result.skipped is True
    assert result.skip_reason and result.skip_reason.startswith("error:")
