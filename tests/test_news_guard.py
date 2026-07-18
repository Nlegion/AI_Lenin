from pathlib import Path

from src.core.safety.news_guard import NewsGuard, load_news_guard_config


def _guard() -> NewsGuard:
    config = load_news_guard_config(Path("config/news_guard.yaml"))
    return NewsGuard(config=config)


def test_input_gate_blocks_forbidden_topic():
    guard = _guard()
    result = guard.evaluate_input(
        title="Матч по футболу завершился",
        content="Спорт и развлекательное событие без социального значения.",
    )
    assert result.decision in {"deny", "quarantine"}


def test_input_gate_allows_economic_topic():
    guard = _guard()
    result = guard.evaluate_input(
        title="Рост инфляции и безработицы",
        content="Экономика переживает кризис и обсуждаются меры правительства.",
    )
    assert result.decision == "allow"


def test_output_guard_blocks_extremist_phrase_and_adds_disclaimer():
    guard = _guard()
    output = guard.guard_output("Необходимо к оружию и к свержение власти.")
    assert output.blocked is True
    assert "компьютерной симуляцией" in output.moderated_text


def test_output_guard_adds_disclaimer_for_safe_text():
    guard = _guard()
    output = guard.guard_output("Анализ империализма и монополий.")
    assert "компьютерной симуляцией" in output.moderated_text
