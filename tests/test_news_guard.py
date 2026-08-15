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
    assert result.decision in {"deny", "quarantine", "skip"}


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
    assert "образовательных целях" in output.moderated_text


def test_output_guard_adds_disclaimer_for_safe_text():
    guard = _guard()
    output = guard.guard_output("Анализ империализма и монополий.")
    assert "образовательных целях" in output.moderated_text


def test_input_gate_blocks_military_context_even_without_exact_keyword():
    guard = _guard()
    result = guard.evaluate_input(
        title="Обсуждение действий армии России в приграничном районе",
        content="Материал о военных подразделениях РФ.",
        source="TASS",
    )
    assert result.decision == "deny"


def test_output_guard_classifier_marks_extremism_in_strict_mode():
    guard = _guard()
    output = guard.guard_output("Этот текст содержит экстремистский призыв.")
    assert output.blocked is True
    assert any(code.startswith("classifier:") for code in output.reason_codes)


def test_input_gate_denies_untrusted_source_for_high_risk_topic():
    guard = _guard()
    result = guard.evaluate_input(
        title="Сообщения о ЧП",
        content="Источник сообщает о чрезвычайном происшествии и социальных волнениях.",
        source="UnknownWire",
    )
    assert result.decision == "deny"
    assert "доверенных" in result.message


def test_input_gate_denies_private_pii_without_public_interest():
    guard = _guard()
    result = guard.evaluate_input(
        title="Сосед пожаловался на шум",
        content="Иванов Иван Иванович, тел. 900-111-22-33, рассказал о семейной ссоре.",
        source="UnknownBlog",
    )
    assert result.decision == "deny"


def test_output_guard_redacts_hallucinated_pii():
    guard = _guard()
    output = guard.guard_output(
        analysis="Контакт: Иванов Иван Иванович, тел. 900-111-22-33.",
        source_text="Рост инфляции и безработицы.",
        warn_only=True,
    )
    assert "«[место]»" in output.moderated_text or "обезличено" in output.moderated_text
    assert any(code.startswith("pii_redact:") for code in output.reason_codes)


def test_mark_unverified_facts_adds_stylized_marker():
    guard = _guard()
    marked, codes = guard.mark_unverified_facts(
        analysis='Как я писал: "несуществующая цитата".',
        retrieval_context="инфляция и капиталистический кризис",
    )
    assert "В стилизованной интерпретации" in marked
    assert "hallucination_marked" in codes


def test_disclaimer_is_footer_in_strict_public_mode():
    guard = _guard()
    output = guard.guard_output("Анализ империализма и монополий.")
    assert output.moderated_text.startswith("Анализ империализма")
    assert "образовательных целях" in output.moderated_text
    assert output.moderated_text.index("Анализ") < output.moderated_text.index(
        "образовательных"
    )
