from src.core.retrieval.query_transform import (
    build_hyde_query,
    decompose_query,
    rewrite_query_to_philosophical_register,
)


def test_rewrite_query_to_philosophical_register_adds_ideology_terms():
    text = "Санкции и инфляция усилились в стране"
    rewritten = rewrite_query_to_philosophical_register(text)
    assert "империалистические экономические санкции" in rewritten
    assert "кризис капиталистического воспроизводства" in rewritten


def test_decompose_query_returns_factual_and_evaluative_parts():
    text = "В 2026 году выросла безработица. Это усилило социальное напряжение."
    factual, evaluative = decompose_query(text)
    assert factual
    assert evaluative
    assert factual != evaluative


def test_build_hyde_query_contains_original_theme():
    text = "Рост цен на энергоносители"
    hyde = build_hyde_query(text)
    assert text in hyde
