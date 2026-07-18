from src.core.preprocessing.cleaning_config import CleaningConfig
from src.core.preprocessing.cleaning_quality import semantic_damage_ratio
from src.core.preprocessing.text_cleaner import clean_document


def _config() -> CleaningConfig:
    return CleaningConfig(
        remove_line_patterns=[r"^ISBN.*$", r"^Тираж.*$"],
        remove_inline_patterns=[r"\bстр\.\s*\d+\b"],
        content_start_markers=[r"^ВВЕДЕНИЕ"],
        min_cleaned_chars=10,
        min_semantic_paragraph_chars=20,
        semantic_overlap_threshold=0.4,
        validation_sample_size=3,
        max_semantic_damage_ratio=0.02,
    )


def test_clean_document_removes_technical_noise():
    text = (
        "Титульный лист\n"
        "ISBN 1234\n"
        "ВВЕДЕНИЕ\n"
        "Это содержательный абзац о материализме и диалектике.\n\n"
        "Тираж 5000\n"
        "Еще один содержательный абзац стр. 10 про классовую борьбу."
    )
    cleaned = clean_document(text=text, config=_config())
    assert "ISBN" not in cleaned
    assert "Тираж" not in cleaned
    assert "стр. 10" not in cleaned
    assert "материализме" in cleaned
    assert "классовую борьбу" in cleaned


def test_semantic_damage_ratio_low_when_text_preserved():
    original = (
        "Материализм утверждает первичность материи и объективной реальности.\n\n"
        "Диалектика рассматривает развитие через противоречия и их разрешение."
    )
    cleaned = (
        "Материализм утверждает первичность материи и объективной реальности.\n\n"
        "Диалектика рассматривает развитие через противоречия и их разрешение."
    )
    damage = semantic_damage_ratio(
        original_text=original,
        cleaned_text=cleaned,
        min_paragraph_chars=20,
        overlap_threshold=0.4,
    )
    assert damage == 0.0
