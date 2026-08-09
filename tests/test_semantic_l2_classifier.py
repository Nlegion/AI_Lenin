from __future__ import annotations

from src.core.safety.semantic_l2_classifier import SemanticL2Classifier


def test_l2_classifier_uses_truncated_text_window() -> None:
    classifier = SemanticL2Classifier(
        model_version="test-v1",
        prototypes={"WAR_OPERATIONAL": ("битв",), "SENSITIVE_POLITICS": ("эконом",)},
        text_max_chars=64,
        cache_size=256,
    )
    text = ("экономика " * 20) + "битва"
    score = classifier.score(text)
    assert score.category == "SENSITIVE_POLITICS"


def test_l2_classifier_cache_is_bounded() -> None:
    classifier = SemanticL2Classifier(
        model_version="test-v1",
        prototypes={"SENSITIVE_POLITICS": ("эконом",)},
        text_max_chars=256,
        cache_size=128,
    )
    for idx in range(500):
        classifier.score(f"экономика {idx}")
    assert len(classifier._cache) <= 128  # type: ignore[attr-defined]

