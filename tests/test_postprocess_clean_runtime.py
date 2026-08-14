"""Shadow mode, passthrough identity, and persist/publish guard scrub."""

from __future__ import annotations

from src.core.generation.postprocess_clean.adapter import apply_pre_guard_for_artifact
from src.core.generation.postprocess_clean.passthrough import passthrough_pipeline_text
from src.core.generation.postprocess_clean.shadow import emit_shadow_record, shadow_log_path
from src.core.generation.publishability import is_error_placeholder
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig


def test_shadow_does_not_change_live_writer() -> None:
    text = (
        "Факт: событие. Механизм: анализ. Вывод: итог. "
        "— Ленин (core_approval)"
    )
    off_cfg = QualityPostcheckConfig(postprocess_clean_mode="off")
    shadow_cfg = QualityPostcheckConfig(postprocess_clean_mode="shadow")
    live_cfg = QualityPostcheckConfig(postprocess_clean_mode="live")
    legacy = apply_pre_guard_for_artifact(text, config=off_cfg, item_id="s1")
    shadowed = apply_pre_guard_for_artifact(text, config=shadow_cfg, item_id="s1")
    live = apply_pre_guard_for_artifact(text, config=live_cfg, item_id="s1")
    assert shadowed.cleaned_text == legacy.cleaned_text
    assert live.cleaned_text == legacy.cleaned_text
    assert "Ленин (core_approval)" not in live.cleaned_text


def test_shadow_log_writes_jsonl(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"

    def _path(*, base_dir=None):
        return path

    import src.core.generation.postprocess_clean.shadow as shadow_mod

    original = shadow_mod.shadow_log_path
    shadow_mod.shadow_log_path = _path  # type: ignore[method-assign]
    try:
        emit_shadow_record(
            phase="pre_guard",
            live_text="a",
            cloned_text="a",
            live_codes=["strip:x"],
            cloned_codes=["strip:x"],
            cloned_status="clean",
            item_id="n1",
            base_dir=tmp_path,
        )
        assert path.is_file()
        line = path.read_text(encoding="utf-8").strip()
        assert "pre_guard" in line
        assert "text_equal" in line
    finally:
        shadow_mod.shadow_log_path = original


def test_passthrough_does_not_re_mutate_triad() -> None:
    text = (
        "Факт: событие произошло.\n"
        "Механизм: концентрация капитала.\n"
        "Вывод: нужен контроль классов.\n"
        "Ответ сгенерирован ИИ в образовательных целях."
    )
    assert passthrough_pipeline_text(text) == text
    assert passthrough_pipeline_text("") == "Не удалось сгенерировать анализ."
    assert is_error_placeholder("Ошибка анализа.")
    assert passthrough_pipeline_text("Ошибка анализа.") == "Ошибка анализа."


def test_shadow_log_path_default_under_artifacts() -> None:
    path = shadow_log_path()
    assert path.name == "postprocess_clean_shadow.jsonl"
    assert "quality" in path.parts
