"""Runtime adapters: live/shadow/off writers for artifact pass."""

from __future__ import annotations

from src.core.generation.answer_body_cleanup import cleanup_answer_body
from src.core.generation.output_artifacts import final_public_scrub
from src.core.generation.postprocess_clean.contract import (
    PostProcessInput,
    PostProcessResult,
    has_required_triad,
    map_postprocess_status,
    resolve_clean_mode,
)
from src.core.generation.postprocess_clean.engine import run_postprocess
from src.core.generation.postprocess_clean.shadow import emit_shadow_record
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig


def apply_pre_guard_for_artifact(
    text: str,
    *,
    config: QualityPostcheckConfig,
    item_id: str | None = None,
    combat_sensitive: bool = False,
    skip_structure_enforce: bool = False,
) -> PostProcessResult:
    """Single writer for pre-guard. Shadow compares clone; never dual-mutates."""
    inp = PostProcessInput(
        raw_text=text,
        phase="pre_guard",
        combat_sensitive=combat_sensitive,
        item_id=item_id,
        skip_structure_enforce=skip_structure_enforce,
        config=config,
    )
    mode = resolve_clean_mode(config)
    if mode == "live":
        return run_postprocess(inp)
    live = _legacy_pre_guard(text=text, inp=inp, config=config)
    if mode == "shadow":
        cloned = run_postprocess(inp)
        emit_shadow_record(
            phase="pre_guard",
            live_text=live.cleaned_text,
            cloned_text=cloned.cleaned_text,
            live_codes=list(live.codes),
            cloned_codes=list(cloned.codes),
            cloned_status=cloned.status,
            item_id=item_id,
        )
    return live


def _legacy_pre_guard(
    *,
    text: str,
    inp: PostProcessInput,
    config: QualityPostcheckConfig,
) -> PostProcessResult:
    body = cleanup_answer_body(text=text, config=config)
    cleaned, public_codes = final_public_scrub(body.text)
    body_codes = list(body.codes)
    integrity_codes = list(body.metadata.get("integrity_codes") or [])
    hard_fail = bool(body.metadata.get("postprocess_hard_fail"))
    structure_error = False
    if not inp.skip_structure_enforce:
        structure_error = not has_required_triad(cleaned)
    return PostProcessResult(
        cleaned_text=cleaned,
        status=map_postprocess_status(
            postprocess_hard_fail=hard_fail,
            structure_error=structure_error,
        ),
        codes=[*body_codes, *public_codes],
        postprocess_hard_fail=hard_fail,
        structure_error=structure_error,
        integrity_error=bool(body.metadata.get("integrity_error")),
        integrity_codes=integrity_codes,
        body_cleanup_codes=list(body.metadata.get("body_cleanup_codes") or body_codes),
        integrity_enforce_mode=str(
            body.metadata.get("integrity_enforce_mode") or "soft"
        ),
    )
