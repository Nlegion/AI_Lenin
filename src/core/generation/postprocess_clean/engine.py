"""Two-phase dispatcher: pre_guard body+public, post_guard public-only."""

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
from src.core.generation.postprocess_clean.shadow import emit_shadow_record
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig


def run_postprocess(inp: PostProcessInput) -> PostProcessResult:
    """Single contract entry. Callers pick phase; module owns rule order."""
    cfg = inp.config or QualityPostcheckConfig()
    if inp.phase == "post_guard":
        return _run_post_guard(text=inp.raw_text)
    return _run_pre_guard(text=inp.raw_text, inp=inp, config=cfg)


def _run_pre_guard(
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
    integrity_error = bool(body.metadata.get("integrity_error"))
    structure_error = False
    if not inp.skip_structure_enforce:
        structure_error = not has_required_triad(cleaned)
    status = map_postprocess_status(
        postprocess_hard_fail=hard_fail,
        structure_error=structure_error,
        deny=False,
    )
    codes = [*body_codes, *public_codes]
    details = None
    if status != "clean":
        flagged = [
            code for code in (*integrity_codes, *codes) if code.startswith("deny:")
        ]
        details = ",".join(
            flagged
            or integrity_codes
            or (["structure_error"] if structure_error else [])
        )
    return PostProcessResult(
        cleaned_text=cleaned,
        status=status,
        codes=codes,
        error_details=details or None,
        postprocess_hard_fail=hard_fail,
        structure_error=structure_error,
        integrity_error=integrity_error,
        integrity_codes=integrity_codes,
        body_cleanup_codes=list(body.metadata.get("body_cleanup_codes") or body_codes),
        integrity_enforce_mode=str(
            body.metadata.get("integrity_enforce_mode") or "soft"
        ),
    )


def _run_post_guard(*, text: str) -> PostProcessResult:
    cleaned, codes = final_public_scrub(text)
    return PostProcessResult(
        cleaned_text=cleaned,
        status="clean",
        codes=list(codes),
        error_details=None,
        postprocess_hard_fail=False,
        structure_error=False,
    )


def apply_terminal_public_scrub(
    text: str,
    *,
    quality_meta: dict | None = None,
    item_id: str | None = None,
    config: QualityPostcheckConfig | None = None,
) -> str:
    """Mandatory post-NewsGuard/yellow (and persist/publish re-guard) scrub."""
    cfg = config or QualityPostcheckConfig()
    mode = resolve_clean_mode(cfg)
    if mode != "live":
        cleaned, codes = final_public_scrub(text)
        if mode == "shadow":
            cloned = run_postprocess(
                PostProcessInput(
                    raw_text=text,
                    phase="post_guard",
                    item_id=item_id,
                    config=cfg,
                )
            )
            emit_shadow_record(
                phase="post_guard",
                live_text=cleaned,
                cloned_text=cloned.cleaned_text,
                live_codes=list(codes),
                cloned_codes=list(cloned.codes),
                cloned_status=cloned.status,
                item_id=item_id,
            )
        _write_terminal_meta(quality_meta, codes=codes, status="clean")
        return cleaned
    result = run_postprocess(
        PostProcessInput(
            raw_text=text,
            phase="post_guard",
            item_id=item_id,
            config=cfg,
        )
    )
    _write_terminal_meta(
        quality_meta,
        codes=result.codes,
        status=result.status,
    )
    return result.cleaned_text


def _write_terminal_meta(
    quality_meta: dict | None,
    *,
    codes: list[str],
    status: str,
) -> None:
    if quality_meta is None:
        return
    if codes:
        quality_meta["final_public_scrub_codes"] = list(codes)
    quality_meta["postprocess_status_post_guard"] = status
    quality_meta["postprocess_codes_post_guard"] = list(codes)


def scrub_after_output_guard(text: str) -> str:
    """Invariant helper: no guard pass may leave public markers unsanitized."""
    return apply_terminal_public_scrub(text)
