"""Wire quote allowlist, loop fix, and grounded-element metadata into generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.core.analysis.evidence_brief import EvidenceBrief
from src.core.generation.grounded_element import has_r1_keyword_overlap
from src.core.generation.loop_detect import detect_and_fix_loops
from src.core.generation.output_artifacts import apply_artifact_pass, scrub_input_text
from src.core.generation.quote_allowlist import (
    extract_quote_candidates,
    quote_allowlist_present,
    usable_for_attribution,
    usable_for_context,
)
from src.core.generation.quote_mode import answer_has_quotes, strip_quotes
from src.core.generation.quote_postcheck import apply_quote_postcheck
from src.core.settings.quality_postcheck_config import (
    QualityPostcheckConfig,
    default_quality_postcheck_path,
    load_quality_postcheck_config,
)

# Re-export for pipeline/RAG scrub callers.
__all__ = [
    "load_postcheck",
    "resolve_quote_mode",
    "apply_quality_post_generate",
    "scrub_chunks_for_prompt",
]


def _has_required_structure(text: str) -> bool:
    return all(
        re.search(pattern=pattern, string=text, flags=re.IGNORECASE) is not None
        for pattern in (
            r"(?:^|\s)\*{0,2}факт\*{0,2}\s*:",
            r"(?:^|\s)\*{0,2}механизм\*{0,2}\s*:",
            r"(?:^|\s)\*{0,2}вывод\*{0,2}\s*:",
        )
    )


def _enforce_required_structure(text: str) -> tuple[str, bool, bool]:
    """Return (text, structure_ok, structure_error_flag).

    Never injects a fake «Механизм: анализ опирается…» stub. Missing labels
    are reported via structure_error so callers can hold/suppress publish.
    """
    if _has_required_structure(text):
        return text, True, False
    return text, False, True


def scrub_chunks_for_prompt(
    chunks: list[tuple[str, float, str]],
) -> tuple[list[tuple[str, float, str]], list[str]]:
    """Detect encoding artifacts in RAG chunk bodies (no blind repair)."""
    codes: list[str] = []
    scrubbed: list[tuple[str, float, str]] = []
    for cid, score, body in chunks:
        cleaned, hit_codes = scrub_input_text(body)
        codes.extend(hit_codes)
        scrubbed.append((cid, score, cleaned))
    return scrubbed, codes


def load_postcheck(base_dir: Path) -> QualityPostcheckConfig:
    path = default_quality_postcheck_path(base_dir)
    if path.is_file():
        return load_quality_postcheck_config(path=path)
    return QualityPostcheckConfig()


def resolve_quote_mode(
    *,
    base_mode: str,
    chunks: list[tuple[str, float, str]],
    config: QualityPostcheckConfig,
) -> tuple[str, list, dict[str, Any]]:
    candidates = extract_quote_candidates(chunks=chunks, config=config)
    flags = {
        "usable_for_context": usable_for_context(chunks),
        "quote_allowlist_present": quote_allowlist_present(candidates),
        "usable_for_attribution": usable_for_attribution(candidates),
        "allowlist_size": len(candidates),
    }
    if not config.quote_allowlist_enabled or not flags["quote_allowlist_present"]:
        return "principles", candidates, flags
    if base_mode == "quote":
        return "quote", candidates, flags
    return "principles", candidates, flags


def apply_quality_post_generate(
    *,
    text: str,
    chunks: list[tuple[str, float, str]],
    candidates: list,
    brief: EvidenceBrief | None,
    config: QualityPostcheckConfig,
    context_has_quotes: bool,
    item_id: str | None = None,
    combat_sensitive: bool = False,
    news_text: str | None = None,
    skip_structure_enforce: bool = False,
) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    working = text
    rag_empty = not usable_for_context(chunks)

    if news_text:
        _, news_codes = scrub_input_text(news_text)
        if news_codes:
            meta["news_artifact_codes"] = news_codes

    # Legacy strip when no context quotes and allowlist disabled path.
    if answer_has_quotes(working) and not context_has_quotes and not candidates:
        working = strip_quotes(working)
        meta["quote_postcheck_stripped"] = True

    if config.quote_allowlist_enabled:
        chunk_texts = {cid: body for cid, _s, body in chunks}
        qres = apply_quote_postcheck(
            text=working,
            candidates=candidates,
            config=config,
            chunk_texts=chunk_texts,
        )
        working = qres.text
        meta.update(
            {
                "quote_postcheck_codes": list(qres.codes),
                "quote_removed": qres.quote_removed,
                "critical_attribution_hallucination": qres.critical_attribution_hallucination,
                "path_leak_scrubbed": qres.path_leak_scrubbed,
                "used_static_template": qres.used_static_template,
                "quote_repair_applied": bool(qres.metadata.get("quote_repair_applied")),
                "repair_success": bool(
                    qres.metadata.get("repair_success", not qres.used_static_template)
                ),
                **qres.metadata,
            }
        )

    loop_res = detect_and_fix_loops(working, config=config, rag_empty=rag_empty)
    working = loop_res.text
    meta.update(
        {
            "paragraph_loop_detected": loop_res.loop_detected,
            "loop_action": loop_res.loop_action,
            **{f"loop_{k}": v for k, v in loop_res.metadata.items()},
        }
    )

    art = apply_artifact_pass(
        text=working,
        config=config,
        item_id=item_id,
        combat_sensitive=combat_sensitive,
    )
    working = art.text
    meta["artifact_ops"] = list(art.codes)
    meta["artifact_codes"] = [
        code
        for code in art.codes
        if code.startswith(("artifact:", "detect:", "fallback:", "deny:"))
        or code in {"broken_syntax", "too_short_after_strip"}
    ]
    meta["artifact_fallback"] = art.used_fallback
    meta["artifact_deny"] = art.deny
    meta["body_cleanup_codes"] = list(art.metadata.get("body_cleanup_codes") or [])
    meta["integrity_codes"] = list(art.metadata.get("integrity_codes") or [])
    meta["integrity_error"] = bool(art.metadata.get("integrity_error"))
    meta["postprocess_hard_fail"] = bool(art.metadata.get("postprocess_hard_fail"))
    meta["integrity_enforce_mode"] = str(
        art.metadata.get("integrity_enforce_mode")
        or getattr(config, "integrity_enforce_mode", "soft")
    )

    if config.grounded_element_check_enabled and brief is not None:
        r1_text = "\n".join(item.text for item in brief.r1_core_self)
        has_quote = bool(candidates) and answer_has_quotes(working)
        has_concept = has_r1_keyword_overlap(analysis=working, r1_text=r1_text)
        meta["grounded_element"] = has_quote or has_concept
        meta["insufficient_context"] = bool(r1_text) and not (has_quote or has_concept)

    if skip_structure_enforce:
        meta["structure_rebuilt"] = False
        meta["structure_ok"] = _has_required_structure(working)
        meta["structure_error"] = not bool(meta["structure_ok"])
        meta["legacy_stub_rebuild"] = False
    else:
        working, structure_ok, structure_error = _enforce_required_structure(working)
        # structure_rebuilt kept for metrics scripts; always False (no stub).
        meta["structure_rebuilt"] = False
        meta["structure_ok"] = structure_ok
        meta["structure_error"] = structure_error
        meta["legacy_stub_rebuild"] = False

    return working, meta
