"""Pre-guard answer body cleanup: stance/instruction scrub, triad truncate, soft integrity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.core.generation.output_artifacts import detect_encoding_artifacts
from src.core.settings.quality_postcheck_config import QualityPostcheckConfig

_STANCE_CORE = (
    r"agreement|disagreement|core_approval|core_criticism|core_self|core_disapproval|"
    r"core_согласие|core_критика"
)
# Known allowlist forms + broken echoes like «Ленин (core_ Lenin )».
_INLINE_STANCE_LENIN = re.compile(
    rf"(?i)[ \t]*[-—*]+[ \t]*\**[ \t]*(?:ленин|lenin)[ \t]*"
    rf"\((?:(?:{_STANCE_CORE})[^)]*|[^)]*core_[^)]*)\)[ \t]*\**[ \t]*[).]*"
)
_INLINE_STANCE_RU_LABEL = re.compile(
    r"(?i)[ \t]*[-—*]+[ \t]*\**[ \t]*(?:согласие|критика|полемика)[ \t]*"
    r"\(core_[a-z_а-яё]+\)[ \t]*\**[ \t]*[).]*"
)
_INSTRUCTION_DUMP = re.compile(
    r"(?i)(?:^|\s)запрещено\s+(?:комментировать|выдумывать|использовать|выдавать|"
    r"выводить|анализировать)[^.]*\."
)
# Prompt-tail requires several markers together (avoid generic slash cuts).
_PROMPT_TASK_TAIL = re.compile(
    r"(?i)(?:\s*---\s*)*задача\s*:\s*краткий\s+анализ(?:\s+в\s+стиле\s+ленина)?"
    r"[^\n]*(?:\bR[123]\b|/)?[^\n]*$"
)
_STRAY_ASTERISK_LINE = re.compile(r"(?m)^\s*\*+\s*$")
# Section-boundary bold labels: start / newline / after sentence punctuation.
_INLINE_BOLD_LABEL = re.compile(
    r"(?mi)(?:^|(?<=[.!?])\s+|\n)\s*\*{1,2}(факт|механизм|вывод)\*{0,2}\s*:\s*\*{0,2}\s*",
)
_LABEL_BOLD_JUNK = re.compile(
    r"(?mi)^(\s*\*{0,2})(факт|механизм|вывод)(\*{0,2})\s*:\s*\*+\s*",
)
# Any triad label with optional spaces around colon (line or inline).
_LABEL_SPACING = re.compile(
    r"(?mi)(?<![\wА-Яа-яЁё*])\*{0,2}(факт|механизм|вывод)\*{0,2}[ \t]*:[ \t]*",
)
# Any triad label not glued to a preceding word (covers --- debris / flatten).
_SECTION_LABEL = re.compile(
    r"(?mi)(?<![\wА-Яа-яЁё*])\*{0,2}(факт|механизм|вывод)\*{0,2}\s*:",
)
_INLINE_SECTION_RESTART = re.compile(
    r"(?mi)(?<=[.!?…])(?:\s|---|\[[^\]]*\])*?\*{0,2}(факт|механизм|вывод)\*{0,2}\s*:",
)
_TRAILING_MD_BEFORE_CUT = re.compile(
    r"(?:\s*---(?:\s*\[[^\]]*\])?\s*)+$",
)
# Terminal markdown debris only (not global ---/##).
_TERMINAL_MD_DEBRIS = re.compile(
    r"(?:\s*(?:---|##))+(?:\s*\.)?\s*$",
)
_INLINE_MD_DEBRIS_CLUSTER = re.compile(
    r"(?<=[.!?])\s*(?:(?:---|##)\s*){2,}(?:\.)?\s*",
)
_EMPTY_MD_SCAFFOLD = re.compile(
    r"(?i)\s*---\s*\[(?:empty|пусто)\]\s*---\s*",
)
_HOLE_PATTERNS = (
    re.compile(r"\bчто\s*,", re.IGNORECASE),
    re.compile(r"утверждение\s+о\s+и\b", re.IGNORECASE),
    re.compile(r"\bи\s+и\b", re.IGNORECASE),
    re.compile(r"\bо\s+может\b", re.IGNORECASE),
    re.compile(r"\bбез\s*,", re.IGNORECASE),
    re.compile(r"\bесть\s*,\s*который\b", re.IGNORECASE),
    re.compile(r"изложенные\s+в\s*\.", re.IGNORECASE),
)
_RESIDUAL_STANCE = re.compile(
    rf"(?i)(?:ленин|lenin)\s*\((?:(?:{_STANCE_CORE})[^)]*|[^)]*core_[^)]*)\)|"
    r"(?:согласие|критика|полемика)\s*\(core_[a-z_а-яё]+\)"
)
_RESIDUAL_PROMPT_TASK = re.compile(
    r"(?i)задача\s*:\s*краткий\s+анализ",
)
_RESIDUAL_MD_DEBRIS = re.compile(r"(?:---\s*){2,}|(?:##\s*){2,}|---\s*##|##\s*---")
_POST_STANCE_DEBRIS = re.compile(r"(?:\s*---\s*|\s*\)\s*\.?\s*){2,}")
_MESTO_MARKER = re.compile(r"«?\s*\[(?:место|обезличено)\]\s*»?", re.IGNORECASE)
_YELLOW_PREFIX = "ограниченный режим анализа"
_DISCLAIMER_HINT = "ответ сгенерирован"


@dataclass
class BodyCleanupResult:
    text: str
    codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_fact_body(text: str) -> str:
    cleaned = re.sub(r"\*{1,2}", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().casefold()
    return cleaned


def _protect_safety_tails(text: str) -> tuple[str, str]:
    """Split trailing yellow/disclaimer so scrub does not remove them."""
    lines = text.split("\n")
    keep_from = len(lines)
    for idx in range(len(lines) - 1, -1, -1):
        raw = lines[idx]
        low = raw.strip().casefold()
        if not low:
            continue
        disc_at = low.find(_DISCLAIMER_HINT)
        is_yellow_line = low.startswith(_YELLOW_PREFIX)
        if is_yellow_line or disc_at >= 0:
            # Content + disclaimer on one line (common after sentence flatten).
            if disc_at > 0:
                abs_pos = raw.casefold().rfind(_DISCLAIMER_HINT)
                content_part = raw[:abs_pos].rstrip()
                disc_part = raw[abs_pos:].lstrip()
                body_lines = (
                    [*lines[:idx], content_part] if content_part else list(lines[:idx])
                )
                body = "\n".join(body_lines).rstrip()
                tail = "\n".join([disc_part, *lines[idx + 1 :]]).strip("\n")
                return body, ("\n" + tail if body and tail else tail)
            keep_from = idx
            continue
        break
    if keep_from >= len(lines):
        return text, ""
    body = "\n".join(lines[:keep_from]).rstrip()
    tail = "\n".join(lines[keep_from:])
    return body, ("\n" + tail if body else tail)


def scrub_synthetic_stance(text: str) -> tuple[str, list[str]]:
    codes: list[str] = []
    working = text
    if _INLINE_STANCE_LENIN.search(working):
        working = _INLINE_STANCE_LENIN.sub(" ", working)
        codes.append("strip:inline_stance_lenin")
    if _INLINE_STANCE_RU_LABEL.search(working):
        working = _INLINE_STANCE_RU_LABEL.sub(" ", working)
        codes.append("strip:inline_stance_ru_label")
    if codes and _POST_STANCE_DEBRIS.search(working):
        working = _POST_STANCE_DEBRIS.sub(" ", working)
        codes.append("strip:stance_debris")
    return working, codes


def scrub_instruction_dumps(text: str) -> tuple[str, list[str]]:
    codes: list[str] = []
    working = text
    if _INSTRUCTION_DUMP.search(working):
        working = _INSTRUCTION_DUMP.sub(" ", working)
        codes.append("strip:instruction_dump")
    if _PROMPT_TASK_TAIL.search(working):
        working = _PROMPT_TASK_TAIL.sub("", working)
        codes.append("strip:prompt_task_tail")
    return working, codes


def scrub_markdown_debris(text: str) -> tuple[str, list[str]]:
    codes: list[str] = []
    working = text
    if _EMPTY_MD_SCAFFOLD.search(working):
        working = _EMPTY_MD_SCAFFOLD.sub(" ", working)
        codes.append("strip:empty_md_scaffold")
    if _INLINE_MD_DEBRIS_CLUSTER.search(working):
        working = _INLINE_MD_DEBRIS_CLUSTER.sub(" ", working)
        codes.append("strip:md_debris_cluster")
    if _TERMINAL_MD_DEBRIS.search(working):
        working = _TERMINAL_MD_DEBRIS.sub("", working)
        codes.append("strip:terminal_md_debris")
    return working, codes


def normalize_section_headers(text: str) -> tuple[str, list[str]]:
    codes: list[str] = []
    working = text
    if _STRAY_ASTERISK_LINE.search(working):
        working = _STRAY_ASTERISK_LINE.sub("", working)
        codes.append("strip:stray_asterisk_line")
    if _INLINE_BOLD_LABEL.search(working):

        def _bold_sub(match: re.Match[str]) -> str:
            raw = match.group(0)
            if raw.startswith("\n") or "\n" in raw[:3]:
                prefix = "\n"
            elif match.start() > 0:
                prefix = " "
            else:
                prefix = ""
            return f"{prefix}{match.group(1).capitalize()}: "

        working = _INLINE_BOLD_LABEL.sub(_bold_sub, working)
        codes.append("fix:inline_bold_label")
    if _LABEL_BOLD_JUNK.search(working):

        def _label_sub(match: re.Match[str]) -> str:
            return f"{match.group(2).capitalize()}: "

        working = _LABEL_BOLD_JUNK.sub(_label_sub, working)
        codes.append("fix:label_bold_junk")
    if _LABEL_SPACING.search(working):

        def _space_sub(match: re.Match[str]) -> str:
            start = match.start()
            if start > 0 and working[start - 1] == "\n":
                prefix = ""
            elif start == 0:
                prefix = ""
            else:
                # Prefer section break when label follows other content.
                prefix = "\n"
            return f"{prefix}{match.group(1).capitalize()}: "

        new_text = _LABEL_SPACING.sub(_space_sub, working)
        if new_text != working:
            working = new_text
            codes.append("fix:label_spacing")
    return working, codes


def truncate_trailing_triad_restart(text: str) -> tuple[str, list[str]]:
    """Keep first Факт/Механизм/Вывод; cut restart after first Вывод."""
    codes: list[str] = []
    matches = list(_SECTION_LABEL.finditer(text))
    first_fact = next((m for m in matches if m.group(1).casefold() == "факт"), None)
    first_vyvod = next((m for m in matches if m.group(1).casefold() == "вывод"), None)
    if first_vyvod is None:
        return text, codes

    cut_at: int | None = None
    after_vyvod = [m for m in matches if m.start() > first_vyvod.start()]
    if after_vyvod:
        cut_at = after_vyvod[0].start()
        restart = after_vyvod[0]
    else:
        inline = None
        for match in _INLINE_SECTION_RESTART.finditer(text):
            if match.start() > first_vyvod.start():
                inline = match
                break
        if inline is None:
            return text, codes
        cut_at = inline.start()
        restart = inline

    if first_fact is not None and restart.group(1).casefold() == "факт":
        next_after_fact = next(
            (m for m in matches if m.start() > first_fact.start()),
            None,
        )
        fact_end = next_after_fact.start() if next_after_fact is not None else len(text)
        canonical_fact = _normalize_fact_body(text[first_fact.end() : fact_end])
        next_after_restart = next(
            (m for m in matches if m.start() > restart.start()),
            None,
        )
        trail_end = (
            next_after_restart.start() if next_after_restart is not None else len(text)
        )
        if next_after_restart is None and cut_at is not None:
            trail_end = len(text)
        trailing = _normalize_fact_body(text[restart.end() : trail_end])
        if trailing and trailing == canonical_fact:
            codes.append("strip:trailing_exact_fact_dup")
        else:
            codes.append("strip:trailing_triad_restart")
    else:
        codes.append("strip:trailing_triad_restart")
    cut_text = text[:cut_at].rstrip()
    cut_text = _TRAILING_MD_BEFORE_CUT.sub("", cut_text).rstrip()
    return cut_text, codes


def detect_integrity_issues(text: str) -> list[str]:
    codes: list[str] = list(detect_encoding_artifacts(text))
    for pattern in _HOLE_PATTERNS:
        if pattern.search(text):
            codes.append("integrity:hole_syntax")
            break
    if _RESIDUAL_STANCE.search(text):
        codes.append("integrity:residual_stance")
    if _INSTRUCTION_DUMP.search(text):
        codes.append("integrity:residual_instruction")
    if _RESIDUAL_PROMPT_TASK.search(text):
        codes.append("integrity:prompt_task_echo")
    if _RESIDUAL_MD_DEBRIS.search(text) or _TERMINAL_MD_DEBRIS.search(text):
        codes.append("integrity:md_debris")
    if _MESTO_MARKER.search(text):
        codes.append("integrity:mesto_marker")
    return codes


def cleanup_answer_body(
    text: str,
    *,
    config: QualityPostcheckConfig | None = None,
) -> BodyCleanupResult:
    """Mutate analysis body before NewsGuard / yellow; preserve safety tails."""
    cfg = config or QualityPostcheckConfig()
    if not bool(getattr(cfg, "answer_body_cleanup_enabled", True)):
        return BodyCleanupResult(text=text)

    body, safety_tail = _protect_safety_tails(text)
    codes: list[str] = []
    working = body

    working, norm_codes = normalize_section_headers(working)
    codes.extend(norm_codes)
    working, stance_codes = scrub_synthetic_stance(working)
    codes.extend(stance_codes)
    working, instr_codes = scrub_instruction_dumps(working)
    codes.extend(instr_codes)
    working, md_codes = scrub_markdown_debris(working)
    codes.extend(md_codes)
    working, triad_codes = truncate_trailing_triad_restart(working)
    codes.extend(triad_codes)

    working = re.sub(r"[ \t]{2,}", " ", working)
    working = re.sub(r"\n{3,}", "\n\n", working).strip()
    # Re-scrub after triad ops (markers may sit on conclusion line).
    working, stance_codes2 = scrub_synthetic_stance(working)
    codes.extend(stance_codes2)
    working, instr_codes2 = scrub_instruction_dumps(working)
    codes.extend(instr_codes2)
    working, md_codes2 = scrub_markdown_debris(working)
    codes.extend(md_codes2)

    integrity_codes = detect_integrity_issues(working)
    enforce = str(getattr(cfg, "integrity_enforce_mode", "soft") or "soft").lower()
    integrity_error = bool(integrity_codes)
    hard_fail = (
        bool(getattr(cfg, "integrity_check_enabled", True))
        and enforce == "strict"
        and integrity_error
    )

    if safety_tail:
        working = (
            f"{working.rstrip()}\n{safety_tail.lstrip()}"
            if working
            else safety_tail.lstrip()
        )

    meta: dict[str, Any] = {
        "body_cleanup_codes": list(codes),
        "integrity_codes": list(integrity_codes),
        "integrity_error": integrity_error,
        "postprocess_hard_fail": hard_fail,
        "integrity_enforce_mode": enforce,
    }
    if hard_fail:
        codes.append("deny:postprocess_hard_fail")
    return BodyCleanupResult(text=working, codes=codes, metadata=meta)
