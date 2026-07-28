"""Shared constants for post-generation safety gates."""

from __future__ import annotations

# Append-only warn audit for cliché / anachronism gates (not NewsGuard eval reports).
GATE_WARN_AUDIT_PATH = ".cursor/artifacts/safety/gate_warn_audit.jsonl"

CLICHE_CODE_NO_R1 = "cliche_no_r1"
CLICHE_CODE_LOW_R1_OVERLAP = "cliche_low_r1_overlap"
CLICHE_CODE_LEXICON_DENSE = "cliche_lexicon_dense"
CLICHE_CODE_SKIPPED_NO_BRIEF = "cliche_skipped_no_brief"

ANACHRONISM_CODE_FIRST_PERSON_TECH = "anachronism_first_person_tech"
