"""Unified answer postprocess: contract, two-phase engine, terminal scrub."""

from src.core.generation.postprocess_clean.contract import (
    PostProcessInput,
    PostProcessResult,
    has_required_triad,
    map_postprocess_status,
    resolve_clean_mode,
)
from src.core.generation.postprocess_clean.engine import (
    apply_terminal_public_scrub,
    run_postprocess,
    scrub_after_output_guard,
)

__all__ = [
    "PostProcessInput",
    "PostProcessResult",
    "apply_terminal_public_scrub",
    "has_required_triad",
    "map_postprocess_status",
    "resolve_clean_mode",
    "run_postprocess",
    "scrub_after_output_guard",
]
