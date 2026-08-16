"""Runtime ops reporting: windowed stats and admin digests."""

from src.core.ops.report_formatter import format_ops_digest
from src.core.ops.window_stats import WindowStats

__all__ = ["WindowStats", "format_ops_digest"]
