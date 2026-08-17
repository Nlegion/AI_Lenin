"""Format WindowSnapshot into a Telegram admin digest (plain text)."""

from __future__ import annotations

from collections import Counter

from src.core.ops.window_stats import WindowSnapshot


def _fmt_uptime(seconds: float) -> str:
    total = int(max(0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}ч {minutes}м"
    if minutes > 0:
        return f"{minutes}м"
    return f"{secs}с"


def _fmt_latency(ms: int | None) -> str:
    if ms is None:
        return "—"
    if ms >= 1000:
        return f"{ms / 1000.0:.1f}с"
    return f"{ms}мс"


def _top_reasons(counter: Counter[str], *, limit: int) -> str:
    if not counter or limit <= 0:
        return ""
    parts = [f"{code}:{count}" for code, count in counter.most_common(limit)]
    return ", ".join(parts)


def format_ops_digest(
    snapshot: WindowSnapshot,
    *,
    interval_seconds: int = 1800,
    workable_backlog: int = 0,
    stale_backlog: int = 0,
    unpublished: int = 0,
    top_reasons: int = 3,
    idle_digest: str = "short",
) -> str:
    """Build the Russian admin digest for one stats window."""
    minutes = max(1, int(round(interval_seconds / 60.0)))
    provider = (snapshot.llm_provider or "unknown").strip() or "unknown"
    header = (
        f"📊 {minutes} мин | uptime {_fmt_uptime(snapshot.uptime_seconds())} "
        f"| LLM {provider}"
    )
    c = snapshot.counters
    rss = c.get("rss_seen", 0)
    dedup = c.get("dedup_dropped", 0)
    inserted = c.get("inserted", 0)
    processed = c.get("news_processed", 0)
    published = c.get("analyses_published", 0)
    skipped = c.get("news_skipped", 0)
    review = c.get("review_continued", 0)
    rejected = c.get("analyses_rejected", 0)
    pub_fail = c.get("publish_failed", 0)
    errors = c.get("errors", 0)
    guard_blocked = c.get("output_guard_blocked", 0)

    stall = snapshot.is_idle() and (
        workable_backlog > 0 or unpublished > 0 or stale_backlog > 0
    )
    if snapshot.is_idle() and idle_digest == "short" and not stall:
        return (
            f"{header}\n"
            f"Тихо: RSS {rss}, новых 0, очередь "
            f"{workable_backlog}/{unpublished}"
        )

    skip_tail = _top_reasons(snapshot.skip_reasons, limit=top_reasons)
    skip_line = f"Пропущено {skipped}"
    if skip_tail:
        skip_line += f" ({skip_tail})"
    if review:
        skip_line += f" | review {review}"
    else:
        skip_line += " | review 0"
    if guard_blocked:
        skip_line += f" | guard {guard_blocked}"
    skip_line += f" | отклонено {rejected} | pub fail {pub_fail} | ошибок {errors}"

    reject_tail = _top_reasons(snapshot.reject_reasons, limit=top_reasons)
    if reject_tail and rejected:
        skip_line += f" [{reject_tail}]"

    queue_line = (
        f"Очередь: рабочих {workable_backlog} | "
        f"застрявших (date>24ч) {stale_backlog} | "
        f"неопубликованных {unpublished}"
    )
    if stall:
        queue_line += " | ⚠ простой при ненулевой очереди"

    lines = [
        header,
        (
            f"Воронка: RSS {rss} / дедуп {dedup} / новых {inserted} / "
            f"обработано {processed} / опубликовано {published}"
        ),
        skip_line,
        queue_line,
    ]

    p50 = snapshot.percentile(50)
    p95 = snapshot.percentile(95)
    if p50 is not None:
        lines.append(
            "LLM: "
            f"p50 {_fmt_latency(p50)} p95 {_fmt_latency(p95)} | "
            f"timeouts {c.get('generation_timeouts', 0)} | "
            f"circuit {c.get('circuit_opens', 0)} | "
            f"held {c.get('degraded_held', 0)}"
        )
    elif (
        c.get("generation_timeouts", 0)
        or c.get("circuit_opens", 0)
        or c.get("degraded_held", 0)
    ):
        lines.append(
            "LLM: timeouts "
            f"{c.get('generation_timeouts', 0)} | "
            f"circuit {c.get('circuit_opens', 0)} | "
            f"held {c.get('degraded_held', 0)}"
        )

    return "\n".join(lines)
