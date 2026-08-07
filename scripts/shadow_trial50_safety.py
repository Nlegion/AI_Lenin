"""Shadow-count would-block for H0.1/H0.3 on Trial50 dump (no enforce side effects)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.safety.drone_combat_guard import combat_adjacent_hit, drone_air_raid_hit
from src.core.safety.news_guard import NewsGuard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cursor/artifacts/quality/live_news_qa_trial50_20260805-2119.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    guard = NewsGuard.from_file(root / "config" / "news_guard.yaml")
    if not args.input.is_file():
        print(f"missing input: {args.input}")
        return 1
    drone_n = 0
    adjacent_n = 0
    deny_n = 0
    rows = 0
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        title = row.get("title") or ""
        content = row.get("content") or row.get("news_content") or ""
        blob = f"{title}\n{content}"
        if drone_air_raid_hit(blob).hit:
            drone_n += 1
        if combat_adjacent_hit(blob).hit:
            adjacent_n += 1
        result = guard.evaluate_input(title=title, content=content, source=row.get("source") or "TASS")
        if result.decision == "deny":
            deny_n += 1
        rows += 1
        if args.limit and rows >= args.limit:
            break
    print(
        json.dumps(
            {
                "rows": rows,
                "drone_air_raid_hit": drone_n,
                "combat_adjacent_hit": adjacent_n,
                "news_guard_deny": deny_n,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
