#!/usr/bin/env python3
"""Evaluate anti-cliché gate against gold JSONL cases."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.safety.cliche_gate import cliche_gate  # noqa: E402


def _load_cases(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_warn_positive(codes: list[str]) -> bool:
    return any(code != "cliche_skipped_no_brief" for code in codes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate anti-cliché gate on gold cases."
    )
    parser.add_argument("--cases", default="data/eval/anti_cliche_cases.jsonl")
    parser.add_argument(
        "--out-json", default=".cursor/artifacts/evaluation/anti_cliche_eval.json"
    )
    parser.add_argument(
        "--out-md", default=".cursor/artifacts/evaluation/anti_cliche_eval.md"
    )
    args = parser.parse_args()

    cases = _load_cases(path=(REPO_ROOT / args.cases).resolve())
    if not cases:
        print("no_cases", file=sys.stderr)
        return 1

    true_pos = false_pos = false_neg = true_neg = 0
    warn_count = 0
    details: list[dict] = []

    for case in cases:
        result = cliche_gate(
            analysis=case["analysis"],
            brief_present=bool(case.get("brief_present", True)),
            r1_text=case.get("r1_text", ""),
            r1_count=int(case.get("r1_count", 0)),
        )
        expected = list(case.get("expect_codes", []))
        expect_positive = _is_warn_positive(expected)
        predicted_positive = _is_warn_positive(result.reason_codes)
        if expect_positive and predicted_positive:
            true_pos += 1
        elif not expect_positive and predicted_positive:
            false_pos += 1
        elif expect_positive and not predicted_positive:
            false_neg += 1
        else:
            true_neg += 1
        if predicted_positive:
            warn_count += 1
        details.append(
            {
                "id": case.get("id"),
                "expected": expected,
                "actual": result.reason_codes,
                "skipped": result.skipped,
            }
        )

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 1.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 1.0
    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": len(cases),
        "warn_rate": warn_count / len(cases),
        "precision": precision,
        "recall": recall,
        "true_pos": true_pos,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "true_neg": true_neg,
        "details": details,
    }
    out_json = (REPO_ROOT / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    out_md = (REPO_ROOT / args.out_md).resolve()
    out_md.write_text(
        "\n".join(
            [
                "# Anti-cliché eval",
                "",
                f"- cases: `{payload['cases']}`",
                f"- precision: `{precision:.3f}`",
                f"- recall: `{recall:.3f}`",
                f"- warn_rate: `{payload['warn_rate']:.3f}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"precision={precision:.3f} recall={recall:.3f} warn_rate={payload['warn_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
