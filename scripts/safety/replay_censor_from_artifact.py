"""Replay pre-RAG censor decisions from an artifact JSONL file.

Input rows are expected to contain at least:
- news_id
- title
- source
- decision (old decision)

If body/content is absent, script can fallback to title as body.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any

from _external_dataset_prestep import ensure_external_dataset_prestep
from src.core.safety.news_guard import NewsGuard
from src.core.safety.pre_rag_censor import PreRagCensor
from src.core.safety.pre_rag_censor_types import CensorInput
from src.core.safety.safety_gate import SafetyGate
from src.core.settings.censorship_runtime_config import (
    default_censorship_runtime_config_path,
    load_censorship_runtime_config,
)
from src.core.safety.censor_hashing import compute_content_hash

_WS_RE = re.compile(r"\s+")


def _pre_censor_dedup_key(*, title: str, body: str) -> str:
    merged = f"{title}\n{body}".strip().lower()
    return _WS_RE.sub(" ", merged)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _replay_rows(
    *,
    rows: list[dict[str, Any]],
    censor: PreRagCensor,
    use_title_as_body: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    seen_pre_censor: dict[str, int] = {}
    dropped_titles: collections.Counter[str] = collections.Counter()
    for idx, row in enumerate(rows, start=1):
        title = str(row.get("title") or "")
        body = str(row.get("body") or row.get("content") or "")
        if not body and use_title_as_body:
            body = title
        dedup_key = _pre_censor_dedup_key(title=title, body=body)
        duplicate_of = seen_pre_censor.get(dedup_key)
        if duplicate_of is not None:
            dropped_titles[title] += 1
            continue
        seen_pre_censor[dedup_key] = idx
        content_hash, _ = compute_content_hash(
            title=title, body=body, url=str(row.get("url") or "")
        )
        result = await censor.evaluate(
            CensorInput(
                news_id=str(row.get("news_id") or f"line-{idx}"),
                title=title,
                body=body,
                source=str(row.get("source") or "unknown"),
                metadata={"url": str(row.get("url") or "")},
            )
        )
        old_decision = str(row.get("decision") or "")
        old_codes = list(row.get("reason_codes") or [])
        changed = old_decision != result.decision or old_codes != list(
            result.reason_codes
        )
        out.append(
            {
                "line_no": idx,
                "news_id": str(row.get("news_id") or ""),
                "source": str(row.get("source") or ""),
                "title": title,
                "content_hash": content_hash,
                "body_missing_in_input": not bool(
                    row.get("body") or row.get("content")
                ),
                "old_decision": old_decision,
                "old_reason_codes": old_codes,
                "new_decision": result.decision,
                "new_category": result.category,
                "new_risk_tier": result.risk_tier,
                "new_reason_codes": list(result.reason_codes),
                "new_config_version_hash": result.audit.get("config_version_hash", ""),
                "changed": changed,
            }
        )
    return out, dict(dropped_titles)


def _write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "line_no",
        "news_id",
        "source",
        "title",
        "old_decision",
        "new_decision",
        "new_category",
        "new_risk_tier",
        "changed",
        "new_reason_codes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "line_no": row["line_no"],
                    "news_id": row["news_id"],
                    "source": row["source"],
                    "title": row["title"],
                    "old_decision": row["old_decision"],
                    "new_decision": row["new_decision"],
                    "new_category": row["new_category"] or "",
                    "new_risk_tier": row["new_risk_tier"],
                    "changed": row["changed"],
                    "new_reason_codes": ",".join(row["new_reason_codes"]),
                }
            )


def _write_dropped_duplicates_csv(path: Path, dropped_titles: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "dropped_count"])
        writer.writeheader()
        for title, count in sorted(
            dropped_titles.items(), key=lambda item: item[1], reverse=True
        ):
            writer.writerow({"title": title, "dropped_count": count})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL artifact path")
    parser.add_argument(
        "--output-jsonl", required=True, help="Output JSONL with replay results"
    )
    parser.add_argument(
        "--output-csv", required=True, help="Output CSV for manual review"
    )
    parser.add_argument(
        "--output-dropped-csv",
        default="",
        help="Optional CSV with titles dropped by pre-censor dedup",
    )
    parser.add_argument(
        "--use-title-as-body",
        action="store_true",
        default=False,
        help="Fallback to body=title when input has no body/content",
    )
    parser.add_argument(
        "--external-max-rows-per-source",
        type=int,
        default=50000,
        help="Mandatory pre-step: external dataset rows to materialize per source.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _read_jsonl(input_path)
    base_dir = Path(__file__).resolve().parents[2]
    ensure_external_dataset_prestep(
        repo_root=base_dir,
        max_rows_per_source=int(args.external_max_rows_per_source),
    )
    gate = SafetyGate.from_base_dir(base_dir)
    guard = NewsGuard.from_file(base_dir / "config" / "news_guard.yaml")
    cfg_path = default_censorship_runtime_config_path(base_dir)
    runtime_cfg = load_censorship_runtime_config(cfg_path)
    censor = PreRagCensor(
        safety_gate=gate,
        news_guard=guard,
        config=runtime_cfg,
        config_path=str(cfg_path),
    )
    replayed, dropped_titles = asyncio.run(
        _replay_rows(
            rows=rows,
            censor=censor,
            use_title_as_body=bool(args.use_title_as_body),
        )
    )
    _write_jsonl(Path(args.output_jsonl), replayed)
    _write_review_csv(Path(args.output_csv), replayed)
    if args.output_dropped_csv:
        _write_dropped_duplicates_csv(Path(args.output_dropped_csv), dropped_titles)
    changed = sum(1 for row in replayed if row["changed"])
    dropped = sum(dropped_titles.values())
    print(
        f"rows={len(replayed)} dropped_duplicates={dropped} changed={changed} unchanged={len(replayed) - changed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
