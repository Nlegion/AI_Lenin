"""PII minimization for external censorship evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_FIO = re.compile(r"\b[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+\b")
_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b")
_DATE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")


def _scrub_text(text: str) -> str:
    value = _FIO.sub("[REDACTED_FIO]", text)
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _PHONE.sub("[REDACTED_PHONE]", value)
    value = _DATE.sub("[REDACTED_DATE]", value)
    return value


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.get("title") or "")
        body_hash = str(row.get("body_hash") or "")
        cleaned = {
            "timestamp_utc": row.get("timestamp_utc"),
            "news_id_hash": _hash(str(row.get("news_id") or "")),
            "source": row.get("source"),
            "title": _scrub_text(title),
            "body_hash": body_hash,
            "decision": row.get("decision"),
            "category": row.get("category"),
            "risk_tier": row.get("risk_tier"),
            "reason_codes": row.get("reason_codes") or [],
            "latency_ms": row.get("latency_ms"),
            "config_version_hash": row.get("config_version_hash"),
            "sample_seed": row.get("sample_seed"),
        }
        out_rows.append(cleaned)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"scrubbed={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
