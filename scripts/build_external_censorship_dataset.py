"""Build external dataset manifest with retry and last-known-good fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml


def _load_sources(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("external_datasets", payload)


def _fetch_url_status(url: str, retries: int = 3) -> dict[str, Any]:
    delay = 1.0
    last_error = ""
    for _ in range(retries):
        try:
            req = Request(url=url, method="GET", headers={"User-Agent": "ai-lenin-dataset-builder/1.0"})
            with urlopen(req, timeout=20) as response:
                return {"ok": True, "status": int(response.status), "error": ""}
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            time.sleep(delay)
            delay *= 2
    return {"ok": False, "status": 0, "error": last_error}


def _validate_row(row: dict[str, Any]) -> None:
    required = {
        "id",
        "source_url",
        "license",
        "allowed_use",
        "dataset_version",
        "retrieval_date",
        "checksum",
        "mapped_labels",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"Manifest row missing fields: {sorted(missing)}")


def _checksum_payload(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_manifest(*, sources_cfg: Path, output_manifest: Path, cache_manifest: Path) -> dict[str, Any]:
    cfg = _load_sources(sources_cfg)
    rows: list[dict[str, Any]] = []
    for source in cfg.get("sources", []):
        status = _fetch_url_status(str(source.get("source_url", "")))
        row = {
            "id": str(source.get("id", "")),
            "source_url": str(source.get("source_url", "")),
            "license": str(source.get("license", "unknown")),
            "allowed_use": str(source.get("allowed_use", "pending_review")),
            "dataset_version": "latest",
            "retrieval_date": datetime.now(timezone.utc).isoformat(),
            "mapped_labels": list(source.get("mapped_labels", [])),
            "download_ok": bool(status["ok"]),
            "download_status": int(status["status"]),
            "download_error": str(status["error"]),
        }
        row["checksum"] = _checksum_payload(row)
        _validate_row(row)
        rows.append(row)

    if rows and all(row.get("download_ok") for row in rows):
        result = {"generated_at": datetime.now(timezone.utc).isoformat(), "sources": rows}
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        cache_manifest.parent.mkdir(parents=True, exist_ok=True)
        cache_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    if cache_manifest.is_file():
        return json.loads(cache_manifest.read_text(encoding="utf-8"))
    raise RuntimeError("External sources unavailable and no last-known-good manifest found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-config", default="config/external_dataset_sources.yaml")
    parser.add_argument("--output-manifest", default=".cursor/artifacts/quality/dataset_manifest.json")
    parser.add_argument("--cache-manifest", default=".cursor/artifacts/quality/dataset_manifest.last_good.json")
    args = parser.parse_args()

    manifest = build_manifest(
        sources_cfg=Path(args.sources_config),
        output_manifest=Path(args.output_manifest),
        cache_manifest=Path(args.cache_manifest),
    )
    print(f"sources={len(manifest.get('sources', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

