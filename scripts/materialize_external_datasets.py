"""Materialize configured external datasets into unified JSONL artifacts."""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen
import zipfile

import yaml
from datasets import load_dataset

LENTA_BZ2_URL = (
    "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2"
)
RUS_NEWS_DATASET_ID = "data-silence/rus_news_classifier"
RU_ETHNO_REPO_URL = "https://github.com/hse-scila/ethnohate-project"
RUS_NEWS_LABELS = {
    0: "climate",
    1: "conflicts",
    2: "culture",
    3: "economy",
    4: "gloss",
    5: "health",
    6: "politics",
    7: "science",
    8: "society",
    9: "sports",
    10: "travel",
}


def _normalize_text(value: str) -> str:
    # Strict one-line JSONL records with deterministic whitespace.
    sanitized = (
        value.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )
    return " ".join(sanitized.split())


def _stable_split(*, text: str, holdout_ratio: float) -> str:
    holdout_threshold = int(max(min(holdout_ratio, 0.9), 0.01) * 100)
    mod = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 100
    return "holdout" if mod < holdout_threshold else "train"


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url=url, headers={"User-Agent": "ai-lenin-materializer/1.0"})
    with urlopen(req, timeout=120) as response, dst.open("wb") as out:
        shutil.copyfileobj(response, out)


def _materialize_lenta(
    *,
    out_path: Path,
    raw_dir: Path,
    holdout_ratio: float,
    max_rows: int,
    download_url: str,
) -> int:
    raw_file = raw_dir / "lenta-ru-news.csv.bz2"
    if not raw_file.is_file():
        _download(download_url, raw_file)

    def _iter_rows():
        with bz2.open(raw_file, "rt", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader, start=1):
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                topic = str(row.get("topic") or "unknown").strip()
                if not text and not title:
                    continue
                merged = _normalize_text((title + "\n" + text).strip())
                yield {
                    "text": merged,
                    "source": "lenta_kaggle",
                    "category": topic or "unknown",
                    "label": topic or "unknown",
                    "split": _stable_split(text=merged, holdout_ratio=holdout_ratio),
                }
                if max_rows > 0 and idx >= max_rows:
                    break

    return _write_rows(out_path, _iter_rows())


def _materialize_rus_news_classifier(
    *,
    out_path: Path,
    holdout_ratio: float,
    max_rows: int,
    dataset_id: str,
) -> int:
    parts = []
    for split in ("train", "test"):
        parts.append(load_dataset(dataset_id, split=split))

    def _iter_rows():
        written = 0
        for dataset in parts:
            for row in dataset:
                news = str(row.get("news") or "").strip()
                if not news:
                    continue
                news = _normalize_text(news)
                label_id = int(row.get("labels", -1))
                label = RUS_NEWS_LABELS.get(label_id, "unknown")
                yield {
                    "text": news,
                    "source": "rus_news_classifier",
                    "category": label,
                    "label": label,
                    "split": _stable_split(text=news, holdout_ratio=holdout_ratio),
                }
                written += 1
                if max_rows > 0 and written >= max_rows:
                    return

    return _write_rows(out_path, _iter_rows())


def _ensure_ru_ethno_repo(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.is_dir():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
        check=True,
    )


def _materialize_ru_ethno_hate(
    *,
    out_path: Path,
    repo_dir: Path,
    repo_url: str,
    holdout_ratio: float,
    max_rows: int,
) -> int:
    _ensure_ru_ethno_repo(repo_dir=repo_dir, repo_url=repo_url)
    zip_path = repo_dir / "dataset" / "RuEthnoHateExtended.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"RuEthnoHateExtended.zip not found in {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        payload = json.loads(zf.read("RuEthnoHateExtended/RuEthnoHateExtended.json").decode("utf-8"))

    def _iter_rows():
        count = 0
        for row in payload:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            if str(row.get("does_text_make_sense") or "").lower() == "no":
                continue
            text = _normalize_text(text)
            label = str(row.get("class") or "unknown").strip() or "unknown"
            category = str(row.get("ethnic_group") or "ethnic").strip() or "ethnic"
            yield {
                "text": text,
                "source": "ru_ethno_hate",
                "category": category,
                "label": label,
                "split": _stable_split(text=text, holdout_ratio=holdout_ratio),
            }
            count += 1
            if max_rows > 0 and count >= max_rows:
                break

    return _write_rows(out_path, _iter_rows())


def _merge_jsonl(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.write(line + "\n")
                    total += 1
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-config", default="config/external_dataset_sources.yaml")
    parser.add_argument("--out-dir", default="data/external_datasets")
    parser.add_argument("--tmp-dir", default=".cursor/tmp")
    parser.add_argument("--max-rows-per-source", type=int, default=250000)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.sources_config).read_text(encoding="utf-8")) or {}
    root = cfg.get("external_datasets", cfg)
    sources = {str(item.get("id", "")): item for item in root.get("sources", [])}
    holdout_ratio = float(root.get("holdout_ratio", 0.1))
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}
    outputs: list[Path] = []
    lenta_cfg = sources.get("lenta_kaggle", {})
    rus_cfg = sources.get("rus_news_classifier", {})
    ethno_cfg = sources.get("ru_ethno_hate", {})

    lenta_out = out_dir / "lenta_kaggle.jsonl"
    stats["lenta_kaggle"] = _materialize_lenta(
        out_path=lenta_out,
        raw_dir=raw_dir,
        holdout_ratio=holdout_ratio,
        max_rows=args.max_rows_per_source,
        download_url=str(lenta_cfg.get("download_url") or LENTA_BZ2_URL),
    )
    outputs.append(lenta_out)

    rus_out = out_dir / "rus_news_classifier.jsonl"
    stats["rus_news_classifier"] = _materialize_rus_news_classifier(
        out_path=rus_out,
        holdout_ratio=holdout_ratio,
        max_rows=args.max_rows_per_source,
        dataset_id=str(rus_cfg.get("hf_dataset_id") or RUS_NEWS_DATASET_ID),
    )
    outputs.append(rus_out)

    ethno_out = out_dir / "ru_ethno_hate.jsonl"
    stats["ru_ethno_hate"] = _materialize_ru_ethno_hate(
        out_path=ethno_out,
        repo_dir=tmp_dir / "ethnohate-repo",
        repo_url=str(ethno_cfg.get("git_repo_url") or RU_ETHNO_REPO_URL),
        holdout_ratio=holdout_ratio,
        max_rows=args.max_rows_per_source,
    )
    outputs.append(ethno_out)

    merged_out = out_dir / "external_unified.jsonl"
    stats["external_unified"] = _merge_jsonl(outputs, merged_out)

    report = {"stats": stats, "outputs": [str(path) for path in outputs] + [str(merged_out)]}
    report_path = out_dir / "materialization_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

