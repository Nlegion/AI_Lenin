#!/usr/bin/env python3
"""Benchmark embedding model candidates on retrieval eval set."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import time
import sys

import psutil
import torch
import yaml
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.embeddings.benchmark import BenchmarkResult, choose_best_model, compute_recall_at_k  # noqa: E402


def _read_eval_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _load_candidates(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("embedding_benchmark", payload)


def _to_list(matrix) -> list[list[float]]:
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    return [list(item) for item in matrix]


def _build_documents(corpus_root: Path, registry_rows: list[dict[str, str]]) -> tuple[list[str], dict[str, int]]:
    documents: list[str] = []
    doc_index: dict[str, int] = {}
    for row in registry_rows:
        source_id = row["source_id"]
        source_path = row["source_path"]
        text = (corpus_root / source_path).read_text(encoding="utf-8", errors="replace")
        doc_index[source_id] = len(documents)
        documents.append(text)
    return documents, doc_index


def _benchmark_model(
    model_name: str,
    query_texts: list[str],
    document_texts: list[str],
    positive_doc_ids: list[int],
    trust_remote_code: bool,
    device: str | None,
) -> BenchmarkResult:
    process = psutil.Process()
    rss_before = process.memory_info().rss / (1024 * 1024)
    vram_peak_mb: float | None = None

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = SentenceTransformer(model_name_or_path=model_name, trust_remote_code=trust_remote_code, device=device)
    query_vectors = _to_list(model.encode(query_texts, normalize_embeddings=True))
    document_vectors = _to_list(model.encode(document_texts, normalize_embeddings=True))

    recall_at_5 = compute_recall_at_k(
        query_embeddings=query_vectors,
        document_embeddings=document_vectors,
        positives=positive_doc_ids,
        k=5,
    )
    elapsed = time.perf_counter() - started
    mean_latency_ms = (elapsed / max(len(query_texts), 1)) * 1000

    if torch.cuda.is_available():
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    rss_after = process.memory_info().rss / (1024 * 1024)
    return BenchmarkResult(
        model_name=model_name,
        recall_at_5=recall_at_5,
        mean_latency_ms=mean_latency_ms,
        ram_delta_mb=max(0.0, rss_after - rss_before),
        vram_peak_mb=vram_peak_mb,
        status="ok",
    )


def _write_results(results: list[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.__dict__ for result in results]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_decision(
    output_path: Path,
    winner: BenchmarkResult | None,
    should_fine_tune: bool,
    min_recall_at_5: float,
    total_models: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Embedding Selection Decision",
        "",
        f"- Generated at (UTC): {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Models evaluated: `{total_models}`",
        f"- Target threshold Recall@5: `{min_recall_at_5:.2f}`",
    ]
    if winner is None:
        lines.extend(
            [
                "- Winner: `N/A`",
                "- Fine-tuning decision: `required` (no successful model run)",
            ]
        )
    else:
        lines.extend(
            [
                f"- Winner: `{winner.model_name}`",
                f"- Winner Recall@5: `{winner.recall_at_5:.3f}`",
                f"- Winner latency (ms/query): `{winner.mean_latency_ms:.2f}`",
                f"- Fine-tuning decision: `{'required' if should_fine_tune else 'not required'}`",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark embedding candidates.")
    parser.add_argument("--eval-dataset", default=".cursor/artifacts/eval/embedding_eval.tsv")
    parser.add_argument("--source-registry", default=".cursor/artifacts/registries/source_registry.tsv")
    parser.add_argument("--corpus-root", default="data/books")
    parser.add_argument("--config", default="config/embedding_benchmark.yaml")
    parser.add_argument("--results-output", default=".cursor/artifacts/embeddings/benchmark_results.json")
    parser.add_argument("--decision-output", default=".cursor/artifacts/embeddings/embedding_selection.md")
    parser.add_argument("--max-models", type=int, default=0, help="Limit models for quick local run.")
    args = parser.parse_args()

    eval_path = (REPO_ROOT / args.eval_dataset).resolve()
    registry_path = (REPO_ROOT / args.source_registry).resolve()
    corpus_root = (REPO_ROOT / args.corpus_root).resolve()
    config_path = (REPO_ROOT / args.config).resolve()
    results_output = (REPO_ROOT / args.results_output).resolve()
    decision_output = (REPO_ROOT / args.decision_output).resolve()

    config = _load_candidates(path=config_path)
    model_entries = [entry for entry in config.get("models", []) if bool(entry.get("enabled", True))]
    if args.max_models > 0:
        model_entries = model_entries[: args.max_models]

    eval_rows = _read_eval_rows(path=eval_path)
    registry_rows = _read_registry(path=registry_path)
    documents, index_by_source = _build_documents(corpus_root=corpus_root, registry_rows=registry_rows)
    query_texts = [row["query"] for row in eval_rows]
    positives = [index_by_source[row["positive_source_id"]] for row in eval_rows if row["positive_source_id"] in index_by_source]
    query_texts = query_texts[: len(positives)]

    results: list[BenchmarkResult] = []
    for entry in model_entries:
        model_name = entry["name"]
        try:
            result = _benchmark_model(
                model_name=model_name,
                query_texts=query_texts,
                document_texts=documents,
                positive_doc_ids=positives,
                trust_remote_code=bool(entry.get("trust_remote_code", False)),
                device=entry.get("device"),
            )
        except Exception as exc:  # noqa: BLE001
            result = BenchmarkResult(
                model_name=model_name,
                recall_at_5=0.0,
                mean_latency_ms=0.0,
                ram_delta_mb=0.0,
                vram_peak_mb=None,
                status="failed",
                notes=str(exc)[:300],
            )
        results.append(result)

    _write_results(results=results, output_path=results_output)
    winner, should_fine_tune = choose_best_model(
        results=results,
        min_recall_at_5=float(config.get("min_recall_at_5", 0.85)),
    )
    _write_decision(
        output_path=decision_output,
        winner=winner,
        should_fine_tune=should_fine_tune,
        min_recall_at_5=float(config.get("min_recall_at_5", 0.85)),
        total_models=len(results),
    )
    print(f"Benchmark results written: {results_output}")
    print(f"Selection decision written: {decision_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
