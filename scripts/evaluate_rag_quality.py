#!/usr/bin/env python3
"""Evaluate RAG quality metrics for Subplan J."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.evaluation.rag_quality_metrics import build_quality_report  # noqa: E402
from src.core.retrieval.qdrant_retrieval_provider import QdrantRetrievalProvider, RetrievalProviderConfig  # noqa: E402
from src.core.settings.release_gates import load_release_gates, metric_passes  # noqa: E402


def _read_eval(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _load_retrieval_provider(config_path: Path) -> QdrantRetrievalProvider:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("retrieval_pipeline", payload)
    config = RetrievalProviderConfig(
        collection_name=section["collection_name"],
        qdrant_path=(REPO_ROOT / section["qdrant_path"]).resolve(),
        dense_model=section["dense_model"],
        sparse_state_path=(REPO_ROOT / section["sparse_state_path"]).resolve(),
        ontology_tags_path=(REPO_ROOT / section["ontology_tags_path"]).resolve(),
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        device=section.get("device", "cpu"),
        top_k=int(section.get("top_k", 20)),
        rrf_k=int(section.get("rrf_k", 60)),
        retriever_weights=dict(section.get("retriever_weights", {})),
        source_boosts=dict(section.get("source_boosts", {})),
        max_context_chunks=int(section.get("max_context_chunks", 7)),
        hyde_enabled=bool(section.get("hyde_enabled", True)),
        query_rewrite_enabled=bool(section.get("query_rewrite_enabled", True)),
        query_decomposition_enabled=bool(section.get("query_decomposition_enabled", True)),
    )
    return QdrantRetrievalProvider(config=config)


def _extract_query(raw_query: str) -> str:
    marker = "Query:"
    if marker in raw_query:
        return raw_query.split(marker, maxsplit=1)[1].strip()
    return raw_query.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG quality metrics.")
    parser.add_argument("--retrieval-config", default="config/retrieval_pipeline.yaml")
    parser.add_argument(
        "--thresholds-config",
        default="config/release_gates.yaml",
        help="Unified release_gates.yaml (or legacy quality_thresholds.yaml via shim).",
    )
    parser.add_argument("--eval-dataset", default=".cursor/artifacts/eval/embedding_eval.tsv")
    parser.add_argument("--output-json", default=".cursor/artifacts/evaluation/rag_quality_metrics.json")
    parser.add_argument("--output-md", default=".cursor/artifacts/evaluation/rag_quality_summary.md")
    parser.add_argument("--max-queries", type=int, default=40)
    args = parser.parse_args()

    provider = _load_retrieval_provider(config_path=(REPO_ROOT / args.retrieval_config).resolve())
    eval_rows = _read_eval(path=(REPO_ROOT / args.eval_dataset).resolve())[: args.max_queries]
    gates = load_release_gates(path=str((REPO_ROOT / args.thresholds_config).resolve()))
    rag = gates.rag_quality
    tolerance = rag.tolerance_relative

    predictions: list[list[str]] = []
    positives: list[str] = []
    contexts: list[str] = []
    stances: list[list[str]] = []
    analyses: list[str] = []
    latencies_ms: list[float] = []

    for row in eval_rows:
        query_text = _extract_query(raw_query=row["query"])
        started = time.perf_counter()
        candidates = provider.retrieve(query_text=query_text)
        latencies_ms.append((time.perf_counter() - started) * 1000)

        predictions.append([candidate.source_id for candidate in candidates])
        positives.append(row["positive_source_id"])
        rendered_context = provider.render_context(candidates=candidates)
        contexts.append(rendered_context)
        stances.append([candidate.stance_type for candidate in candidates])

        if candidates:
            quote = candidates[0].text[:120].strip().replace("\n", " ")
            analysis = f'Как я писал: "{quote}"'
        else:
            analysis = "Данная тема не входит в круг моих исследований."
        analyses.append(analysis)

    report = build_quality_report(
        predictions=predictions,
        positives=positives,
        contexts=contexts,
        candidate_stances=stances,
        analyses=analyses,
        latencies_ms=latencies_ms,
    )

    metric_values = {
        "recall_at_5": report.recall_at_5,
        "core_self_ratio": report.core_self_ratio,
        "ideology_consistency": report.ideology_consistency,
        "citation_hallucination_rate_max": report.citation_hallucination_rate,
        "empty_context_rate_max": report.empty_context_rate,
    }
    checks: dict[str, bool] = {}
    thresholds_flat: dict[str, float] = {}
    for name, value in metric_values.items():
        metric = rag.metrics.get(name)
        if metric is None:
            continue
        thresholds_flat[name] = metric.threshold
        checks[f"{name}_passed"] = metric_passes(
            value=value,
            threshold=metric.threshold,
            direction=metric.direction,
            tolerance_relative=tolerance,
        )

    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "queries_evaluated": len(eval_rows),
        "metrics": asdict(report),
        "thresholds": thresholds_flat,
        "tolerance_relative": tolerance,
        "config_version": gates.version,
        "checks": checks,
    }

    output_json = (REPO_ROOT / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = (REPO_ROOT / args.output_md).resolve()
    lines = [
        "# RAG Quality Summary",
        "",
        f"- Generated at (UTC): {payload['generated_at_utc']}",
        f"- Queries evaluated: `{payload['queries_evaluated']}`",
        f"- Recall@5: `{report.recall_at_5:.4f}`",
        f"- Core self ratio: `{report.core_self_ratio:.4f}`",
        f"- Empty context rate: `{report.empty_context_rate:.4f}`",
        f"- Ideology consistency: `{report.ideology_consistency:.4f}`",
        f"- Citation hallucination rate: `{report.citation_hallucination_rate:.4f}`",
    ]
    for name, value in checks.items():
        lines.append(f"- `{name}`: `{'yes' if value else 'no'}`")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluated queries: {len(eval_rows)}")
    print(f"Recall@5: {report.recall_at_5:.4f}")
    if checks and not all(checks.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
