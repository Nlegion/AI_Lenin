#!/usr/bin/env python3
"""Run retrieval sandbox experiments for dense/sparse/onto/HyDE variants."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
import time

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.vector.bm25_sparse import Bm25SparseEncoder  # noqa: E402
from src.core.vector.sandbox_metrics import apply_source_boost, recall_at_k, reciprocal_rank_fusion  # noqa: E402


@dataclass(frozen=True)
class SandboxConfig:
    collection_name: str
    qdrant_path: Path
    dense_model: str
    trust_remote_code: bool
    device: str
    eval_dataset_path: Path
    ontology_tags_path: Path
    sparse_state_path: Path
    top_k: int
    rrf_k: int
    retriever_weights: dict[str, float]
    source_boosts: dict[str, float]
    modes: list[str]


def _load_config(path: Path) -> SandboxConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("retrieval_sandbox", payload)
    return SandboxConfig(
        collection_name=section["collection_name"],
        qdrant_path=(REPO_ROOT / section["qdrant_path"]).resolve(),
        dense_model=section["dense_model"],
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        device=section.get("device", "cpu"),
        eval_dataset_path=(REPO_ROOT / section["eval_dataset_path"]).resolve(),
        ontology_tags_path=(REPO_ROOT / section["ontology_tags_path"]).resolve(),
        sparse_state_path=(REPO_ROOT / section["sparse_state_path"]).resolve(),
        top_k=int(section.get("top_k", 20)),
        rrf_k=int(section.get("rrf_k", 60)),
        retriever_weights=dict(section.get("retriever_weights", {})),
        source_boosts=dict(section.get("source_boosts", {})),
        modes=list(section.get("modes", [])),
    )


def _read_eval(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _read_ontology(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle, delimiter="\t"))


def _extract_query_text(raw_query: str) -> str:
    match = re.search(r"Query:\s*(.*)$", raw_query, flags=re.IGNORECASE)
    return match.group(1).strip() if match else raw_query


def _hyde_query(query_text: str) -> str:
    return (
        "Гипотетический ленинский анализ: классовый характер явления, "
        "материальные причины, противоречия и выводы. "
        f"Тема: {query_text}"
    )


def _dense_search(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    limit: int,
) -> list[dict]:
    response = client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def _sparse_search(
    client: QdrantClient,
    collection_name: str,
    sparse_indices: list[int],
    sparse_values: list[float],
    limit: int,
) -> list[dict]:
    response = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(indices=sparse_indices, values=sparse_values),
        using="sparse",
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def _ontology_search(query_text: str, ontology_rows: list[dict[str, str]], limit: int) -> list[str]:
    terms = set(re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", query_text.lower()))
    scored: list[tuple[str, int]] = []
    for row in ontology_rows:
        concepts = [item for item in row.get("concepts", "").split("|") if item]
        score = sum(1 for concept in concepts if concept.lower() in terms)
        if score > 0:
            scored.append((row["source_id"], score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [source_id for source_id, _ in scored[:limit]]


def _unique_source_ids(points: list[dict]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for point in points:
        source_id = str(point.payload.get("source_id"))
        if source_id in seen:
            continue
        seen.add(source_id)
        result.append(source_id)
    return result


def _source_stance_map(points: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for point in points:
        source_id = str(point.payload.get("source_id"))
        mapping[source_id] = str(point.payload.get("stance_type", "contextual"))
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval sandbox experiments.")
    parser.add_argument("--config", default="config/retrieval_sandbox.yaml")
    parser.add_argument("--out-json", default=".cursor/artifacts/sandbox/retrieval_sandbox_results.json")
    parser.add_argument("--out-md", default=".cursor/artifacts/sandbox/retrieval_sandbox_summary.md")
    parser.add_argument("--max-queries", type=int, default=40)
    args = parser.parse_args()

    config = _load_config(path=(REPO_ROOT / args.config).resolve())
    eval_rows = _read_eval(path=config.eval_dataset_path)[: args.max_queries]
    ontology_rows = _read_ontology(path=config.ontology_tags_path)

    dense_model = SentenceTransformer(
        model_name_or_path=config.dense_model,
        trust_remote_code=config.trust_remote_code,
        device=config.device,
    )
    sparse_encoder = Bm25SparseEncoder.load(path=config.sparse_state_path)
    client = QdrantClient(path=str(config.qdrant_path))

    mode_predictions: dict[str, list[list[str]]] = {mode: [] for mode in config.modes}
    positives: list[str] = []
    mode_latency: dict[str, list[float]] = {mode: [] for mode in config.modes}

    for row in eval_rows:
        query_text = _extract_query_text(raw_query=row["query"])
        positive_source = row["positive_source_id"]
        positives.append(positive_source)

        dense_query_vector = dense_model.encode([query_text], normalize_embeddings=True)[0].tolist()
        sparse_vector = sparse_encoder.encode_query(text=query_text)
        dense_points = _dense_search(
            client=client,
            collection_name=config.collection_name,
            dense_vector=dense_query_vector,
            limit=config.top_k,
        )
        sparse_points = _sparse_search(
            client=client,
            collection_name=config.collection_name,
            sparse_indices=sparse_vector.indices,
            sparse_values=sparse_vector.values,
            limit=config.top_k,
        )
        onto_source_ids = _ontology_search(query_text=query_text, ontology_rows=ontology_rows, limit=config.top_k)

        dense_ids = _unique_source_ids(points=dense_points)
        sparse_ids = _unique_source_ids(points=sparse_points)
        source_stance = _source_stance_map(points=dense_points + sparse_points)

        for mode in config.modes:
            started = time.perf_counter()
            rank_lists: dict[str, list[str]] = {}
            if mode in {"dense", "hybrid", "hybrid_onto", "hyde_hybrid"}:
                rank_lists["dense"] = dense_ids
            if mode in {"hybrid", "hybrid_onto", "hyde_hybrid"}:
                rank_lists["sparse"] = sparse_ids
            if mode in {"hybrid_onto"}:
                rank_lists["onto"] = onto_source_ids
            if mode == "hyde_hybrid":
                hyde_query = _hyde_query(query_text=query_text)
                hyde_dense = dense_model.encode([hyde_query], normalize_embeddings=True)[0].tolist()
                hyde_points = _dense_search(
                    client=client,
                    collection_name=config.collection_name,
                    dense_vector=hyde_dense,
                    limit=config.top_k,
                )
                rank_lists["dense_hyde"] = _unique_source_ids(points=hyde_points)

            weights = config.retriever_weights.copy()
            if "dense_hyde" in rank_lists and "dense_hyde" not in weights:
                weights["dense_hyde"] = weights.get("dense", 1.0)

            fused = reciprocal_rank_fusion(
                rank_lists=rank_lists,
                retriever_weights=weights,
                k=config.rrf_k,
            )
            boosted = apply_source_boost(
                scores=fused,
                source_stance=source_stance,
                boosts=config.source_boosts,
            )
            ranked = [source_id for source_id, _ in sorted(boosted.items(), key=lambda item: item[1], reverse=True)]
            mode_predictions[mode].append(ranked)
            mode_latency[mode].append((time.perf_counter() - started) * 1000)

    results: dict[str, dict[str, float]] = {}
    for mode in config.modes:
        mode_recall = recall_at_k(predictions=mode_predictions[mode], positives=positives, k=5)
        latencies = mode_latency[mode]
        results[mode] = {
            "recall_at_5": mode_recall,
            "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
        }

    best_mode = max(results.items(), key=lambda item: item[1]["recall_at_5"])[0] if results else "none"
    payload = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "queries_evaluated": len(eval_rows),
        "best_mode": best_mode,
        "results": results,
    }

    out_json = (REPO_ROOT / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = (REPO_ROOT / args.out_md).resolve()
    lines = [
        "# Retrieval Sandbox Summary",
        "",
        f"- Generated at (UTC): {payload['generated_at_utc']}",
        f"- Queries evaluated: `{payload['queries_evaluated']}`",
        f"- Best mode by Recall@5: `{payload['best_mode']}`",
        "",
    ]
    for mode, metrics in results.items():
        lines.append(
            f"- `{mode}`: Recall@5=`{metrics['recall_at_5']:.4f}`, "
            f"mean latency=`{metrics['latency_ms_mean']:.2f} ms`"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Best mode: {best_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
