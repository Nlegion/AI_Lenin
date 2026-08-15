#!/usr/bin/env python3
"""Local one-shot dry-run for fetch -> retrieval trace -> analysis."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from time import perf_counter
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.generation.fallback import recommend_persona_model  # noqa: E402
from src.core.generation.pipeline import AnalysisGenerationPipeline  # noqa: E402
from src.core.lenin_analyzer import LeninAnalyzer  # noqa: E402
from src.core.retrieval.provider_factory import load_retrieval_pipeline_config  # noqa: E402
from src.core.safety.news_guard import NewsGuard, load_news_guard_config  # noqa: E402
from src.core.settings.device import hardware_report, resolve_torch_device  # noqa: E402
from src.core.settings.generation_config import load_generation_config  # noqa: E402
from src.modules.news_system.fetcher import NewsFetcher  # noqa: E402


@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    url: str
    news_id: str | None = None


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_fixtures(path: Path) -> dict[str, NewsItem]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = payload.get("fixtures", {})
    fixtures: dict[str, NewsItem] = {}
    for name, row in section.items():
        fixtures[name] = NewsItem(
            title=str(row.get("title", "")),
            content=str(row.get("content", "")),
            source=str(row.get("source", "fixture")),
            url=str(row.get("url", "about:blank")),
            news_id=f"fixture:{name}",
        )
    return fixtures


def _load_news_text(path_arg: str) -> NewsItem:
    if path_arg == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path_arg).read_text(encoding="utf-8")
    parts = [line.strip() for line in raw.splitlines() if line.strip()]
    if not parts:
        raise ValueError("news text input is empty")
    title = parts[0]
    content = "\n".join(parts[1:]).strip() or parts[0]
    return NewsItem(
        title=title,
        content=content,
        source="manual_input",
        url="stdin://",
        news_id="manual",
    )


def _fetch_item(news_id: str | None) -> NewsItem | None:
    fetcher = NewsFetcher()
    rows = fetcher.fetch_all()
    if not rows:
        return None
    if news_id:
        for row in rows:
            if str(row.get("id")) == news_id:
                return NewsItem(
                    title=str(row["title"]),
                    content=str(row["content"]),
                    source=str(row.get("source", "unknown")),
                    url=str(row.get("url", "")),
                    news_id=str(row.get("id")),
                )
        return None
    latest = sorted(rows, key=lambda item: item.get("date"), reverse=True)[0]
    return NewsItem(
        title=str(latest["title"]),
        content=str(latest["content"]),
        source=str(latest.get("source", "unknown")),
        url=str(latest.get("url", "")),
        news_id=str(latest.get("id")),
    )


def _resolve_provider(analyzer: LeninAnalyzer):
    provider = analyzer.retrieval_provider
    # Migration wrapper was removed; qdrant_only provider is used directly.
    qdrant_provider = provider
    return provider, qdrant_provider


def _provider_inputs_available(
    *, retrieval_config, repo_root: Path
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    sparse_state = (repo_root / retrieval_config.sparse_state_path).resolve()
    ontology_tags = (repo_root / retrieval_config.ontology_tags_path).resolve()
    if not sparse_state.exists():
        missing.append(str(sparse_state))
    if not ontology_tags.exists():
        missing.append(str(ontology_tags))
    return (len(missing) == 0), missing


def _safe_print(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    stream = getattr(sys.stdout, "buffer", None)
    payload = (text + "\n").encode(encoding=encoding, errors="replace")
    if stream is not None:
        stream.write(payload)
        stream.flush()
    else:
        sys.stdout.write(payload.decode(encoding=encoding, errors="replace"))
        sys.stdout.flush()


def _render_section(title: str, payload: str) -> None:
    _safe_print(f"\n## {title}")
    _safe_print(payload.strip() if payload.strip() else "(empty)")


def _append_audit_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _recent_high_risk_count(path: Path, tail: int = 200) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    recent = lines[-tail:]
    count = 0
    for line in recent:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("high_risk"):
            count += 1
    return count


def _top_n(items: list[dict[str, Any]], limit: int) -> str:
    if not items:
        return "(empty)"
    lines: list[str] = []
    for row in items[:limit]:
        lines.append(
            " | ".join(
                [
                    f"query={row.get('query', '')}",
                    f"rank={row.get('rank', '')}",
                    f"chunk={row.get('chunk_id', '')}",
                    f"source={row.get('source_id', '')}",
                    f"stance={row.get('stance_type', '')}",
                    f"score={row.get('score', '')}",
                ]
            )
        )
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local RAG dry-run with retrieval diagnostics."
    )
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--news-id", default=None)
    parser.add_argument(
        "--news-text",
        default=None,
        help="Path to file with title/content or '-' for stdin.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--bypass-safety", action="store_true")
    parser.add_argument("--allow-legacy-fallback", action="store_true")
    parser.add_argument("--persona-model", choices=["base_strong"], default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--retrieval-config", default="config/retrieval_pipeline.yaml")
    parser.add_argument("--generation-config", default="config/generation.yaml")
    parser.add_argument("--news-guard-config", default="config/news_guard.yaml")
    parser.add_argument("--fixtures-config", default="config/dryrun_fixtures.yaml")
    parser.add_argument(
        "--audit-log", default=".cursor/artifacts/safety/dryrun_audit.jsonl"
    )
    parser.add_argument("--alert-threshold", type=int, default=5)
    args = parser.parse_args()

    retrieval_config_path = (REPO_ROOT / args.retrieval_config).resolve()
    retrieval_config = load_retrieval_pipeline_config(config_path=retrieval_config_path)
    guard_config = load_news_guard_config(
        path=(REPO_ROOT / args.news_guard_config).resolve()
    )
    guard = NewsGuard(config=guard_config)
    fixtures = _load_fixtures(path=(REPO_ROOT / args.fixtures_config).resolve())
    inputs_ok, missing_inputs = _provider_inputs_available(
        retrieval_config=retrieval_config,
        repo_root=REPO_ROOT,
    )
    if (
        retrieval_config.migration.mode == "qdrant_only"
        and not inputs_ok
        and not args.allow_legacy_fallback
    ):
        _render_section(
            "SAFETY",
            "retrieval provider unavailable in qdrant_only mode; missing inputs: "
            + ", ".join(missing_inputs),
        )
        return 3

    item: NewsItem | None = None
    notice = ""
    if args.news_text:
        item = _load_news_text(path_arg=args.news_text)
    elif args.fixture:
        item = fixtures.get(args.fixture)
        if item is None:
            raise ValueError(f"Unknown fixture: {args.fixture}")
    else:
        item = _fetch_item(news_id=args.news_id)
        if item is None:
            item = fixtures.get("economy")
            notice = "FETCH_EMPTY_FALLBACK_FIXTURE=economy"

    if item is None:
        raise ValueError("No input news item could be resolved.")

    if args.bypass_safety:
        _safe_print("\n## WARNING")
        _safe_print(
            "BYPASS_SAFETY enabled: NewsGate checks are skipped. Developer-only override."
        )
        safety_info = "BYPASS_SAFETY=true (developer override)"
        gate = None
    else:
        gate = guard.evaluate_input(
            title=item.title, content=item.content, source=item.source
        )
        if gate.decision in {"deny", "quarantine", "skip"}:
            _render_section(
                "SAFETY",
                "\n".join(
                    [
                        f"decision={gate.decision}",
                        f"reason={gate.reason}",
                        f"codes={','.join(gate.reason_codes)}",
                        f"message={gate.message}",
                    ]
                ),
            )
            return 2
        safety_info = f"decision={gate.decision}; reason={gate.reason}; codes={','.join(gate.reason_codes)}"

    generation_config = load_generation_config(
        path=(REPO_ROOT / args.generation_config).resolve()
    )
    if args.persona_model:
        generation_config = generation_config.with_persona_model(args.persona_model)
    recommended = recommend_persona_model(config=generation_config, base_dir=REPO_ROOT)
    if recommended != generation_config.persona_model:
        _safe_print(
            f"\n## FALLBACK_RECOMMENDATION\npersona_model={recommended} "
            "(NewsGuard incident threshold exceeded; auto-switch disabled in dry-run)"
        )

    analyzer = LeninAnalyzer(persona_model=generation_config.persona_model)
    started_at = perf_counter()
    provider, qdrant_provider = _resolve_provider(analyzer=analyzer)
    mode = retrieval_config.migration.mode
    if provider is None and mode == "qdrant_only" and not args.allow_legacy_fallback:
        _render_section("SAFETY", "retrieval provider unavailable in qdrant_only mode")
        return 3

    key_concepts = analyzer.extract_key_concepts(item.content)
    enhanced_query = f"{item.title} {item.content[:200]} {' '.join(key_concepts)}"

    trace: dict[str, Any] = {}
    fallback_used = False
    context = ""
    if qdrant_provider is not None and hasattr(qdrant_provider, "retrieve_with_trace"):
        candidates, trace = qdrant_provider.retrieve_with_trace(
            query_text=enhanced_query,
            apply_judge=not args.skip_judge,
        )
        context = qdrant_provider.render_context(candidates=candidates)
    elif provider is not None and mode != "qdrant_only":
        result = provider.retrieve_context(
            query_text=enhanced_query, author_filter="Ленин"
        )
        context = result.context
        trace = {
            "query_variants": [enhanced_query],
            "dense": [],
            "sparse": [],
            "onto": [],
        }
    elif args.allow_legacy_fallback:
        fallback_used = True
        context = analyzer.context_orchestrator.build_context(
            enhanced_query=enhanced_query
        )
        trace = {
            "query_variants": [enhanced_query],
            "dense": [],
            "sparse": [],
            "onto": [],
        }
    else:
        _render_section(
            "SAFETY", "qdrant provider unavailable and legacy fallback disabled"
        )
        return 4

    def _fixed_context(_query: str) -> str:
        return context

    pipeline = AnalysisGenerationPipeline(
        base_dir=REPO_ROOT,
        context_builder=_fixed_context,
        news_guard=guard,
        text_cleaner=analyzer.text_cleaner,
        generation_config=generation_config,
        persona_model=generation_config.persona_model,
        apply_fallback_recommendation=False,
    )
    try:
        await analyzer.initialize_session()
        pipeline_result = await pipeline.generate(
            news_title=item.title,
            news_content=item.content,
            enhanced_query=enhanced_query,
            warn_only_guard=True,
        )
        output = pipeline_result.guard_result
        hallucination_codes = pipeline_result.hallucination_codes
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        persona_model = pipeline_result.backend
        model_name = pipeline_result.model_name
        gen_latency_ms = pipeline_result.latency_ms
    except Exception as error:  # noqa: BLE001
        from src.core.safety.news_guard import OutputGuardResult

        output = OutputGuardResult(
            blocked=False,
            moderated_text=f"Ошибка анализа: недоступен LLM endpoint ({error}).",
            reason_codes=["generation_error"],
        )
        hallucination_codes = []
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        persona_model = generation_config.persona_model
        model_name = generation_config.active_backend().model_name
        gen_latency_ms = elapsed_ms
    finally:
        await pipeline.close()
        await analyzer.close_session()

    _render_section(
        "INPUT",
        "\n".join(
            [
                f"title={item.title}",
                f"source={item.source}",
                f"url={item.url}",
                f"news_id={item.news_id or 'n/a'}",
                f"notice={notice or 'none'}",
            ]
        ),
    )
    _render_section("REWRITE", "\n".join(trace.get("query_variants", [enhanced_query])))
    if args.verbose:
        _render_section(
            "RETRIEVAL_DENSE", _top_n(trace.get("dense", []), limit=args.top_n)
        )
        _render_section(
            "RETRIEVAL_SPARSE", _top_n(trace.get("sparse", []), limit=args.top_n)
        )
        _render_section(
            "RETRIEVAL_ONTO", _top_n(trace.get("onto", []), limit=args.top_n)
        )
        _render_section(
            "ARBITER",
            json.dumps(
                {
                    "merged_scores": trace.get("merged_scores", {}),
                    "boosted_scores": trace.get("boosted_scores", {}),
                    "judge_scores": trace.get("judge_scores", {}),
                    "final_scores": trace.get("final_scores", {}),
                    "judge_enabled": trace.get("judge_enabled", not args.skip_judge),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    _render_section("RAG_CONTEXT", context[:6000])
    _render_section("ANALYSIS", output.moderated_text)
    _render_section(
        "SAFETY",
        "\n".join(
            [
                safety_info,
                f"output_blocked={output.blocked}",
                f"output_codes={','.join(output.reason_codes + hallucination_codes)}",
                f"safe_mode={guard.config.output_guard.safe_mode}",
                f"persona_model={persona_model}",
                f"legacy_fallback={str(fallback_used).lower()}",
            ]
        ),
    )
    _render_section(
        "METADATA",
        json.dumps(
            {
                "migration_mode": mode,
                "collection_name": retrieval_config.collection_name,
                "embedding_model": retrieval_config.dense_model,
                "persona_model": persona_model,
                "generation_model": model_name,
                "safe_mode": guard.config.output_guard.safe_mode,
                "policy_version": guard.config.policy_version,
                "config_hash_retrieval": _sha256_file(retrieval_config_path),
                "config_hash_news_guard": _sha256_file(
                    (REPO_ROOT / args.news_guard_config).resolve()
                ),
                "config_hash_generation": _sha256_file(
                    (REPO_ROOT / args.generation_config).resolve()
                ),
                "corpus_manifest_hash": _sha256_file(
                    (
                        REPO_ROOT
                        / ".cursor/artifacts/cleaning/corpus_repro_manifest_v1.tsv"
                    ).resolve()
                ),
                "latency_ms_total": elapsed_ms,
                "latency_ms_generation": gen_latency_ms,
                "fallback_hook_enabled": generation_config.safety.fallback.enabled,
                "fallback_recommendation": recommended,
                "hardware": hardware_report(
                    resolved_device=resolve_torch_device(
                        preferred=retrieval_config.device
                    ),
                    fallback_to_cpu=bool(
                        getattr(retrieval_config, "fallback_to_cpu", True)
                    ),
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    audit_path = (REPO_ROOT / args.audit_log).resolve()
    output_codes = output.reason_codes + hallucination_codes
    high_risk = any(
        token in " ".join(output_codes).lower()
        for token in ("classifier:", "block:", "hallucination_marked", "pii_redact")
    ) or (gate is not None and gate.decision in {"deny", "quarantine", "skip"})
    audit_payload = {
        "ts_epoch_ms": int(time.time() * 1000),
        "news_id": item.news_id,
        "source": item.source,
        "title_hash": hashlib.sha256(item.title.encode("utf-8")).hexdigest(),
        "mode": mode,
        "persona_model": persona_model,
        "safe_mode": guard.config.output_guard.safe_mode,
        "gate_decision": gate.decision if gate is not None else "bypass",
        "gate_codes": gate.reason_codes if gate is not None else [],
        "output_blocked": output.blocked,
        "output_codes": output_codes,
        "high_risk": high_risk,
    }
    _append_audit_log(path=audit_path, payload=audit_payload)
    if _recent_high_risk_count(path=audit_path) >= args.alert_threshold:
        _render_section(
            "ALERT",
            "High-risk filter triggers exceeded threshold; notify administrator.",
        )
    if fallback_used:
        _safe_print("\nLEGACY FALLBACK: context source is legacy path")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
