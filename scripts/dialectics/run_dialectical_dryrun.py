#!/usr/bin/env python3
"""Dry-run dialectical brief (+ optional analysis) for smoke metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.analysis.context_orchestrator import AnalysisContextOrchestrator  # noqa: E402
from src.core.analysis.dialectical_config import load_dialectical_config  # noqa: E402
from src.core.analysis.jaccard_metrics import jaccard_overlap  # noqa: E402
from src.core.retrieval.provider_factory import build_provider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Dialectical dry-run smoke.")
    parser.add_argument("--config", default="config/retrieval_pipeline.yaml")
    parser.add_argument("--fixtures", default="config/dryrun_fixtures.yaml")
    parser.add_argument(
        "--out-jsonl",
        default=".cursor/artifacts/qdrant/dialectical_dryrun_smoke.jsonl",
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    dialectical = load_dialectical_config(config_path=config_path)
    from dataclasses import replace

    dialectical = replace(dialectical, enabled=True)

    provider = build_provider(
        config_path=config_path,
        base_dir=REPO_ROOT,
    )
    orchestrator = AnalysisContextOrchestrator(
        retrieval_provider=provider,
        dialectical_config=dialectical,
        taxonomy_path=REPO_ROOT / "config" / "ontology_taxonomy.yaml",
    )

    fixtures_path = REPO_ROOT / args.fixtures
    news_items: list[dict] = []
    if fixtures_path.exists():
        import yaml

        payload = yaml.safe_load(fixtures_path.read_text(encoding="utf-8")) or {}
        news_items = payload.get("news", payload.get("items", []))
    if not news_items:
        news_items = [
            {
                "title": f"Fixture news {index}",
                "content": "Санкции, капитал и империализм в международной политике. "
                * 5,
            }
            for index in range(args.limit)
        ]
    news_items = news_items[: args.limit]

    out_path = (REPO_ROOT / args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in news_items:
            title = str(item.get("title", ""))
            content = str(item.get("content", item.get("body", "")))
            concepts = [
                token
                for token in ("капитал", "империализм", "санкции")
                if token in content.casefold()
            ]
            brief = orchestrator.build_evidence_brief(
                news_title=title,
                news_content=content,
                key_concepts=concepts,
                enhanced_query=f"{title} {content[:200]}",
            )
            r1_text = " ".join(entry.text for entry in brief.r1_core_self)
            r2_text = " ".join(entry.text for entry in brief.r2_influence_agree)
            r3_text = " ".join(entry.text for entry in brief.r3_influence_critical)
            fake_analysis = r1_text[:500] or content[:200]
            row = {
                "title": title,
                "mode": brief.trace.get("orchestration_mode"),
                "r1_count": len(brief.r1_core_self),
                "r2_count": len(brief.r2_influence_agree),
                "r3_count": len(brief.r3_influence_critical),
                "warnings": brief.warnings,
                "r1_jaccard": jaccard_overlap(
                    left_text=fake_analysis, right_text=r1_text
                ),
                "r2_jaccard": jaccard_overlap(
                    left_text=fake_analysis, right_text=r2_text
                ),
                "r3_jaccard": jaccard_overlap(
                    left_text=fake_analysis, right_text=r3_text
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
