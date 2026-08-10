import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_local_rag_dryrun import _append_audit_log, _load_fixtures, _load_news_text, _recent_high_risk_count
from src.core.safety.news_guard import NewsGuard, load_news_guard_config


def test_load_news_text_from_file(tmp_path: Path):
    news_file = tmp_path / "news.txt"
    news_file.write_text("Заголовок\nСтрока 1\nСтрока 2", encoding="utf-8")
    item = _load_news_text(str(news_file))
    assert item.title == "Заголовок"
    assert "Строка 1" in item.content


def test_fixtures_available():
    fixtures = _load_fixtures(Path("config/dryrun_fixtures.yaml"))
    assert {
        "economy",
        "politics",
        "conflict",
        "sport",
        "provocative",
        "pii_private",
        "untrusted_disaster",
        "borderline_protest",
    }.issubset(set(fixtures))


def test_script_denies_military_fixture():
    command = [
        sys.executable,
        "scripts/run_local_rag_dryrun.py",
        "--fixture",
        "conflict",
    ]
    result = subprocess.run(command, cwd=Path.cwd(), check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "политикой безопасности" in (result.stdout + result.stderr)


def test_news_guard_hard_blocks_military_topic():
    guard = NewsGuard(config=load_news_guard_config(Path("config/news_guard.yaml")))
    result = guard.evaluate_input(
        title="Сводка по действиям вооруженных сил РФ",
        content="В материале обсуждаются действия ВС РФ и мобилизация.",
        source="TASS",
    )
    assert result.decision == "deny"
    assert "политикой безопасности" in result.message


def test_script_returns_nonzero_on_missing_provider(tmp_path: Path):
    config_path = tmp_path / "retrieval_bad.yaml"
    config_path.write_text(
        "\n".join(
            [
                "retrieval_pipeline:",
                "  enabled: false",
                "  collection_name: test_collection",
                "  qdrant_path: database/qdrant_local",
                "  dense_model: models/Giga-Embeddings-instruct",
                "  sparse_state_path: missing/sparse.json",
                "  ontology_tags_path: .cursor/artifacts/ontology/ontology_tags.tsv",
                "  migration:",
                "    mode: qdrant_only",
            ]
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "scripts/run_local_rag_dryrun.py",
        "--fixture",
        "economy",
        "--retrieval-config",
        str(config_path),
    ]
    result = subprocess.run(command, cwd=Path.cwd(), check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "provider unavailable" in (result.stdout + result.stderr).lower()


def test_audit_log_high_risk_counter(tmp_path: Path):
    audit_path = tmp_path / "audit.jsonl"
    _append_audit_log(audit_path, {"high_risk": False})
    _append_audit_log(audit_path, {"high_risk": True})
    _append_audit_log(audit_path, {"high_risk": True})
    assert _recent_high_risk_count(path=audit_path, tail=10) == 2


@pytest.mark.skipif(
    os.getenv("AI_LENIN_ENABLE_DRYRUN_INTEGRATION") != "1",
    reason="Integration dry-run is enabled only in prepared runtime environments.",
)
def test_integration_dryrun_fixture_execution():
    command = [
        sys.executable,
        "scripts/run_local_rag_dryrun.py",
        "--fixture",
        "economy",
        "--skip-judge",
        "--verbose",
    ]
    result = subprocess.run(command, cwd=Path.cwd(), check=False, capture_output=True, text=True)
    stdout = result.stdout
    assert result.returncode == 0
    for section in [
        "## INPUT",
        "## REWRITE",
        "## RETRIEVAL_DENSE",
        "## RETRIEVAL_SPARSE",
        "## RETRIEVAL_ONTO",
        "## ARBITER",
        "## RAG_CONTEXT",
        "## ANALYSIS",
        "## SAFETY",
        "## METADATA",
    ]:
        assert section in stdout
    assert "разжиган" not in stdout.lower()
