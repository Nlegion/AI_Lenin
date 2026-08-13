"""Tests for manual censor terms loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.core.safety.manual_terms_loader import ManualTermsLoader
from src.core.safety.manual_terms_policy import AMBIGUITY_BANLIST

ROOT = Path(__file__).resolve().parents[1]
TERMS_DIR = ROOT / "config" / "censor_terms"


def test_loader_reads_production_index() -> None:
    bundle = ManualTermsLoader(terms_dir=TERMS_DIR).load()
    assert bundle.content_hash
    assert any(rule.category_id == "WAR_OPERATIONAL" for rule in bundle.rules)
    assert any(rule.category_id == "SPORT_BLOCKED" for rule in bundle.rules)


def test_loaded_terms_exclude_ambiguity_banlist() -> None:
    bundle = ManualTermsLoader(terms_dir=TERMS_DIR).load()
    for rule in bundle.rules:
        overlap = sorted(term for term in rule.terms if term in AMBIGUITY_BANLIST)
        assert overlap == [], f"{rule.category_id} contains banlist terms: {overlap}"


def test_first_wins_dedup_prefers_earlier_index_category(tmp_path: Path) -> None:
    (tmp_path / "index.yaml").write_text(
        yaml.safe_dump(
            {
                "categories": [
                    {
                        "id": "WAR_OPERATIONAL",
                        "file": "war.yaml",
                        "enabled": True,
                        "decision": "hard_block",
                        "reason_code": "manual_war_operational_hard_block",
                    },
                    {
                        "id": "AUTO",
                        "file": "auto.yaml",
                        "enabled": True,
                        "decision": "hard_block",
                        "reason_code": "manual_auto_hard_block",
                    },
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "war.yaml").write_text(
        yaml.safe_dump({"terms": ["shared-token", "бпла"]}, allow_unicode=True),
        encoding="utf-8",
    )
    (tmp_path / "auto.yaml").write_text(
        yaml.safe_dump({"terms": ["shared-token", "toyota"]}, allow_unicode=True),
        encoding="utf-8",
    )
    (tmp_path / "overrides.yaml").write_text(
        yaml.safe_dump({"force_include": [], "force_exclude": []}),
        encoding="utf-8",
    )
    bundle = ManualTermsLoader(terms_dir=tmp_path).load()
    by_id = {rule.category_id: rule for rule in bundle.rules}
    assert "shared-token" in by_id["WAR_OPERATIONAL"].terms
    assert "shared-token" not in by_id["AUTO"].terms
    assert bundle.duplicates
    assert any(item[0] == "shared-token" for item in bundle.duplicates)


def test_corrupt_yaml_keeps_last_good(tmp_path: Path) -> None:
    (tmp_path / "index.yaml").write_text(
        yaml.safe_dump(
            {
                "categories": [
                    {
                        "id": "FIRE",
                        "file": "fire.yaml",
                        "enabled": True,
                        "decision": "hard_block",
                        "reason_code": "manual_fire_hard_block",
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "fire.yaml").write_text(
        yaml.safe_dump({"terms": ["пожар"]}, allow_unicode=True),
        encoding="utf-8",
    )
    loader = ManualTermsLoader(terms_dir=tmp_path)
    good = loader.get_bundle()
    assert "пожар" in good.rules[0].terms

    (tmp_path / "fire.yaml").write_text("{ not: valid: yaml [", encoding="utf-8")
    retained = loader.reload_if_changed()
    assert retained.content_hash == good.content_hash
    assert "пожар" in retained.rules[0].terms


def test_force_include_and_exclude_overrides(tmp_path: Path) -> None:
    (tmp_path / "index.yaml").write_text(
        yaml.safe_dump(
            {
                "categories": [
                    {
                        "id": "CINEMA",
                        "file": "cinema.yaml",
                        "enabled": True,
                        "decision": "hard_block",
                        "reason_code": "manual_cinema_hard_block",
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "cinema.yaml").write_text(
        yaml.safe_dump({"terms": ["блокбастер", "drop-me"]}, allow_unicode=True),
        encoding="utf-8",
    )
    (tmp_path / "overrides.yaml").write_text(
        yaml.safe_dump(
            {
                "force_include": [{"category": "CINEMA", "term": "extra-term"}],
                "force_exclude": [{"category": "CINEMA", "term": "drop-me"}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    rule = ManualTermsLoader(terms_dir=tmp_path).load().rules[0]
    assert "блокбастер" in rule.terms
    assert "extra-term" in rule.terms
    assert "drop-me" not in rule.terms
