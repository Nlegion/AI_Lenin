from pathlib import Path

from src.core.utils.source_registry import (
    build_registry_summary,
    build_source_registry,
    classify_stance_type,
    export_source_registry_tsv,
    load_source_registry_rules,
)


def test_classify_stance_type_with_defaults():
    rules = load_source_registry_rules(config_path=None)
    assert (
        classify_stance_type(
            author="Ленин", relative_path="Ленин/work.txt", rules=rules
        )
        == "core_self"
    )
    assert (
        classify_stance_type(
            author="МарксЭнгельс", relative_path="МарксЭнгельс/work.txt", rules=rules
        )
        == "influence_agree"
    )
    assert (
        classify_stance_type(
            author="Unknown", relative_path="Unknown/work.txt", rules=rules
        )
        == "contextual"
    )


def test_build_source_registry_and_tsv_export(tmp_path: Path):
    corpus_root = tmp_path / "books"
    (corpus_root / "Ленин").mkdir(parents=True)
    (corpus_root / "МарксЭнгельс").mkdir(parents=True)
    (corpus_root / "Прочее").mkdir(parents=True)

    (corpus_root / "Ленин" / "труд_1.txt").write_text("a", encoding="utf-8")
    (corpus_root / "МарксЭнгельс" / "труд_2.txt").write_text("b", encoding="utf-8")
    (corpus_root / "Прочее" / "труд_3.md").write_text("c", encoding="utf-8")

    rules = load_source_registry_rules(config_path=None)
    records = build_source_registry(corpus_root=corpus_root, rules=rules)
    summary = build_registry_summary(records=records)

    assert summary["total_records"] == 3
    assert summary["stance_core_self"] == 1
    assert summary["stance_influence_agree"] == 1
    assert summary["stance_contextual"] == 1

    output_path = tmp_path / "registry.tsv"
    export_source_registry_tsv(records=records, output_path=output_path)
    exported = output_path.read_text(encoding="utf-8")
    assert "source_id" in exported
    assert "stance_type" in exported
