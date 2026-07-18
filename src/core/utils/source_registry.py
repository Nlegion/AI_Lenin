"""Source registry builder for corpus inventory and source typing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib

import yaml

from src.core.settings.source_registry_rules import (
    DEFAULT_SOURCE_REGISTRY_RULES,
    SourceRegistryRules,
)


STANCE_VALUES = {"core_self", "influence_agree", "influence_critical", "contextual"}
CONTAINER_DIRECTORIES = {"intellectual", "ultimate_cleaned_ontology", "books"}


@dataclass(frozen=True)
class SourceRegistryRecord:
    source_id: str
    source_path: str
    author: str
    work: str
    stance_type: str
    period: str
    language: str
    include: bool
    notes: str


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", " ").replace("_", " ")


def _load_rules_from_yaml(config_path: Path) -> SourceRegistryRules:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("source_registry", payload)
    default_rules = DEFAULT_SOURCE_REGISTRY_RULES

    return SourceRegistryRules(
        core_authors=set(section.get("core_authors", list(default_rules.core_authors))),
        influence_agree_authors=set(
            section.get("influence_agree_authors", list(default_rules.influence_agree_authors))
        ),
        influence_critical_authors=set(
            section.get("influence_critical_authors", list(default_rules.influence_critical_authors))
        ),
        contextual_authors=set(section.get("contextual_authors", list(default_rules.contextual_authors))),
        path_overrides=dict(section.get("path_overrides", default_rules.path_overrides)),
        allowed_extensions=tuple(section.get("allowed_extensions", default_rules.allowed_extensions)),
    )


def load_source_registry_rules(config_path: Path | None = None) -> SourceRegistryRules:
    if config_path is None or not config_path.exists():
        return DEFAULT_SOURCE_REGISTRY_RULES
    return _load_rules_from_yaml(config_path=config_path)


def classify_stance_type(author: str, relative_path: str, rules: SourceRegistryRules) -> str:
    normalized_author = _normalize(author)
    normalized_path = relative_path.lower().replace("\\", "/")

    for prefix, stance_type in rules.path_overrides.items():
        if normalized_path.startswith(prefix.lower().replace("\\", "/")):
            if stance_type in STANCE_VALUES:
                return stance_type

    if normalized_author in {_normalize(item) for item in rules.core_authors}:
        return "core_self"
    if normalized_author in {_normalize(item) for item in rules.influence_agree_authors}:
        return "influence_agree"
    if normalized_author in {_normalize(item) for item in rules.influence_critical_authors}:
        return "influence_critical"
    return "contextual"


def _build_source_id(source_path: str) -> str:
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()
    return f"src_{digest[:16]}"


def _extract_author(relative_path: str) -> str:
    parts = relative_path.split("/")
    if not parts:
        return "unknown"
    if parts[0].lower() in CONTAINER_DIRECTORIES and len(parts) > 1:
        return parts[1]
    return parts[0]


def _iter_corpus_files(corpus_root: Path, allowed_extensions: tuple[str, ...]) -> list[Path]:
    discovered: list[Path] = []
    normalized_ext = {item.lower() for item in allowed_extensions}
    for file_path in corpus_root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in normalized_ext:
            discovered.append(file_path)
    return sorted(discovered)


def build_source_registry(corpus_root: Path, rules: SourceRegistryRules) -> list[SourceRegistryRecord]:
    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus root does not exist: {corpus_root}")

    records: list[SourceRegistryRecord] = []
    for file_path in _iter_corpus_files(corpus_root=corpus_root, allowed_extensions=rules.allowed_extensions):
        relative = file_path.relative_to(corpus_root).as_posix()
        author = _extract_author(relative_path=relative)
        stance_type = classify_stance_type(author=author, relative_path=relative, rules=rules)
        record = SourceRegistryRecord(
            source_id=_build_source_id(source_path=relative),
            source_path=relative,
            author=author,
            work=file_path.stem,
            stance_type=stance_type,
            period="unknown",
            language="ru",
            include=True,
            notes="",
        )
        records.append(record)
    return records


def export_source_registry_tsv(records: list[SourceRegistryRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(SourceRegistryRecord.__annotations__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_registry_summary(records: list[SourceRegistryRecord]) -> dict[str, int]:
    summary: dict[str, int] = {"total_records": len(records)}
    for stance_type in STANCE_VALUES:
        summary[f"stance_{stance_type}"] = 0
    for record in records:
        summary[f"stance_{record.stance_type}"] += 1
    return summary
