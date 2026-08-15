"""Load curated manual censor terms from config/censor_terms."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.safety.pre_rag_censor_types import CensorDecision

logger = logging.getLogger(__name__)

ALLOWED_DECISIONS = frozenset({"hard_block", "review", "skip", "allow"})


@dataclass(frozen=True)
class ManualTermRule:
    category_id: str
    decision: CensorDecision
    reason_code: str
    enabled: bool
    terms: frozenset[str]


@dataclass
class ManualTermsBundle:
    rules: tuple[ManualTermRule, ...]
    content_hash: str
    duplicates: tuple[tuple[str, str, str], ...] = ()
    source_dir: str = ""


@dataclass
class ManualTermsLoader:
    """Load/validate manual terms with first-wins dedup and last-good reload."""

    terms_dir: Path
    _bundle: ManualTermsBundle | None = field(default=None, init=False, repr=False)

    def get_bundle(self) -> ManualTermsBundle:
        if self._bundle is None:
            self._bundle = self.load()
        return self._bundle

    def reload_if_changed(self) -> ManualTermsBundle:
        """Reload from disk; keep last-good bundle on failure."""
        try:
            new_bundle = self.load()
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "manual_terms_reload_failed dir=%s err=%s",
                self.terms_dir,
                error,
            )
            if self._bundle is None:
                raise
            return self._bundle
        if self._bundle is None or new_bundle.content_hash != self._bundle.content_hash:
            self._bundle = new_bundle
            logger.info(
                "manual_terms_reloaded hash=%s rules=%s",
                new_bundle.content_hash,
                len(new_bundle.rules),
            )
        return self._bundle

    def load(self) -> ManualTermsBundle:
        terms_dir = self.terms_dir
        index_path = terms_dir / "index.yaml"
        overrides_path = terms_dir / "overrides.yaml"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing censor terms index: {index_path}")

        index_payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
        categories = index_payload.get("categories") or []
        if not isinstance(categories, list) or not categories:
            raise ValueError("censor_terms index.yaml must define non-empty categories")

        overrides_payload: dict[str, Any] = {}
        if overrides_path.is_file():
            overrides_payload = (
                yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
            )

        force_include = _parse_override_entries(
            overrides_payload.get("force_include") or []
        )
        force_exclude = {
            (
                str(item.get("category", "")).strip().upper(),
                str(item.get("term", "")).strip().casefold(),
            )
            for item in (overrides_payload.get("force_exclude") or [])
            if isinstance(item, dict)
        }

        hash_parts: list[bytes] = [index_path.read_bytes()]
        if overrides_path.is_file():
            hash_parts.append(overrides_path.read_bytes())

        seen_terms: dict[str, str] = {}
        duplicates: list[tuple[str, str, str]] = []
        rules: list[ManualTermRule] = []

        for entry in categories:
            if not isinstance(entry, dict):
                raise ValueError("Each categories entry must be a mapping")
            category_id = str(entry.get("id") or "").strip().upper()
            file_name = str(entry.get("file") or "").strip()
            reason_code = str(entry.get("reason_code") or "").strip()
            decision_raw = str(entry.get("decision") or "hard_block").strip().casefold()
            enabled = bool(entry.get("enabled", True))
            if not category_id or not file_name or not reason_code:
                raise ValueError(f"Invalid category entry: {entry!r}")
            if decision_raw not in ALLOWED_DECISIONS:
                raise ValueError(
                    f"Unsupported decision for {category_id}: {decision_raw}"
                )
            decision: CensorDecision = decision_raw  # type: ignore[assignment]

            terms_path = terms_dir / file_name
            if not terms_path.is_file():
                raise FileNotFoundError(
                    f"Missing terms file for {category_id}: {terms_path}"
                )
            hash_parts.append(terms_path.read_bytes())
            payload = yaml.safe_load(terms_path.read_text(encoding="utf-8")) or {}
            raw_terms = payload.get("terms") or []
            if not isinstance(raw_terms, list):
                raise ValueError(f"{file_name} terms must be a list")

            kept: set[str] = set()
            for raw in raw_terms:
                term = str(raw).strip().casefold()
                if not term:
                    continue
                if (category_id, term) in force_exclude:
                    continue
                owner = seen_terms.get(term)
                if owner is not None:
                    duplicates.append((term, owner, category_id))
                    continue
                seen_terms[term] = category_id
                kept.add(term)

            for term in force_include.get(category_id, ()):
                if (category_id, term) in force_exclude:
                    continue
                owner = seen_terms.get(term)
                if owner is not None and owner != category_id:
                    duplicates.append((term, owner, category_id))
                    continue
                seen_terms[term] = category_id
                kept.add(term)

            rules.append(
                ManualTermRule(
                    category_id=category_id,
                    decision=decision,
                    reason_code=reason_code,
                    enabled=enabled,
                    terms=frozenset(kept),
                )
            )

        digest = hashlib.sha256(b"\n".join(hash_parts)).hexdigest()[:16]
        if duplicates:
            logger.warning(
                "manual_terms_duplicates count=%s sample=%s",
                len(duplicates),
                duplicates[:5],
            )
        return ManualTermsBundle(
            rules=tuple(rules),
            content_hash=digest,
            duplicates=tuple(duplicates),
            source_dir=str(terms_dir),
        )


def default_censor_terms_dir(base_dir: Path) -> Path:
    return base_dir / "config" / "censor_terms"


def _parse_override_entries(entries: list[Any]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip().upper()
        term = str(item.get("term") or "").strip().casefold()
        if not category or not term:
            continue
        grouped.setdefault(category, []).append(term)
    return {key: tuple(values) for key, values in grouped.items()}
