"""Pytest fixtures for dialectical retrieval tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def ensure_stance_index_if_local(repo_root: Path):
    """Best-effort index ensure for local Qdrant path used in integration tests."""
    qdrant_path = repo_root / "database" / "qdrant_local"
    if not qdrant_path.exists():
        return None
    import importlib.util

    script_path = repo_root / "scripts" / "ensure_qdrant_stance_index.py"
    spec = importlib.util.spec_from_file_location("ensure_qdrant_stance_index", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.ensure_stance_index(
        qdrant_path=qdrant_path,
        collection_name="philosophy_ontology_giga_v1",
        wait_timeout_sec=120.0,
    )
    if code not in {0}:
        pytest.skip(f"stance index ensure failed with code={code}")
    return code
