from pathlib import Path

from src.core.settings.legacy_registry import load_legacy_registry


def test_load_legacy_registry_from_yaml():
    registry = load_legacy_registry(path=Path("config/legacy_rag_components.yaml"))
    assert registry.policy_version == "1.0.0"
    assert len(registry.components) >= 1
    assert all(component.path for component in registry.components)
