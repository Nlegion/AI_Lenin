"""Tests for llama.cpp release runtime resolution."""

from __future__ import annotations

from pathlib import Path

from src.core.settings.llama_runtime import LlamaRuntimePaths, resolve_llama_runtime
from src.core.llm.runtime import resolve_llama_runtime as llm_resolve_llama_runtime


def test_shim_exports_same_resolve_as_llm_package() -> None:
    assert resolve_llama_runtime is llm_resolve_llama_runtime
    assert LlamaRuntimePaths is not None


def test_resolve_prefers_newest_release(tmp_path: Path) -> None:
    llama_dir = tmp_path / "llama.cpp"
    old = llama_dir / "release_b6248" / "llama"
    new = llama_dir / "release_b10167" / "llama"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "llama-server.exe").write_text("old", encoding="utf-8")
    (new / "llama-server.exe").write_text("new", encoding="utf-8")
    (llama_dir / "release_b10167" / "cudart").mkdir()
    runtime = resolve_llama_runtime(llama_dir=llama_dir)
    assert runtime.release_tag == "b10167"
    assert runtime.server_path == new / "llama-server.exe"
    assert runtime.cudart_dir == llama_dir / "release_b10167" / "cudart"


def test_resolve_current_pointer_file(tmp_path: Path) -> None:
    llama_dir = tmp_path / "llama.cpp"
    release = llama_dir / "release_b10170"
    (release / "llama").mkdir(parents=True)
    (release / "llama" / "llama-server.exe").write_text("x", encoding="utf-8")
    (release / "cudart").mkdir()
    current = llama_dir / "current"
    current.mkdir()
    (current / "RELEASE_PATH.txt").write_text(str(release.resolve()), encoding="utf-8")
    runtime = resolve_llama_runtime(llama_dir=llama_dir)
    assert runtime.release_tag == "b10170"
    assert runtime.server_path == release / "llama" / "llama-server.exe"


def test_resolve_falls_back_to_legacy(tmp_path: Path) -> None:
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "llama-server.exe").write_text("legacy", encoding="utf-8")
    runtime = resolve_llama_runtime(llama_dir=llama_dir)
    assert runtime.release_tag is None
    assert runtime.server_path == llama_dir / "llama-server.exe"
