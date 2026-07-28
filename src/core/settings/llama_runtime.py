"""Helpers to resolve llama.cpp Windows CUDA release packs for GigaChat3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_RELEASE_DIR_RE = re.compile(r"^release_(b\d+)$")


@dataclass(frozen=True)
class LlamaRuntimePaths:
    server_path: Path
    runtime_dir: Path
    cudart_dir: Path | None
    release_tag: str | None


def _release_sort_key(path: Path) -> int:
    match = _RELEASE_DIR_RE.match(path.name)
    if not match:
        return -1
    return int(match.group(1).removeprefix("b"))


def _paths_from_release_root(release_root: Path) -> LlamaRuntimePaths | None:
    root = release_root
    try:
        if root.is_symlink() or root.exists():
            root = root.resolve()
    except OSError:
        root = release_root
    server = root / "llama" / "llama-server.exe"
    if not server.exists():
        server = root / "llama-server.exe"
        if not server.exists():
            return None
        runtime_dir = root
        cudart = root.parent / "cudart"
        if root.name == "llama":
            cudart = root.parent / "cudart"
            tag_match = _RELEASE_DIR_RE.match(root.parent.name)
        else:
            tag_match = _RELEASE_DIR_RE.match(root.name)
        return LlamaRuntimePaths(
            server_path=server,
            runtime_dir=runtime_dir,
            cudart_dir=cudart if cudart.exists() else None,
            release_tag=tag_match.group(1) if tag_match else None,
        )
    cudart = root / "cudart"
    tag_match = _RELEASE_DIR_RE.match(root.name)
    return LlamaRuntimePaths(
        server_path=server,
        runtime_dir=root / "llama",
        cudart_dir=cudart if cudart.exists() else None,
        release_tag=tag_match.group(1) if tag_match else None,
    )


def resolve_llama_runtime(llama_dir: Path) -> LlamaRuntimePaths:
    """Prefer llama.cpp/current, else newest release_b*, else legacy llama-server.exe."""
    current = llama_dir / "current"
    if current.exists():
        pointer = current / "RELEASE_PATH.txt"
        if pointer.exists():
            pointed = Path(pointer.read_text(encoding="utf-8").strip())
            resolved = _paths_from_release_root(pointed)
            if resolved is not None:
                return resolved
        resolved = _paths_from_release_root(current)
        if resolved is not None:
            return resolved

    release_dirs = sorted(
        [path for path in llama_dir.glob("release_b*") if path.is_dir()],
        key=_release_sort_key,
        reverse=True,
    )
    for release_root in release_dirs:
        resolved = _paths_from_release_root(release_root)
        if resolved is not None:
            return resolved

    legacy = llama_dir / "llama-server.exe"
    return LlamaRuntimePaths(
        server_path=legacy,
        runtime_dir=llama_dir,
        cudart_dir=None,
        release_tag=None,
    )
