# -*- coding: utf-8 -*-
"""Download the latest llama.cpp Windows CUDA release that can run GigaChat3."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = REPO_ROOT / "llama.cpp"
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
USER_AGENT = "AI_Lenin-llama-updater"

logger = logging.getLogger("update_llama_cpp_release")


def _require_http_url(url: str) -> str:
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {url!r}")
    return url


def _http_json(url: str) -> object:
    url = _require_http_url(url)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def _http_download(url: str, dest: Path) -> None:
    url = _require_http_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=600) as response, dest.open("wb") as handle:  # nosec B310
        shutil.copyfileobj(response, handle)


def find_latest_cuda124_release() -> tuple[str, str, str]:
    """Return (tag, llama_zip_url, cudart_zip_url)."""
    releases = _http_json(RELEASES_API)
    if not isinstance(releases, list):
        raise RuntimeError("Unexpected GitHub releases payload")
    for release in releases:
        tag = str(release.get("tag_name", ""))
        assets = {str(a.get("name")): str(a.get("browser_download_url")) for a in release.get("assets", [])}
        llama_name = f"llama-{tag}-bin-win-cuda-12.4-x64.zip"
        cudart_name = "cudart-llama-bin-win-cuda-12.4-x64.zip"
        if llama_name in assets and cudart_name in assets:
            return tag, assets[llama_name], assets[cudart_name]
    raise RuntimeError("No recent release with win-cuda-12.4 assets found")


def _extract_zip(zip_path: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dest)


def install_release(*, tag: str, llama_url: str, cudart_url: str, force: bool) -> Path:
    release_root = LLAMA_DIR / f"release_{tag}"
    marker = release_root / "llama" / "llama-server.exe"
    if marker.exists() and not force:
        logger.info("Already installed: %s", release_root)
        return release_root

    staging = LLAMA_DIR / f".staging_{tag}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    llama_zip = staging / "llama.zip"
    cudart_zip = staging / "cudart.zip"
    try:
        logger.info("Downloading %s ...", llama_url)
        _http_download(llama_url, llama_zip)
        logger.info("Downloading %s ...", cudart_url)
        _http_download(cudart_url, cudart_zip)
        _extract_zip(llama_zip, staging / "llama")
        _extract_zip(cudart_zip, staging / "cudart")
        if release_root.exists():
            shutil.rmtree(release_root)
        staging.rename(release_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    if not (release_root / "llama" / "llama-server.exe").exists():
        raise RuntimeError(f"llama-server.exe missing after install into {release_root}")
    logger.info("Installed %s", release_root)
    return release_root


def write_current_pointer(release_root: Path) -> None:
    current = LLAMA_DIR / "current"
    if current.exists() or current.is_symlink():
        if current.is_dir() and not current.is_symlink():
            shutil.rmtree(current)
        else:
            current.unlink()
    # Directory junction on Windows is more reliable than symlink without admin.
    try:
        current.symlink_to(release_root, target_is_directory=True)
    except OSError:
        # Fallback: copy a tiny pointer file.
        current.mkdir(parents=True, exist_ok=True)
        (current / "RELEASE_PATH.txt").write_text(str(release_root.resolve()), encoding="utf-8")
        logger.warning("symlink unavailable; wrote %s", current / "RELEASE_PATH.txt")
    else:
        logger.info("Pointed %s -> %s", current, release_root)


def list_local_releases() -> list[Path]:
    if not LLAMA_DIR.exists():
        return []
    pattern = re.compile(r"^release_b\d+$")
    return sorted(
        [path for path in LLAMA_DIR.iterdir() if path.is_dir() and pattern.match(path.name)],
        key=lambda path: int(path.name.removeprefix("release_b")),
        reverse=True,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Update llama.cpp Windows CUDA release for GigaChat3.")
    parser.add_argument("--force", action="store_true", help="Re-download even if tag already installed.")
    parser.add_argument("--tag", default=None, help="Optional explicit tag, e.g. b10167.")
    args = parser.parse_args()

    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    if args.tag:
        tag = args.tag if args.tag.startswith("b") else f"b{args.tag}"
        base = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}"
        llama_url = f"{base}/llama-{tag}-bin-win-cuda-12.4-x64.zip"
        cudart_url = f"{base}/cudart-llama-bin-win-cuda-12.4-x64.zip"
    else:
        tag, llama_url, cudart_url = find_latest_cuda124_release()
    logger.info("Selected tag=%s", tag)
    release_root = install_release(tag=tag, llama_url=llama_url, cudart_url=cudart_url, force=bool(args.force))
    write_current_pointer(release_root=release_root)
    locals_ = list_local_releases()
    logger.info("Local releases (newest first): %s", ", ".join(path.name for path in locals_) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
