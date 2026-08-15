"""Health probes for local llama-server."""

from __future__ import annotations

from urllib.parse import urlparse
from urllib.request import urlopen


def is_llama_server_active(
    server_url: str = "http://127.0.0.1:8080", timeout_sec: float = 0.4
) -> bool:
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    base = server_url.rstrip("/")
    for path in ("/health", "/v1/models", "/"):
        probe_url = f"{base}{path}"
        if urlparse(probe_url).scheme not in {"http", "https"}:
            continue
        if _probe_llama_url(probe_url=probe_url, timeout_sec=timeout_sec):
            return True
    return False


def _probe_llama_url(*, probe_url: str, timeout_sec: float) -> bool:
    try:
        with urlopen(probe_url, timeout=timeout_sec) as response:  # nosec B310
            return 200 <= int(response.status) < 500
    except Exception:  # noqa: BLE001 - probe only
        return False
