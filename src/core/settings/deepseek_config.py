"""DeepSeek-specific generation config helpers."""

from __future__ import annotations

import os
from urllib.parse import urlparse

DEEPSEEK_DEFAULT_SERVER_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
LOCAL_DEFAULT_SERVER_URL = "http://127.0.0.1:8080"


def is_local_or_insecure_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https":
        return True
    if host in {"127.0.0.1", "localhost", "::1"} or host.startswith("192.168."):
        return True
    return False


def deepseek_allow_insecure_url_from_env() -> bool:
    raw = os.getenv("LLM_DEEPSEEK_ALLOW_INSECURE_URL", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def validate_deepseek_payload(*, payload: dict, normalize_server_url) -> None:
    if payload.get("spawn_local"):
        raise ValueError(
            "LLM_PROVIDER=deepseek requires LLM_SPAWN_LOCAL=false "
            "(remote API only; do not spawn local llama-server)"
        )
    if not (payload.get("api_key") or "").strip():
        raise ValueError("DeepSeek provider requires LLM_API_KEY or DEEPSEEK_API_KEY")
    persona = payload["persona_model"]
    backends = payload.get("backends") or {}
    active = backends.get(persona) or {}
    model_name = str(active.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("DeepSeek provider requires a non-empty model_name")

    server_url = normalize_server_url(str(payload.get("server_url") or ""))
    if (
        is_local_or_insecure_url(server_url)
        and not deepseek_allow_insecure_url_from_env()
    ):
        raise ValueError(
            "DeepSeek provider requires an HTTPS remote URL "
            f"(got {server_url!r}); set GENERATION_SERVER_URL or "
            "LLM_DEEPSEEK_ALLOW_INSECURE_URL=true for a trusted proxy"
        )
    payload["server_url"] = server_url
