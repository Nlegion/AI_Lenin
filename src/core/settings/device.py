"""Torch device resolution, GPU memory logging, and safe embedding loads."""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import urlopen

import torch

logger = logging.getLogger(__name__)

GIGA_EMBEDDING_DIM = 2048


def resolve_torch_device(
    preferred: str = "auto", *, fallback_to_cpu: bool = True
) -> str:
    normalized = (preferred or "auto").strip().lower()
    if normalized in {"cpu"}:
        return "cpu"
    if normalized in {"cuda", "gpu", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        if fallback_to_cpu or normalized == "auto":
            logger.warning(
                "torch_device_fallback",
                extra={"preferred": preferred, "reason": "cuda_unavailable"},
            )
            return "cpu"
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    raise ValueError(f"Unsupported device preference: {preferred}")


def log_gpu_memory(tag: str) -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    free, total = torch.cuda.mem_get_info()
    payload = {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "free_mb": round(free / (1024**2), 2),
        "total_mb": round(total / (1024**2), 2),
    }
    logger.info(
        "gpu_memory tag=%s allocated_mb=%.2f reserved_mb=%.2f free_mb=%.2f total_mb=%.2f",
        tag,
        payload["allocated_mb"],
        payload["reserved_mb"],
        payload["free_mb"],
        payload["total_mb"],
    )
    return payload


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


def ensure_exclusive_gpu_for_embeddings(
    *,
    preferred: str = "auto",
    fallback_to_cpu: bool = True,
    server_url: str = "http://127.0.0.1:8080",
    stop_llm_callback: Callable[[], None] | None = None,
    interactive: bool = False,
) -> str:
    device = resolve_torch_device(preferred=preferred, fallback_to_cpu=fallback_to_cpu)
    if device != "cuda":
        return device
    if not is_llama_server_active(server_url=server_url):
        return "cuda"
    if stop_llm_callback is not None and not interactive:
        logger.warning(
            "stopping_llm_for_exclusive_gpu_embeddings server_url=%s", server_url
        )
        stop_llm_callback()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "cuda"
    logger.warning(
        "embeddings_cpu_because_llm_active server_url=%s",
        server_url,
    )
    return "cpu"


def load_sentence_transformer(
    model_path: str,
    *,
    preferred_device: str = "auto",
    trust_remote_code: bool = False,
    fallback_to_cpu: bool = True,
    expected_dim: int | None = None,
    local_files_only: bool = False,
) -> Any:
    from sentence_transformers import SentenceTransformer

    device = resolve_torch_device(
        preferred=preferred_device, fallback_to_cpu=fallback_to_cpu
    )
    attempts = [device]
    if device == "cuda" and fallback_to_cpu:
        attempts.append("cpu")

    last_error: Exception | None = None
    for attempt_device in attempts:
        try:
            log_gpu_memory(tag=f"before_load:{attempt_device}")
            model = SentenceTransformer(
                model_name_or_path=model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                device=attempt_device,
            )
            vector = model.encode(["device_probe"], normalize_embeddings=True)[0]
            dim = int(len(vector))
            if expected_dim is not None and dim != expected_dim:
                raise RuntimeError(
                    f"Embedding dim mismatch for {model_path}: got {dim}, expected {expected_dim}"
                )
            logger.info(
                "sentence_transformer_loaded model=%s device=%s dim=%s",
                model_path,
                attempt_device,
                dim,
            )
            log_gpu_memory(tag=f"after_load:{attempt_device}")
            return model
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as error:
            last_error = error
            logger.exception(
                "sentence_transformer_load_failed model=%s device=%s",
                model_path,
                attempt_device,
            )
            log_gpu_memory(tag=f"load_failed:{attempt_device}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    raise RuntimeError(
        f"Failed to load SentenceTransformer from {model_path} "
        f"(trust_remote_code={trust_remote_code}): {last_error}"
    ) from last_error


def release_embedding_model(model: Any | None) -> None:
    if model is None:
        return
    try:
        del model
    except Exception:  # noqa: BLE001
        logger.exception("release_embedding_model_delete_failed")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log_gpu_memory(tag="after_release")


def hardware_report(
    *, resolved_device: str, fallback_to_cpu: bool
) -> dict[str, str | bool]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "resolved_device": resolved_device,
        "fallback_to_cpu": fallback_to_cpu,
    }
