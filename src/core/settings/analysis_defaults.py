"""Centralized defaults for analysis runtime behavior."""

from __future__ import annotations

from typing import Any


LLAMA_SERVER_URL = "http://127.0.0.1:8080"
ANALYSIS_CACHE_LIMIT = 1000

REFUSAL_PHRASES: tuple[str, ...] = (
    "не входит в круг моих исследований",
    "данная тема не подлежит анализу",
    "отказываюсь от анализа",
)


def default_generation_params() -> dict[str, Any]:
    """Return generation params for llama.cpp completion endpoint."""
    return {
        "temperature": 0.4,
        "top_p": 0.8,
        "top_k": 40,
        "repeat_penalty": 1.5,
        "typical_p": 0.9,
        "stop": ["<|eot_id|>", "\n\n", "###"],
        "n_predict": 300,
        "mirostat": 2,
        "mirostat_tau": 3.0,
        "mirostat_eta": 0.1,
        "threads": 4,
    }
