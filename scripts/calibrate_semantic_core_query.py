"""Calibrate semantic_core max_query_chars on legitimate worst-case queries.

Re-run after changing retrieval_terms / max_term_tokens / max_query_chars.
Writes artifact under .cursor/artifacts/semantic_core/.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.analysis.semantic_core_config import load_semantic_core_config
from src.core.analysis.semantic_normalize import normalize_routing
from src.core.analysis.semantic_query import join_terms_with_budget

logger = logging.getLogger(__name__)


def _git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return completed.stdout.strip() or None
    except OSError:
        return None
    return None


def _worst_case_terms(config) -> list[str]:
    all_terms: list[str] = []
    for topic in config.topics:
        all_terms.extend(topic.retrieval_terms)
    all_terms.sort(key=len, reverse=True)
    selected = all_terms[: config.max_terms_per_topic]
    while len(selected) < config.max_terms_per_topic:
        selected.append(" ".join(["производительные"] * min(config.max_term_tokens, 3)))
    return selected


def _try_token_count(model_path: Path, text: str) -> tuple[int | None, str | None, int | None]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, None, None
    if not model_path.exists():
        return None, None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        model_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max, int) and model_max > 100_000:
            model_max = None
        return len(tokens), tokenizer.__class__.__name__, model_max
    except Exception as error:  # noqa: BLE001
        logger.warning("tokenizer_unavailable: %s", error)
        return None, None, None


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    config = load_semantic_core_config(path=ROOT / "config" / "semantic_core.yaml")
    terms = _worst_case_terms(config)
    body = join_terms_with_budget(terms=terms, max_chars=config.max_query_chars)
    title = normalize_routing(
        "Синтетический заголовок для калибровки семантического ядра " * 3
    )[: config.max_title_anchor_chars]
    with_title = join_terms_with_budget(
        terms=[body, title] if body else [title],
        max_chars=config.max_query_chars,
    )
    # Title must not displace terms: if with_title dropped terms, keep body.
    if not with_title.startswith(body.split(" ")[0] if body else ""):
        with_title = body

    model_path = ROOT / config.embedder_model_path
    token_count, tokenizer_class, model_max = _try_token_count(
        model_path=model_path,
        text=body,
    )
    title_token_count, _, _ = _try_token_count(model_path=model_path, text=with_title)

    if token_count is not None and model_max is not None:
        limit = model_max - config.embedder_token_margin
        if token_count > limit:
            logger.warning(
                "worst_case_tokens_exceed_limit tokens=%s limit=%s",
                token_count,
                limit,
            )

    artifact_dir = ROOT / ".cursor" / "artifacts" / "semantic_core"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    artifact = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": _git_head(ROOT),
        "dense_model": config.embedder_model_path,
        "tokenizer_class": tokenizer_class,
        "model_max_tokens": model_max or config.embedder_model_max_tokens,
        "embedder_token_margin": config.embedder_token_margin,
        "max_query_chars_config": config.max_query_chars,
        "worst_case_body": body,
        "worst_case_body_len": len(body),
        "worst_case_body_token_count": token_count,
        "worst_case_with_title_token_count": title_token_count,
        "hint_only_policy": (
            "Keep topic as hint_only when synthesis_hint is analytically useful; "
            "otherwise drop from YAML."
        ),
        "retokenize_checklist": [
            "compose per-topic queries via join_terms_with_budget",
            "tokenize with dense tokenizer",
            "compare to model_max_tokens - margin",
            "write short artifact note",
        ],
        "apply_to_legacy_rule": (
            "Keep apply_to_legacy false when author_known_rate < author_known_rate_min "
            "and human scores are unavailable."
        ),
    }
    out = artifact_dir / f"{stamp}-query_calibration.json"
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote_artifact path=%s", out)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
