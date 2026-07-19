#!/usr/bin/env python3
"""Smoke-load local Giga-Embeddings with device resolver and dim checks."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
import os
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.embeddings.benchmark import cosine_similarity  # noqa: E402
from src.core.settings.device import (  # noqa: E402
    GIGA_EMBEDDING_DIM,
    ensure_exclusive_gpu_for_embeddings,
    hardware_report,
    load_sentence_transformer,
    release_embedding_model,
)

DEFAULT_LOCAL_MODEL = "models/Giga-Embeddings-instruct"
PROBE_TEXT = "инфляция и классовая борьба"


def _resolve_model_path(model: str) -> str:
    path = Path(model)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        return str(path.resolve())
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test local Giga-Embeddings-instruct.")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--online", action="store_true")
    parser.add_argument(
        "--out-md",
        default=".cursor/artifacts/embeddings/embedding_selection_gigachat.md",
    )
    parser.add_argument(
        "--out-json",
        default=".cursor/artifacts/embeddings/giga_smoke_reference.json",
    )
    args = parser.parse_args()

    offline = not args.online
    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    model_path = _resolve_model_path(args.model)
    device = ensure_exclusive_gpu_for_embeddings(
        preferred=args.device,
        fallback_to_cpu=True,
        interactive=True,
    )
    hw = hardware_report(resolved_device=device, fallback_to_cpu=True)

    out_path = (REPO_ROOT / args.out_md).resolve()
    out_json = (REPO_ROOT / args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    load_ok = False
    error_text = ""
    embedding_dim = 0
    reference: list[float] = []
    model = None
    try:
        model = load_sentence_transformer(
            model_path=model_path,
            preferred_device=device,
            trust_remote_code=True,
            fallback_to_cpu=True,
            expected_dim=GIGA_EMBEDDING_DIM,
            local_files_only=offline and Path(model_path).exists(),
        )
        vector = model.encode([PROBE_TEXT], normalize_embeddings=True)[0]
        reference = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        embedding_dim = len(reference)
        load_ok = embedding_dim == GIGA_EMBEDDING_DIM
    except Exception as error:  # noqa: BLE001
        error_text = f"{error}\n{traceback.format_exc()}"
    finally:
        release_embedding_model(model)

    lines = [
        "# Embedding Selection: Giga-Embeddings-instruct",
        "",
        f"- Generated at (UTC): `{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- Production local_path: `{model_path}`",
        "- Provenance hf_id: `ai-sage/Giga-Embeddings-instruct`",
        f"- Torch: `{hw['torch_version']}` GPU=`{hw['gpu_name']}`",
        f"- Resolved device: `{hw['resolved_device']}` fallback_to_cpu=`{hw['fallback_to_cpu']}`",
        f"- Offline load: `{str(offline).lower()}`",
        "",
        "## Smoke Load",
        f"- status: `{'ok' if load_ok else 'failed'}`",
        f"- embedding_dim: `{embedding_dim}` (expected `{GIGA_EMBEDDING_DIM}`)",
        "",
        "## Decision",
        "- production_target_configured: `true`",
        f"- smoke_load_ok: `{str(load_ok).lower()}`",
        "- runtime_collection: `philosophy_ontology_giga_v1`",
        "- obsolete_collection: `philosophy_ontology_v2` (MiniLM; keep until explicit cleanup)",
        "",
    ]
    if error_text:
        lines.extend(["## Error Detail", "```", error_text[:4000], "```", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")

    if load_ok:
        out_json.write_text(
            json.dumps(
                {
                    "probe_text": PROBE_TEXT,
                    "embedding_dim": embedding_dim,
                    "reference_vector": reference,
                    "hardware": hw,
                    "self_cosine": cosine_similarity(reference, reference),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"smoke_load_ok={load_ok}")
    print(f"device={device}")
    print(f"artifact={out_path}")
    return 0 if load_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
