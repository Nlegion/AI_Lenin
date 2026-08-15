# Third-party components and licenses

This file lists **key** open-source libraries, models, and external services used by AI_Lenin. It is **not** a complete SPDX inventory of every transitive dependency.

Pinned Python package versions: [`requirements.txt`](requirements.txt).  
Eval / censorship datasets: [`NOTICE`](NOTICE) and [`config/external_dataset_sources.yaml`](config/external_dataset_sources.yaml).

License strings below are best-effort attributions for operators; always verify upstream LICENSE files for redistribution.

## Runtime libraries (hot path)

| Component | License (summary) | Notes |
|-----------|-------------------|--------|
| qdrant-client | Apache-2.0 | Vector DB client |
| sentence-transformers | Apache-2.0 | Embeddings loaders |
| transformers | Apache-2.0 | HF model stack |
| datasets | Apache-2.0 | Dataset utilities |
| llama.cpp (binary) | MIT | Local `llama-server`; not the Python wheel |
| llama_cpp_python | MIT | Pinned in requirements; **unused on hot path** |
| PyTorch | BSD-style | `torch` / torchvision / torchaudio |
| NumPy | BSD | |
| pandas | BSD | |
| scikit-learn | BSD | |
| SciPy | BSD | |
| spaCy | MIT | |
| pymorphy3 | MIT | |
| SQLAlchemy | MIT | |
| Alembic | MIT | |
| aiosqlite | MIT | |
| pydantic | MIT | |
| PyYAML | MIT | |
| httpx | BSD-3-Clause | |
| aiohttp | Apache-2.0 AND MIT | Dual-licensed; do not simplify to MIT-only |
| structlog | MIT OR Apache-2.0 | |
| python-dotenv | BSD-3-Clause | |
| feedparser | BSD-2-Clause | TASS RSS parse |
| faiss-cpu | MIT (upstream) | **Training leftover only**; not hot path |

## Models

| Component | License | Notes |
|-----------|---------|--------|
| ai-sage/GigaChat3-10B-A1.8B | MIT (Copyright 2025 Salute Developers) | Default local generator; local GGUF is a quantized copy, not the full HF repo |
| ai-sage/Giga-Embeddings-instruct | MIT | Dense embeddings; load from local path offline |

Identities: [`config/model_registry.yaml`](config/model_registry.yaml).

## External services (not redistributed software)

| Service | Role | Notes |
|---------|------|--------|
| DeepSeek API | Optional remote generation (`LLM_PROVIDER=deepseek`) | Proprietary ToS; operator supplies `DEEPSEEK_API_KEY` / `LLM_API_KEY`; project does not ship keys |
| TASS RSS | Live news input | `https://tass.ru/rss/v2.xml` in `src/modules/news_system/fetcher.py`; comply with source terms |
| Telegram Bot API | Publish path | Operator Bot Token; Telegram ToS |

## Eval / censorship datasets

Do not duplicate the dataset table here. See [NOTICE](NOTICE).

At the time of writing, [`config/external_dataset_sources.yaml`](config/external_dataset_sources.yaml) marks:

- `dzen_russian_articles` — `apache-2.0`, `allowed_use: yes`
- `lenta_kaggle`, `rus_news_classifier`, `ru_ethno_hate` — `unknown` / `pending_review`

Only open-license, allowed sources should be used for eval; license-change audit: `scripts/corpus/audit_external_dataset_licenses.py`.
