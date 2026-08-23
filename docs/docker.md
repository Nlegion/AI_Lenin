# Docker (VPS RAG replica)

Deploy AI_Lenin on a Linux VPS with a **full copy of the local RAG index** (embedded Qdrant + Giga-Embeddings). Local Windows GPU + GigaChat3/`llama-server` stay on the workstation. Generation uses either the explicit DeepSeek provider (`LLM_PROVIDER=deepseek`) or a generic OpenAI-compatible remote URL (`LLM_SPAWN_LOCAL=false`, `LLM_PROVIDER=llama`).

## What runs in the container

| Component | In image / volume |
|-----------|-------------------|
| App (`python src/main.py`) | Image |
| `config/` YAML + censor terms | Image |
| Giga-Embeddings-instruct | Bind mount (complete HF repo) |
| `database/qdrant_local` | Bind mount |
| Sparse encoder + ontology TSV | Under `.cursor/artifacts` bind mount |
| SQLite `ai_lenin.db` | **Pre-created host file** bind mount |
| Runtime JSONL under `.cursor/artifacts` | Same writable bind mount |
| `llama.cpp` / GGUF | **Not shipped** |

## Host prerequisites

- Docker Engine + Compose on Linux (or Docker Desktop with Linux containers)
- Outbound HTTPS (TASS RSS, Telegram, remote LLM)
- Roughly **8–16 GB RAM** for CPU embeddings; multi-GB disk for Qdrant + model
- Same `qdrant-client` major/minor as the snapshot producer (`1.18.0`)

## Build the RAG snapshot (on the Windows workstation)

Stop the local app first (embedded Qdrant locks files).

```powershell
# Verify collection + embedding repo completeness
python scripts/ops/pack_rag_snapshot.py

# Optional tarball
python scripts/ops/pack_rag_snapshot.py --output .cursor/artifacts/rag_snapshot.tar.gz
```

Required assets (see [`config/retrieval_pipeline.yaml`](../config/retrieval_pipeline.yaml)):

- `models/Giga-Embeddings-instruct/` — full HF repo (`config.json`, `modules.json`, `config_sentence_transformers.json`, weights, tokenizer, custom code for `trust_remote_code`)
- `database/qdrant_local/` — collection `philosophy_ontology_giga_v1` with `points_count > 0` and dense dim 2048
- `.cursor/artifacts/qdrant/sparse_encoder_state_giga_v1.json`
- `.cursor/artifacts/ontology/ontology_tags.tsv`

Do **not** copy `models/gigachat3/*.gguf`, `llama.cpp/`, or `data/books/`.

## SQLite file contract

[`src/core/database/db_core.py`](../src/core/database/db_core.py) hardcodes `/app/ai_lenin.db`. Compose bind-mounts a **host file** at that path.

```powershell
# Empty DB on VPS (migrations run inside main.py)
New-Item -ItemType File -Path ai_lenin.db -Force
# or: copy your existing ai_lenin.db
```

Never attach a Docker **named volume** to `/app/ai_lenin.db` (volumes are directories; SQLite will fail).

## Configure env

```powershell
copy .env.example .env
# fill TELEGRAM_* and either DEEPSEEK_API_KEY or LLM_API_KEY / LLM_MODEL_NAME
```

| Variable | VPS value |
|----------|-----------|
| `LLM_SPAWN_LOCAL` | `false` |
| `LLM_PROVIDER` | `deepseek` (recommended) or `llama` for generic OpenAI-compatible hosts |
| `GENERATION_SERVER_URL` | Optional for DeepSeek (defaults to `https://api.deepseek.com`). For generic remote: base URL **without** trailing `/v1` |
| `DEEPSEEK_API_KEY` | Bearer for DeepSeek when `LLM_API_KEY` is unset |
| `LLM_API_KEY` | Generic Bearer (wins over `DEEPSEEK_API_KEY`) |
| `LLM_MODEL_NAME` | DeepSeek default `deepseek-v4-flash`; required for generic remote |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHANNEL_ID` / `TELEGRAM_ADMIN_ID` | Required |
| `TELEGRAM_PROXY_REQUIRED` | `true` on VPS when Bot API must exit via host SOCKS (Beget VLESS) |
| `TELEGRAM_PROXY_URL` | e.g. `socks5://user:pass@host.docker.internal:1080` — **only** TelegramClient; never set global `HTTP(S)_PROXY` (TASS/DeepSeek must stay direct) |

`GENERATION_SERVER_URL` is normalized (strip `/` and trailing `/v1`). DeepSeek uses `POST /chat/completions`; the generic llama remote path keeps `POST /v1/chat/completions`. See [`docs/llm_client.md`](llm_client.md).

Compose sets `extra_hosts: host.docker.internal:host-gateway` so the container reaches the host Xray SOCKS listener.

## Compose up

```bash
docker compose build
docker compose up -d
docker compose logs -f app
```

Entrypoint runs **`python src/main.py` only**. Alembic runs inside `main.py` via `apply_migrations()` — do not add a second `alembic upgrade` in the entrypoint.

## First-boot checks

1. Telegram env present (process exits otherwise).
2. With `LLM_SPAWN_LOCAL=false`, RAG preflight runs **before** `NewsProcessor` and exits non-zero if assets/collection are bad (no silent `retrieval_provider=None`).
3. Logs show remote LLM mode / local llama-server skipped.
4. No `llama-server` process inside the container.

## Local Windows (unchanged)

Default `LLM_SPAWN_LOCAL=true` (or unset) keeps spawning local `llama-server` from `llama.cpp/` and GGUF under `models/gigachat3/`.

## TASS RSS from the container

`NewsFetcher` sends a browser `User-Agent` and `Accept: application/rss+xml,...` (`Settings.TASS_RSS_USER_AGENT`). StormWall returns HTTP 403 for feedparser’s default UA. HTTP status ≥400 or a bozo parse with no entries logs `tass_rss_http_error` / `tass_rss_parse_failed` and returns no items. Keep TASS (and DeepSeek) off any global `HTTP(S)_PROXY`; SOCKS is Telegram-only.

## Out of scope

- CUDA / `llama-server` in the VPS image
- Qdrant as a network service
- Rebuilding the corpus index inside the container

DeepSeek chat payloads, quote allowlists, and triad repair live in `src/core/generation/` (`deepseek_prompts.py`, `deepseek_quote_validate.py`) and [`docs/llm_client.md`](llm_client.md) — not in the Docker image.
