# VPS deploy — beget-nl-vps (2026-08-16)

## Outcome

**Partial success.** Files, image, and Compose stack are on `/opt/ai-lenin`, but the container **cannot stay up** on this host.

Stop reason: `Exited (137)` during RAG preflight (SIGKILL / OOM). Observed `RestartCount=1` before intentional `docker compose stop` to end thrashing.

## Host facts

| Resource | Value |
|----------|-------|
| Host | `beget-nl-vps` / `91.218.141.234` |
| RAM | 1.9 GiB |
| Swap | 2 GiB (resized down from 8 GiB to free disk for model) |
| Disk | 28G root, ~98% used after deploy (~628M free) |
| Docker | 29.7.2 + Compose v5.4.0 installed |

## What was deployed

- App tree at `/opt/ai-lenin` (Dockerfile, compose, `src/` incl. `src/core/database`, `config/`, artifacts sparse+ontology)
- `.env` + `ai_lenin.db` (file bind-mount)
- `models/Giga-Embeddings-instruct/` (~13G, no `.cache`)
- `database/qdrant_local/` (~2.3G)
- Image `ai-lenin-app:latest` built (~2.5G)

## Startup signals that worked

- Telegram env present
- `Generation provider=deepseek spawn_local=False model=deepseek-v4-flash`
- Entered `Remote LLM mode: running RAG preflight`

## Signals that never appeared

- `rag_preflight_ok`
- `sentence_transformer_loaded` / `Retrieval provider initialized`
- Migrations / processing loops

## Local publisher

- `docker compose down` completed; no local `ai-lenin` container left.

## Fix required before re-`up`

Upgrade Beget VPS to roughly **≥16 GB RAM** and **≥50 GB disk** (plan baseline), then:

```bash
cd /opt/ai-lenin
# enlarge swap if RAM still < ~16G
docker compose up -d
docker compose logs -f app
```

Do not run two publishers (local + VPS) against the same Telegram channel.
