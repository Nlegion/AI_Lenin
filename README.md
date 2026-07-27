# AI_Lenin

Local news analysis pipeline (Telegram) with Qdrant hybrid retrieval and optional dialectical R1–R3 orchestration.

See `AGENTS.md` for agent conventions and `docs/dialectical_orchestration_r1_r3.md` for the R1–R3 SoT.

## Настройка Qdrant

Requires **qdrant-client >= 1.7.0** (repo environment currently uses 1.18.x).

Dialectical filtered retrieve (`retrieve_by_stance`) needs a payload index on `stance_type`.

Run **once per environment database** (dev/staging/prod path) — not on every app instance boot:

```powershell
.\.venv\Scripts\python.exe scripts/ensure_qdrant_stance_index.py
```

The script is idempotent (check-then-create) and waits until the index is ready on **server** Qdrant. Embedded/local Qdrant ignores payload indexes (filters still work via scan); prefer server Qdrant for filter performance. Hot path never auto-creates the index.

## Dialectical orchestration

Config block `dialectical_orchestration` in `config/retrieval_pipeline.yaml` (default `enabled: false`).

When enabled, analysis uses structured EvidenceBrief slots R1/R2/R3 instead of a flat RAG merge.
