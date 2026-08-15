# AI_Lenin — Agent Guide

Project-level guide for Cursor Swarm agents. Read this file before starting any task.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3 (asyncio) |
| Runtime entry | `src/main.py` — news fetch → LLM analysis → Telegram publish |
| LLM inference | llama.cpp (`llama-server.exe`), `llama_cpp_python`, local GGUF models |
| ML / NLP | PyTorch, transformers, sentence-transformers, spacy, pymorphy3 |
| RAG | Qdrant hybrid + embeddings |
| Database | SQLite via SQLAlchemy async (`aiosqlite`) |
| Migrations | Alembic (`alembic.ini`, `src/core/database/migrations/`) |
| HTTP clients | httpx, aiohttp |
| Logging | stdlib logging + structlog |
| Config | `python-dotenv`, `src/core/settings/config.py` |
| Tests | Integration scripts in `tests/` (async, runnable via `python` or `pytest`) |
| Training | RAG ontology / worldview stages in `training/` (LLM LoRA abandoned) |

**No web frontend.** User-facing surface is Telegram only.

## Project Structure

```
AI_Lenin/
├── src/
│   ├── main.py                 # Application entry point
│   ├── core/
│   │   ├── processor.py        # News processing orchestrator
│   │   ├── news_item_pipeline.py
│   │   ├── lenin_analyzer.py   # LLM analysis via llama.cpp HTTP
│   │   ├── publisher.py        # Telegram publishing
│   │   ├── llama_server.py     # Local model server wrapper
│   │   ├── generation/         # pipeline, prompts, postprocess_clean
│   │   ├── dialectics/         # reasoning engine
│   │   ├── analysis/           # EvidenceBrief, semantic core
│   │   ├── safety/             # PreRagCensor, NewsGuard, gates
│   │   ├── settings/           # config.py, log.py
│   │   ├── database/           # SQLAlchemy models, repos, migrations
│   │   ├── retrieval/          # Qdrant retrieval providers
│   │   ├── adapters/telegram/  # Telegram client/service
│   │   └── utils/
│   └── modules/news_system/    # fetcher, classifier
├── tests/                      # Integration test scripts
├── training/                   # RAG ontology / worldview (not LLM fine-tune)
├── scripts/                    # Domain CLIs + root shims (see scripts/README.md)
│   ├── quality/ retrieval/ safety/ dialectics/ corpus/ ops/ lib/
├── docs/                       # Technical SoT (index: docs/README.md)
├── config/                     # YAML runtime SoT
├── alembic.ini
├── requirements.txt
└── .env                        # Secrets (gitignored)
```

**Gitignored runtime assets:** `models/`, `database/`, `data/`, `llama.cpp/`, `model_cache/`, `.venv/`

## Coding Conventions

- Follow PEP 8; prefer type hints on new and modified functions.
- Use **keyword arguments** for calls with 3+ parameters (see `.cursor/rules/keyword-arguments.mdc`).
- Keep files under **200 lines** when creating or significantly expanding code (see `.cursor/rules/file-size-splitting.mdc`).
- Put reusable policy literals in `src/core/settings/` — do not scatter magic numbers across modules.
- Use structured logging; never bare `except:` or silent `pass` in exception handlers (see `.cursor/rules/logging.mdc`).
- Async code: respect `asyncio` patterns; on Windows use `WindowsSelectorEventLoopPolicy` (already in `main.py`).
- Database changes require Alembic migrations in `src/core/database/migrations/versions/`.
- Do not commit secrets, model weights, or local DB files.

## Key Commands

```powershell
# Setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run application
python src/main.py

# Database migrations
alembic upgrade head

# Tests (integration scripts)
python tests/test_news_fetcher.py
python tests/test_news_processor.py
python tests/test_telegram_publisher.py

# Pytest (when tests are pytest-compatible)
pytest tests -q
pytest tests/test_news_fetcher.py -q
pytest tests -k "fetcher" -q

# Quality / dry-run / release
# Canonical paths live under scripts/<domain>/; root scripts/*.py are thin shims.
python scripts/quality/run_local_rag_dryrun.py --fixture economy --verbose
python scripts/retrieval/evaluate_rag_quality.py
python scripts/safety/evaluate_news_guard.py
python scripts/quality/evaluate_anti_cliche.py
python scripts/quality/run_live_news_qa_batch.py --help
python scripts/quality/run_live_news_qa_24h.py --help
python scripts/safety/calibrate_combat_gate.py
python scripts/safety/rollback_gate_config.py snapshot   # or: restore
python scripts/ops/release_pass.py --help
python scripts/quality/collect_anti_cliche_label_batch.py
python scripts/ops/update_llama_cpp_release.py
python scripts/quality/run_quality_qa_batch.py --guard-check-only
python scripts/quality/run_quality_qa_batch.py --limit 50 --persona-model base_strong --start-server --allow-legacy-fallback
python scripts/dialectics/calibrate_semantic_core_query.py
python scripts/dialectics/evaluate_semantic_core.py
python scripts/dialectics/run_dialectical_reasoning_dryrun.py --fixture neftegaz

# Version bump
python scripts/ops/version_update.py patch   # or major / minor

# RAG ontology rebuild (not LLM fine-tune)
python scripts/corpus/build_ontology_worldview.py --help
python scripts/retrieval/build_qdrant_index.py --help
```

**Required env vars:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHANNEL_ID`, `TELEGRAM_ADMIN_ID`
**Optional:** `NEWSAPI_KEY`, `HUGGINGFACE_TOKEN`

## Language & Communication

- All agent prompts, reports, code comments, and technical documentation: **English only**.
- User-facing chat explanations may be in Russian when the user writes in Russian.
- Never mix languages inside technical artifacts (rules, agent profiles, skills, code comments).

## Shared Context Bootstrap

Before starting any subtask, every agent must:

1. Read this file (`AGENTS.md`).
2. Read `context.md` if it exists in the project root.
3. Skim relevant `.cursor/rules/*.mdc` for the files being touched.

## Safety & Destructive Commands

The following require **explicit confirmation** from the architect agent or the user:

- `rm -rf`, `del /s`, recursive file deletion
- Destructive SQL: `DROP TABLE`, `DELETE FROM` without WHERE, schema drops
- `git reset --hard`, force push, branch deletion
- Killing running processes that hold DB locks
- Modifying `.env`, production DB files, or model weights

Agents must not modify global Cursor settings. All configuration stays inside this repository.

## Model Versioning & Runtime Compatibility

Recommended agent models (project guidance, not enforced):

| Agent | Recommended model tier |
|-------|----------------------|
| architect | High-reasoning model (planning, verification, integration) |
| backend-dev | Fast coding model (implementation, tests, migrations) |
| frontend-dev | Fast coding model (contracts, mocks, SDK stubs) |

Runtime baseline:

- Python 3.10+ recommended
- Dependencies pinned in `requirements.txt`
- Before adding new deps, check compatibility with PyTorch/CUDA stack
- Local LLM models are large gitignored assets — never commit them

## Rules Baseline

Project rules in `.cursor/rules/*.mdc` are adapted from an external baseline (`P:\rules`) with project-specific paths and commands. Active rules:

| Rule file | Scope |
|-----------|-------|
| `safety.mdc` | Security and destructive command policy |
| `logging.mdc` | Structured logging and exception handling |
| `testing.mdc` | Test standards and structure |
| `linter.mdc` | Pre-commit lint checklist |
| `release-quality-gates.mdc` | Pre-release verification |
| `keyword-arguments.mdc` | Named argument preference |
| `file-size-splitting.mdc` | File size and decomposition |
| `application-constants.mdc` | Centralized constants |
| `log-incident-registry.mdc` | Optional incident triage workflow |
| `swarm-collaboration.mdc` | Swarm output contract and handoff rules |

Dialectical R1–R3 orchestration SoT: `docs/dialectical_orchestration_r1_r3.md` (`dialectical_orchestration.enabled: true` in `config/retrieval_pipeline.yaml`).
Dialectical reasoning engine: `docs/dialectical_reasoning_engine.md` (`dialectical_reasoning.mode`, default `orchestration_single_pass`).
Semantic core (modern→Lenin abstract topics): `docs/semantic_core.md` (`config/semantic_core.yaml`, `enabled: true`). Answer post-processing / public scrub: `docs/answer_postprocess.md` (`postprocess_clean_mode: live`). Crisis recovery / anti-cliché priorities: `docs/priority_crisis_recovery_and_hardening.md`. Human eval loop: `docs/human_eval_checklist.md`. Docs index: `docs/README.md`. Unified release thresholds: `config/release_gates.yaml`.

## Agent Collaboration Rules

### Roles

| Agent | Responsibility |
|-------|---------------|
| **architect** | Decompose tasks, assign sub-agents, verify results, integrate final output |
| **backend-dev** | `src/`, `tests/`, migrations, ML runtime integration |
| **frontend-dev** | Optional — UI, API contracts, client SDK, mock clients only |

### Allowed Tools By Role

| Agent | Allowed tools (policy-level) |
|-------|------------------------------|
| **architect** | `list_files`, `semantic_search`, read-only inspection, orchestration/handoff tools |
| **backend-dev** | `read_file`, `write_file`, `grep_search`, `run_terminal_cmd` (safe project-scoped commands only) |
| **frontend-dev** | contract/design tools by architect delegation; implementation tools only for explicit UI/API/SDK scope |

### Handoff Protocol

1. Architect creates a plan with acceptance criteria before delegating.
2. Sub-agents implement within their scope and return structured output.
3. Architect runs **Verification** against acceptance criteria.
4. Architect performs a short pre-merge code review (style, types, tests).
5. Final integrated result is delivered to the user.

### Output Contract

Every agent response must include these sections:

```
## Plan
(brief steps)

## Assumptions
(what was assumed if info was missing)

## Questions
(open items; max 1 clarification request per subtask via architect)

## Result
(what was done, files changed, test status)
```

### Clarification Policy

- If a subtask is ambiguous, the sub-agent may request **one** clarification through the architect.
- Do not block on multiple round-trips; proceed with stated assumptions if no answer.

### Artifacts

Save plans, sub-agent reports, and integration summaries to:

```
.cursor/artifacts/YYYYMMDD-HHMM-<topic>.md
```

Use artifacts for audit trail and context reuse across sessions.

### frontend-dev Activation

Invoke **frontend-dev** only when the task explicitly involves:

- UI components or web frontend
- API contract design or OpenAPI specs
- Client SDK generation or mock clients

Do **not** invoke frontend-dev for pure backend, ML, or database tasks.
