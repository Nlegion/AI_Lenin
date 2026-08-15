# Scripts layout

Domain folders under `scripts/`. Root `scripts/*.py` files are thin compatibility shims for older CLI paths.

| Directory | Contents |
|-----------|----------|
| `lib/` | Shared helpers (`_quality_qa_*`, `_live_news_qa_*`) |
| `safety/` | Censor, NewsGuard, combat calib, gate rollback |
| `quality/` | Live/QA batches, dry-run, anti-cliché metrics |
| `retrieval/` | Qdrant, embeddings, RAG eval, AB sandbox |
| `dialectics/` | Dialectical dry-runs, semantic core |
| `corpus/` | Cleaning, chunking, ontology, source registry |
| `ops/` | Release pass, subplan gates, version, bootstrap |

Prefer:

```powershell
python scripts/quality/run_local_rag_dryrun.py --fixture economy --verbose
```

Legacy still works via shim:

```powershell
python scripts/run_local_rag_dryrun.py --fixture economy --verbose
```

Python imports use the package path, e.g. `from scripts.lib._quality_qa_io import QaItem`.
