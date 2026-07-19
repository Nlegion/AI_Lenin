# Postplan Completion Summary (A-L/M)

## Scope
- Source metaplan: `C:/Users/npara/.cursor/plans/ai_lenin_rag_metaplan_bc42f41b.plan.md`
- Covered subplans: `A, B, C, D, E, F, G, H, I, J, M, K, L`
- Verification basis:
  - subplan artifacts in `.cursor/artifacts/20260718-*-subplan-*.md`
  - commits `516ab4f` through `a35225e`

## Completion Evidence Map

| Subplan | Artifact | Commit | Gate status |
|---|---|---|---|
| A | `.cursor/artifacts/20260718-1805-subplan-a-governance.md` | `516ab4f` | `alembic`: pass, `pytest`: pass |
| B | `.cursor/artifacts/20260718-1814-subplan-b-source-registry.md` | `17e3f1d` | `alembic`: pass, `pytest`: pass |
| C | `.cursor/artifacts/20260718-1829-subplan-c-ontology-worldview.md` | `a495cfe` | `alembic`: pass, `pytest`: pass |
| D | `.cursor/artifacts/20260718-1849-subplan-d-embedding-benchmark.md` | `afba4a0` | `alembic`: pass, `pytest`: pass |
| E | `.cursor/artifacts/20260718-1901-subplan-e-cleaning-rebuild.md` | `302fec1` | `alembic`: pass, `pytest`: pass |
| F | `.cursor/artifacts/20260718-1921-subplan-f-chunking-v2.md` | `ec7a36e` | `alembic`: pass, `pytest`: pass |
| G | `.cursor/artifacts/20260718-1939-subplan-g-qdrant-ingestion.md` | `c665365` | `alembic`: pass, `pytest`: pass |
| H | `.cursor/artifacts/20260718-1951-subplan-h-retrieval-sandbox.md` | `d33cbaa` | `alembic`: pass, `pytest`: pass |
| I | `.cursor/artifacts/20260718-2000-subplan-i-retrieval-prompting.md` | `44287b9` | `alembic`: pass, `pytest`: pass |
| J | `.cursor/artifacts/20260718-2009-subplan-j-quality-metrics.md` | `55558ca` | `alembic`: pass, `pytest`: pass, `ruff/bandit/vulture`: pass |
| M | `.cursor/artifacts/20260718-2018-subplan-m-newsguard.md` | `520e6f8` | `alembic`: pass, `pytest`: pass, `ruff/bandit/vulture`: pass |
| K | `.cursor/artifacts/20260718-2029-subplan-k-migration-compat.md` | `fd47691` | `alembic`: pass, `pytest`: pass, static checks: pass |
| L | `.cursor/artifacts/20260718-2036-subplan-l-codebase-refactor.md` | `a35225e` | `alembic`: pass, `pytest`: pass, static checks: pass |

## Confirmed Post-subplan Recovery Work
- Full bulk repair of `data/books/ultimate_cleaned_ontology` from `data/books/intellectual`.
- QC artifacts produced:
  - `.cursor/artifacts/cleaning/bulk_repair_ultimate_cleaned_summary.md`
  - `.cursor/artifacts/cleaning/short_docs_qc.md`

## Difficulties Register (Root Cause -> Fix -> Preventive Control)

1. Missing test/dev dependencies in clean `.venv`  
   - Root cause: runtime-only installation profile.  
   - Fix: install `pytest`, `pytest-asyncio`, and optional gate tools when needed.  
   - Preventive control: add bootstrap script with explicit dev profile (see `P4`).

2. Async test failures in pytest  
   - Root cause: plugin/config mismatch for async tests.  
   - Fix: add `pytest.ini` with `asyncio_mode=auto` and align async tests.  
   - Preventive control: enforce async test style check in release-pass.

3. Outdated integration test contracts  
   - Root cause: `NewsProcessor` API changed while tests still expected legacy methods.  
   - Fix: refactor tests to current contract.  
   - Preventive control: mark integration tests by ownership and contract version.

4. CLI module import failures (`ModuleNotFoundError: src`)  
   - Root cause: ad-hoc script execution without repo path normalization.  
   - Fix: standard `REPO_ROOT` + `sys.path.insert(0, REPO_ROOT)` bootstrap in scripts.  
   - Preventive control: unified CLI entrypoint convention.

5. Qdrant API and ID compatibility issues  
   - Root cause: point-id/string assumptions and `query_points` usage differences.  
   - Fix: deterministic int IDs (`sha256`), explicit `using` for dense/sparse queries.  
   - Preventive control: compatibility smoke test in release-pass.

6. Sparse retrieval instability after rebuilds  
   - Root cause: BM25 vocabulary/idf state not persisted across runs.  
   - Fix: save/load sparse encoder state, wire to ingestion/retrieval configs.  
   - Preventive control: hard fail when sparse state is missing or stale.

7. Retrieval quality below acceptance thresholds  
   - Root cause: weak baseline on partial index and unresolved weighting/model fit.  
   - Fix: implemented metrics/sandbox; migration and quality loops prepared.  
   - Preventive control: mandatory full-index E->J rerun after corpus repair.

8. Chroma->Qdrant migration parity too low for cutover  
   - Root cause: candidate overlap differences across providers/configs.  
   - Fix: introduced `RetrievalProvider` abstraction + A/B shadow audit.  
   - Preventive control: formal cutover gate with overlap and rollback drill.

9. Corrupted cleaned corpus artifacts (`4-line` files)  
   - Root cause: stale/incorrect prior clean outputs in `ultimate_cleaned_ontology`.  
   - Fix: full bulk rebuild from `intellectual` + QC scans.  
   - Preventive control: reproducibility manifest + periodic QC diff checks.
