---
name: run-tests
description: Standardized test execution for AI_Lenin with migration check and failure triage
---

# Run Tests

Use this skill to run repository tests consistently and report actionable outcomes.

## Preconditions

1. Read `AGENTS.md` and relevant `.cursor/rules/*.mdc`.
2. Ensure task scope is known (full regression or targeted verification).

## Standard Commands

Run commands from repository root.

### 1) Migration sync (required before tests)

```powershell
alembic upgrade head
```

### 2) Full tests

```powershell
pytest tests -q
```

### 3) Single-file tests

```powershell
pytest tests/test_<module>.py -q
```

### 4) Filtered tests (`-k`)

```powershell
pytest tests -k "<pattern>" -q
```

## Reporting Format

Always return:

- executed commands
- pass/fail summary
- list of failed tests (if any)
- probable root cause for each failure
- recommended next fix step

## Failure Triage Heuristics

When tests fail, infer likely cause from traceback category:

- `ImportError` / `ModuleNotFoundError`: dependency or path mismatch
- DB/Alembic errors: migration not applied or schema drift
- network/API timeout: unstable external dependency, missing mocks
- assertion mismatch: behavioral regression or outdated expected values
- env var errors: missing required runtime configuration

Do not suppress failures; summarize and propose minimal corrective actions.
