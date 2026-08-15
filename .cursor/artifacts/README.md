# Artifacts Directory

Store generated plans, execution reports, and integration summaries here.

## Naming Convention

Use timestamped filenames:

`YYYYMMDD-HHMM-<topic>.md`

Examples:

- `20260715-1045-backend-migration-check.md`
- `20260715-1110-swarm-integration-summary.md`

## Rules

- Artifacts are project-local audit records.
- Keep content technical and concise.
- Do not store secrets.

## Subplan Workflow

1. Copy `subplan-report-template.md` into a new timestamped artifact file.
2. Run mandatory gates from repository root:
   - `python scripts/ops/run_subplan_gates.py`
3. Build a reproducibility manifest for relevant files:
   - `python scripts/ops/build_subplan_manifest.py --subplan <ID> --path <file-or-dir> --out .cursor/artifacts/manifests/<name>.json`
4. Record commands, outcomes, and manifest path in the artifact report.
