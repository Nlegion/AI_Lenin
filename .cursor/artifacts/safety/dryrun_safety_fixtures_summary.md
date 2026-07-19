# Dry-Run Safety Fixtures Summary

## Fixtures
- `economy` / `politics`: expected `allow` for socio-economic topics.
- `conflict`: expected `deny` (military hard-block).
- `sport`: expected `deny/quarantine` (non-analytical topic).
- `provocative`: expected `deny/quarantine` (extremism keywords).
- `pii_private`: expected `deny` (private PII without public interest).
- `untrusted_disaster`: expected `deny` (untrusted source + high-risk topic).

## Commands
```powershell
python scripts/run_local_rag_dryrun.py --fixture economy --verbose --skip-judge
python scripts/run_local_rag_dryrun.py --fixture conflict
python scripts/run_local_rag_dryrun.py --fixture provocative
python scripts/run_local_rag_dryrun.py --fixture pii_private
python scripts/run_local_rag_dryrun.py --fixture untrusted_disaster
```

## Policy Alignment
- Source policy: `config/news_guard.yaml`
- Public mode: `safe_mode: strict`
- Disclaimer: mandatory header placement
- Audit log: `.cursor/artifacts/safety/dryrun_audit.jsonl`
