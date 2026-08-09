# ADR: Censorship Dedup And External Datasets

## Status
Accepted

## Context
- Duplicate news items produced inconsistent moderation outcomes.
- Keyword-only routing misses some high-risk war/SVO content.
- External datasets must be open-license only.

## Decision
- Use canonical content hash for dedup with fixed normalization order.
- Persist moderation decisions in DB keyed by `(content_hash, config_version_hash)`.
- Compute `config_version_hash` from:
  - `normalizer_version_hash`
  - `policy_version_hash`
  - `model_version_hash` (or `l2_off` when disabled)
- Introduce fallback policy:
  - L2 failure -> L1 decision
  - L1 failure -> review (or hard_block for high war signal)
- External datasets are managed through a versioned manifest and automated license audit.

## Consequences
- Deterministic duplicate decisions across restarts.
- Safer rollout via canary + shadow agreement checks.
- Clear legal guardrails with automated license-change blocking.
