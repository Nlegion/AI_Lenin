# Censorship Pipeline Wiring

This document tracks runtime call-points for censorship, before and after pre-RAG refactor.

## Current Runtime Order

1. `NewsProcessor.process_single_news()` calls `PreRagCensor.evaluate()` before any classifier/RAG/LLM work.
2. `PreRagCensor` wraps legacy `SafetyGate` / `NewsGuard` and normalizes decisions into:
   - `allow`
   - `hard_block`
   - `review`
   - `skip`
3. `NewsProcessor` stops processing when decision is not `allow`.
4. Only `allow` branch reaches `LeninAnalyzer.generate_analysis()` (RAG + LLM pipeline).
5. Post-generation output guard remains in `NewsGuard.guard_output()` and `post_generate_gates`.

## Key Integration Files

- `src/core/processor.py` — enforcement point and stop-before-RAG branching.
- `src/core/safety/pre_rag_censor.py` — standalone pre-RAG censorship module.
- `src/core/safety/safety_gate.py` — legacy shadow/enforcement source for migration.
- `src/core/safety/news_guard.py` — legacy input/output safety rules.
- `src/core/safety/batch_metrics.py` — routing and drift metrics (`hard_block/review/skip/allow`).
- `src/core/safety/safety_gate_metrics.py` — shadow parity and gate share observability.

## Legacy Mapping (Transition)

- `deny -> hard_block`
- `quarantine -> review`
- `skip -> skip`
- `allow -> allow`

This mapping is applied in pre-RAG censor contract and consumed by metrics.
