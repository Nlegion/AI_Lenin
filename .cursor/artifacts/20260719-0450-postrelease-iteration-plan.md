# Post-release Iteration Plan

## Goal
Drive measurable improvement in retrieval relevance and ideology consistency while keeping migration safety gates explicit.

## Iteration Loop
1. **Evaluate**
   - Run sandbox + quality metrics on latest index.
   - Record: Recall@5, core_self_ratio, ideology_consistency, parity.
2. **Adapt embeddings**
   - Attempt corpus-adaptive embedding strategy:
     - lightweight fine-tuning if memory permits, or
     - model substitution experiments with strict benchmark capture.
3. **Re-measure**
   - Re-run the same evaluation set and compare deltas.
4. **Gate decision**
   - Continue if objective metric improvement is observed.
   - Stop if no improvement in 2 consecutive iterations.

## Expert-in-the-loop
- Add domain expert review set for Lenin corpus:
  - annotate relevance and ideological fit on sampled retrieval outputs,
  - prioritize disagreement cases for retraining/retagging.

## Active Learning Proposal (Ontology Classifier)
1. Score uncertainty on ontology zero-shot labels.
2. Sample top uncertain documents for manual labeling.
3. Update taxonomy/tagging heuristics from corrected set.
4. Rebuild ontology tags + graph and re-evaluate.

## Initial Interim Targets
- Recall@5 >= 0.05
- Core self ratio >= 0.50
- Ideology consistency >= 0.20
- A/B parity mean >= 0.15

## Escalation Criteria
- If embedding model upgrades are blocked by hardware limits:
  - schedule remote/cloud benchmark lane or memory-optimized quantized alternatives.
- If ideology metric remains near zero despite retrieval gains:
  - prioritize judge/prompt-layer redesign before further retrieval-only tuning.
