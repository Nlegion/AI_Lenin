# Human evaluation checklist (M3 calibration)

## Purpose

Calibrate anti-cliché / anachronism gates and dialectical R1–R3 quality.  
This is a **pilot**, not validation for enabling `mode: block`.

## H1-d block bar (do not flip without all)

- ≥50 unique labeled `(query, answer)` pairs
- Soft balance ~50% warn / ~50% pass
- Precision ≥ 0.8 and recall ≥ 0.7 on that set
- Disagreement review recorded
- Explicit maintainer / architect sign-off

Until then keep `anti_cliche.mode: warn_only` and `block` experimental.

## Weekly accumulation loop (owner: maintainer / architect)

1. Collect candidates: `python scripts/collect_anti_cliche_label_batch.py` (and/or recent `run_local_rag_dryrun` outputs). Prefer real dry-run news over synthetic fixtures.
2. Stratify by topic (economy / geopolitics / tech / social).
3. Human-label a sample (prefer 2 raters; if 1, second-pass all fails + 20% of passes).
4. Merge unique pairs into `.cursor/artifacts/human_eval/`; track count toward 50 in the summary artifact.
5. Feed failures into `config/anti_cliche.yaml` / fixtures in the same PR (update YAML comments + README).
6. Repeat weekly until the H1-d bar is met.

## Scoring questions (per item)

1. Is the answer a **stylistic cliché** (dense stereotype lexicon without news concreteness)?
2. Is it an **unsupported generalization** (Lenin-authoritative tone with weak/no R1 overlap)?
3. Does it use R1 / PSS substantively when R1 was available?
4. Is opposition / R3 considered when relevant?
5. **Anachronism:** does the model claim firsthand modern-tech experience (not quoting the news)?
6. Safety / NewsGuard issues?
7. Overall: publishable with warn-only metadata? (yes/no)
8. Uncertainty? Mark `disagreement` / `uncertain` when unsure.

## Class examples

**Cliché (positive):** “Всё это опять эксплуатация пролетариата и диктатура буржуазии” with no news facts and no R1 quote.

**Not cliché:** Short analysis that cites R1 (“как писал…”) and ties to the specific wage dispute in the news.

**Anachronism (positive):** “Я пользовался TikTok и видел…”

**Not anachronism:** “В новости говорится о TikTok…” or a quoted expert statement.

## Disagreement process

Log both scores. Resolve via short discussion **or** a third pass on that item. Do not silently majority-vote without a record in the human_eval artifact.
