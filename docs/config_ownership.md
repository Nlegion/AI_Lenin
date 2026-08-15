# Config ownership (runtime)

## Source of truth

| Concern | File | Loader |
|---------|------|--------|
| Retrieval / Qdrant | `config/retrieval_pipeline.yaml` → `retrieval_pipeline` | `src/core/retrieval/provider_factory.py` |
| Dialectical slot orchestration | `config/retrieval_pipeline.yaml` → `dialectical_orchestration` | `src/core/analysis/dialectical_config.py` |
| Dialectical reasoning limits | `config/retrieval_pipeline.yaml` → `dialectical_reasoning` | `src/core/dialectics/config.py` |
| LLM runtime knobs (ctx/temp/max_tokens) | `config/generation.yaml` | `src/core/settings/generation_config.py` (**SoT**) |
| Semantic core | `config/semantic_core.yaml` | `src/core/analysis/semantic_core_config.py` |
| Anti-cliché | `config/anti_cliche.yaml` | `src/core/safety/anti_cliche_config.py` |
| Censorship runtime | `config/safety_gate_config.yaml` | `src/core/settings/censorship_runtime_config.py` |
| Manual censor terms | `config/censor_terms/` | `src/core/safety/manual_terms_loader.py` |
| NewsGuard policy | `config/news_guard.yaml` | `src/core/safety/news_guard.py` (`load_news_guard_config`) |
| Quality postcheck | `config/quality_postcheck.yaml` | `src/core/settings/quality_postcheck_config.py` |
| Answer postprocess writer | same YAML → `postprocess_clean_mode` | `src/core/generation/postprocess_clean/` |
| Release thresholds | `config/release_gates.yaml` | `src/core/settings/release_gates.py` |

Current YAML defaults (restart required after edits): `dialectical_orchestration.enabled: true`, `semantic_core.enabled: true`, `postprocess_clean_mode: live`, `anti_cliche.mode: warn_only`.

## Overlap policy

Shared LLM knobs (`ctx_size`, `temperature`, `max_tokens`) may still appear under
`dialectical_reasoning` for backward compatibility, but runtime must prefer
`generation.yaml` via `load_reasoning_config_with_generation_sot()`.

Physical YAML split (separate files) is optional and requires updating all three
loaders together with a one-release fallback.
