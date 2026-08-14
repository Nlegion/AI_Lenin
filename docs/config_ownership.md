# Config ownership (runtime)

## Source of truth

| Concern | File | Loader |
|---------|------|--------|
| Retrieval / Qdrant | `config/retrieval_pipeline.yaml` → `retrieval_pipeline` | `src/core/retrieval/provider_factory.py` |
| Dialectical slot orchestration | `config/retrieval_pipeline.yaml` → `dialectical_orchestration` | `src/core/analysis/dialectical_config.py` |
| Dialectical reasoning limits | `config/retrieval_pipeline.yaml` → `dialectical_reasoning` | `src/core/dialectics/config.py` |
| LLM runtime knobs (ctx/temp/max_tokens) | `config/generation.yaml` | `src/core/settings/generation_config.py` (**SoT**) |
| Censorship runtime | `config/safety_gate_config.yaml` | `src/core/settings/censorship_runtime_config.py` |
| Quality postcheck | `config/quality_postcheck.yaml` | `src/core/settings/quality_postcheck_config.py` |
| Answer postprocess writer | same YAML → `postprocess_clean_mode` | `src/core/generation/postprocess_clean/` |

## Overlap policy

Shared LLM knobs (`ctx_size`, `temperature`, `max_tokens`) may still appear under
`dialectical_reasoning` for backward compatibility, but runtime must prefer
`generation.yaml` via `load_reasoning_config_with_generation_sot()`.

Physical YAML split (separate files) is optional and requires updating all three
loaders together with a one-release fallback.
