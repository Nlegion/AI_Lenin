# LLM client (`src/core/llm/`)

Transport and process lifecycle for the local OpenAI-compatible llama-server, plus an explicit remote provider switch for DeepSeek and a generic OpenAI-compatible HTTP seam (`LLM_SPAWN_LOCAL=false`). Prompting, pipeline, and postprocess stay in `src/core/generation/`.

## Layout

| Module | Role |
|--------|------|
| `base.py` | `GenerationRequest` / `GenerationResponse` / `GenerationBackend` Protocol |
| `chat_completions.py` | `ChatCompletionsBackend` → `POST /v1/chat/completions` |
| `deepseek.py` | `DeepSeekBackend` → `POST /chat/completions` (thinking payload) |
| `factory.py` | `build_generation_backend` (provider + `api_style` gate) |
| `server.py` | `LeninServer` process start/stop (local spawn only) |
| `health.py` | `is_llama_server_active` |
| `runtime.py` | `resolve_llama_runtime` / `LlamaRuntimePaths` |

Config SoT remains [`config/generation.yaml`](../config/generation.yaml) via [`src/core/settings/generation_config.py`](../src/core/settings/generation_config.py). DeepSeek URL/key validation helpers live in [`src/core/settings/deepseek_config.py`](../src/core/settings/deepseek_config.py).

## Provider switch

Top-level `generation.provider` (`llama` | `deepseek`, default `llama`) selects the adapter in `factory.py`. Override with `LLM_PROVIDER`.

| Provider | Adapter | Endpoint | Local spawn |
|----------|---------|----------|-------------|
| `llama` | `ChatCompletionsBackend` | `{server_url}/v1/chat/completions` | Allowed (`LLM_SPAWN_LOCAL=true` default) |
| `deepseek` | `DeepSeekBackend` | `{server_url}/chat/completions` | Forbidden (`LLM_SPAWN_LOCAL=false` required) |

`api_style` stays `chat_completions` for both paths.

## Env overrides (remote / VPS / DeepSeek)

Applied in `load_generation_config` via `apply_generation_env_overrides` (centralized fail-fast validation):

| Env | Effect |
|-----|--------|
| `LLM_PROVIDER` | `llama` (default) or `deepseek`. Invalid values raise at config load. |
| `LLM_SPAWN_LOCAL` | Default `true`. When `false`, `NewsProcessor` does not create/start/stop `LeninServer`. Required `false` for DeepSeek. |
| `GENERATION_SERVER_URL` | Overrides `server_url`; trailing `/` and `/v1` are stripped. For DeepSeek, defaults to `https://api.deepseek.com` when unset / still local. |
| `LLM_API_KEY` | Bearer token (wins over `DEEPSEEK_API_KEY`). |
| `DEEPSEEK_API_KEY` | Used only when `provider=deepseek` and `LLM_API_KEY` is empty. |
| `LLM_MODEL_NAME` | Overrides active backend `model_name`. Required for generic remote (`provider=llama`, spawn off). For DeepSeek defaults to `deepseek-v4-flash`. |
| `LLM_DEEPSEEK_ALLOW_INSECURE_URL` | Escape hatch for trusted HTTP proxies when `provider=deepseek` (default: HTTPS-only). |

`main.py` loads generation config before RAG preflight and `NewsProcessor` construction so misconfiguration fails fast.

In remote llama mode, chat payloads omit llama-only fields `repetition_penalty` and `seed`. DeepSeek payloads always include top-level `thinking` (`disabled` by default); `reasoning_effort` is sent only when thinking is enabled.

## Economy preset (DeepSeek from local PC)

Recommended `.env` for cost-conscious remote generation without local GGUF:

```env
LLM_SPAWN_LOCAL=false
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
LLM_MODEL_NAME=deepseek-v4-flash
```

Defaults: non-thinking mode, model `deepseek-v4-flash`, HTTPS API host, no streaming. Prefer tuning either `temperature` or `top_p` (not both aggressively). Keep `max_tokens` bounded in `config/generation.yaml`.

## Dependency rule

`src.core.llm` must **not** import `src.core.generation` at module level. Persona fallback uses a function-local import inside `build_generation_backend` when `apply_fallback_recommendation=True`.

`GenerationBackend` does not require `close()`; callers use duck-typed `getattr(backend, "close", None)`.

## Session injection (`--llm-timeout`)

`LeninAnalyzer` owns an optional `aiohttp.ClientSession` and passes it into the factory. QA scripts set `analyzer.session = ClientSession(timeout=...)` before `_get_pipeline()` so `--llm-timeout` reaches the HTTP client. If no session is injected, backends create and own one (`_owns_session`).

## Adding a provider

1. Implement (or subclass) `async def generate(self, request: GenerationRequest) -> GenerationResponse`.
2. Branch on top-level `provider` in `factory.py` (keep `api_style=chat_completions` unless the wire format truly differs).
3. Keep process lifecycle (`LeninServer`) separate from the HTTP/client class when the new provider is remote.
4. Add env validation in `apply_generation_env_overrides` / a dedicated settings helper.

One-release shims: `src/core/llama_server.py`, `src/core/generation/{base,chat_backend,factory}.py`, `src/core/settings/llama_runtime.py`, and `device.is_llama_server_active` (function-local import of `llm.health`).
