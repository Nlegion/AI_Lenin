# Contributing

Thank you for interest in AI_Lenin. Contributions that improve code quality, safety, tests, or documentation are welcome.

## How to contribute

1. Fork the repository (or use a branch if you have write access).
2. Create a focused branch (`feature/…`, `fix/…`).
3. Make changes that match project conventions (see below).
4. Add or update tests when behavior changes.
5. Open a Pull Request with a short description of **why** the change is needed.

## Code style

Aligned with [`AGENTS.md`](AGENTS.md) and `.cursor/rules/`:

- **PEP 8** for Python formatting.
- **Type hints** on new and modified public functions (when practical).
- Prefer **keyword arguments** for calls with 3+ parameters.
- Keep files under roughly **200 lines** when creating or significantly expanding modules; split by responsibility.
- Use **structured logging** (`logging` / `structlog`); do not use `print` as a substitute.
- Catch **specific** exception types; no bare `except:` and no silent `except …: pass`.

## Testing

Before opening a PR, run what you can locally:

```powershell
pytest tests -q
```

For a single module: `pytest tests/test_<name>.py -q`.  
Some scripts under `tests/` are integration-style; see `.cursor/rules/testing.mdc`.

## License

By contributing, you agree that your contributions are licensed under the project’s MIT License ([LICENSE](LICENSE)). No CLA is required. Do not add per-file copyright banners unless there is a strong reason.

Please read [DISCLAIMER.md](DISCLAIMER.md) before contributing features that affect public-facing generated text, corpus handling, or safety gates.

## Scope notes

- Do not commit secrets (`.env`), model weights, local databases, or the `/data/` corpus.
- Do not invent corpus digitization URLs or claim whole PSS volume files are public domain.
- Agent-oriented conventions live in [`AGENTS.md`](AGENTS.md); this file is for human contributors.
