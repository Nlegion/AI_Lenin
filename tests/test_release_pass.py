"""Unit tests for release_pass CLI skip/override behavior."""

from __future__ import annotations

from pathlib import Path

import scripts.release_pass as release_pass


def test_skip_rag_does_not_invoke_rag(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> None:
        calls.append(command)

    monkeypatch.setattr(release_pass, "_run", fake_run)
    monkeypatch.setattr(release_pass, "REPO_ROOT", Path(release_pass.REPO_ROOT))
    code = release_pass.main(
        [
            "--skip-subplan",
            "--skip-rag-quality",
            "--skip-security-m",
            "--skip-anti-cliche",
        ]
    )
    assert code == 0
    assert not any("evaluate_rag_quality.py" in " ".join(cmd) for cmd in calls)


def test_override_rag_writes_artifact(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> None:
        calls.append(command)

    monkeypatch.setattr(release_pass, "_run", fake_run)
    repo = Path(release_pass.REPO_ROOT)
    code = release_pass.main(
        [
            "--skip-subplan",
            "--override-rag-quality",
            "embedding_cutover",
            "--skip-security-m",
            "--skip-anti-cliche",
        ]
    )
    assert code == 0
    override = repo / ".cursor/artifacts/evaluation/rag_quality_override.json"
    assert override.is_file()
    assert not any("evaluate_rag_quality.py" in " ".join(cmd) for cmd in calls)


def test_eval_failure_exits_nonzero(monkeypatch) -> None:
    from src.core.settings.release_gates import RagQualityGate, ReleaseGatesConfig

    def fake_run(command: list[str], *, cwd: Path) -> None:
        joined = " ".join(command)
        if "evaluate_rag_quality.py" in joined:
            raise RuntimeError("rag boom")

    monkeypatch.setattr(release_pass, "_run", fake_run)
    monkeypatch.setattr(
        release_pass,
        "load_release_gates",
        lambda path=None: ReleaseGatesConfig(rag_quality=RagQualityGate(enabled=True)),
    )
    code = release_pass.main(
        ["--skip-subplan", "--skip-security-m", "--skip-anti-cliche"]
    )
    assert code == 1
