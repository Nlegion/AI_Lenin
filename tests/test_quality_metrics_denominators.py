from __future__ import annotations

from scripts.quality.evaluate_quality_qa_metrics import evaluate_rows


def test_metrics_expose_generation_denominators_and_quote_applicability() -> None:
    rows = [
        {
            "id": "a",
            "answer": "Анализ данной темы невозможен в соответствии с политикой безопасности.",
            "status": "blocked",
            "blocked": True,
            "skipped_llm": True,
            "skipped_llm_reason": "pre_quarantine",
            "llm_attempted": False,
            "llm_generated": False,
            "llm_final_used": False,
            "topic": "review",
        },
        {
            "id": "b",
            "answer": "Факт: инфляция выросла. Механизм: капитал переносит издержки. Вывод: рост цен усиливает неравенство.",
            "status": "done",
            "blocked": False,
            "skipped_llm": False,
            "llm_attempted": True,
            "llm_generated": True,
            "llm_final_used": True,
            "topic": "allow",
            "latency_ms": 1000,
        },
    ]
    metrics = evaluate_rows(rows=rows, suite="full")
    assert metrics["n"] == 2
    assert metrics["eligible_generated_n"] == 1
    assert metrics["llm_attempted_n"] == 1
    assert metrics["llm_generated_n"] == 1
    assert metrics["skipped_llm_n"] == 1
    assert metrics["quote_metrics_applicable"] is False
