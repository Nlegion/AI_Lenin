"""Mock GenerationBackend modes for dialectical engine tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from src.core.llm.base import GenerationRequest, GenerationResponse


@dataclass
class MockBackend:
    mode: str = "valid"
    principle_id: str = "pc-test"
    chunk_id: str = "c1"
    calls: int = 0

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        if self.mode == "timeout":
            await asyncio.sleep(3600)
        if self.mode == "backend_error":
            raise RuntimeError("mock backend failure")
        if self.mode == "length":
            return GenerationResponse(
                text='{"fact": "x"',
                backend="mock",
                model_name="mock",
                latency_ms=1,
                finish_reason="length",
            )
        if self.mode == "fenced":
            payload = self._valid_payload()
            text = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
            return GenerationResponse(
                text=text,
                backend="mock",
                model_name="mock",
                latency_ms=1,
                finish_reason="stop",
            )
        if self.mode == "bad_ids":
            payload = self._valid_payload()
            payload["used_principle_ids"] = ["pc-missing"]
            payload["evidence_ids"] = ["missing-chunk"]
            return GenerationResponse(
                text=json.dumps(payload, ensure_ascii=False),
                backend="mock",
                model_name="mock",
                latency_ms=1,
                finish_reason="stop",
            )
        if self.mode == "malformed":
            return GenerationResponse(
                text="not-json at all",
                backend="mock",
                model_name="mock",
                latency_ms=1,
                finish_reason="stop",
            )
        if self.mode == "repair_then_valid" and self.calls == 1:
            return GenerationResponse(
                text="broken",
                backend="mock",
                model_name="mock",
                latency_ms=1,
                finish_reason="stop",
            )
        payload = self._valid_payload()
        return GenerationResponse(
            text=json.dumps(payload, ensure_ascii=False),
            backend="mock",
            model_name="mock",
            latency_ms=1,
            finish_reason="stop",
        )

    def _valid_payload(self) -> dict:
        return {
            "fact": "Правительство ввело регулирование нефтегаза.",
            "thesis": "Меры подаются как защита граждан и стабильность.",
            "antithesis": "Госрегулирование при капитализме обслуживает монополии.",
            "synthesis": "Предел реформизма — сохранение частной собственности.",
            "mechanism_steps": [
                "Из-за сохранения капиталистической собственности "
                "регулирование при буржуазном государстве "
                "приводит к переносу издержек на трудящихся."
            ],
            "conclusion": "Для трудящихся это не устранение эксплуатации.",
            "used_principle_ids": [self.principle_id],
            "evidence_ids": [self.chunk_id],
            "causal_links": [
                {
                    "cause": "частная собственность на отрасль",
                    "condition": "госрегулирование без смены классового характера государства",
                    "effect": "поддержка монополий за счёт бюджета",
                    "theoretical_basis": "госкапитализм",
                    "evidence_ids": [self.chunk_id],
                    "principle_ids": [self.principle_id],
                    "confidence": 0.8,
                }
            ],
            "thesis_from": self.chunk_id,
            "antithesis_from": "c2",
            "synthesis_basis": self.chunk_id,
            "r3_handling": "r3_absent",
        }
