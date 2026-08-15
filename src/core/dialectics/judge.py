"""Optional sampled LLM judge (never upgrades hard-validator fail to publish)."""

from __future__ import annotations

import json
from typing import Any

from src.core.dialectics.parse import parse_json_object
from src.core.dialectics.schemas import DialecticalResult
from src.core.llm.base import GenerationBackend, GenerationRequest

JUDGE_SYSTEM = """Ты — строгий академический проверяющий диалектического анализа.
Оцени JSON-анализ. Не повышай оценку при пустых общих фразах.
Верни только JSON: {"scores":{"grounding":0-5,"causal_specificity":0-5,"boilerplate":0-5,"r3_honesty":0-5},"fatal":false,"notes":[]}
"""


async def sample_judge(
    *,
    backend: GenerationBackend,
    result: DialecticalResult,
) -> dict[str, Any]:
    payload = {
        "fact": result.fact,
        "mechanism_steps": result.mechanism_steps,
        "conclusion": result.conclusion,
        "used_principle_ids": [p.principle_id for p in result.used_principles],
        "errors": result.quality.errors,
    }
    user = json.dumps(payload, ensure_ascii=False)
    request = GenerationRequest(
        system_prompt=JUDGE_SYSTEM,
        user_content=user,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    response = await backend.generate(request=request)
    parsed = parse_json_object(response.text)
    if parsed.data is None:
        return {"available": False, "fatal": False, "parse_status": parsed.status}
    data = parsed.data
    return {
        "available": True,
        "fatal": bool(data.get("fatal")),
        "scores": data.get("scores") or {},
        "notes": data.get("notes") or [],
        "parse_status": parsed.status,
    }
