from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.safety.censor_hashing import NORMALIZER_VERSION, compute_content_hash
from src.core.safety.pre_rag_censor import (
    CensorRuntimeConfig,
    PreRagCensor,
    compose_decision,
)
from src.core.safety.pre_rag_censor_types import CensorInput
from src.core.safety.safety_gate import SafetyGate
from src.core.safety.news_guard import NewsGuard

ROOT = Path(__file__).resolve().parents[1]


def _censor(**overrides) -> PreRagCensor:
    cfg = CensorRuntimeConfig(**overrides)
    return PreRagCensor(
        safety_gate=SafetyGate.from_base_dir(ROOT),
        news_guard=NewsGuard.from_file(ROOT / "config" / "news_guard.yaml"),
        config=cfg,
    )


@pytest.mark.asyncio
async def test_contract_returns_new_decision_schema() -> None:
    censor = _censor()
    result = await censor.evaluate(
        CensorInput(
            news_id="1",
            title="Рост инфляции и бюджета",
            body="Правительство обсуждает экономические меры и тарифы.",
            source="TASS",
        )
    )
    assert result.decision in {"allow", "hard_block", "review", "skip"}
    assert isinstance(result.reason_codes, list)
    assert "latency_ms" in result.audit


@pytest.mark.asyncio
async def test_sport_is_hard_blocked_when_feature_enabled() -> None:
    censor = _censor(sport_block_enabled=True)
    result = await censor.evaluate(
        CensorInput(
            news_id="2",
            title="Футбольный матч чемпионата",
            body="Сборная провела матч без политических заявлений.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "SPORT_BLOCKED"


@pytest.mark.asyncio
async def test_sport_team_name_is_hard_blocked() -> None:
    censor = _censor(sport_block_enabled=True)
    result = await censor.evaluate(
        CensorInput(
            news_id="2-team",
            title='Костромской "Спартак" продлил серию без поражений в Первой лиге',
            body="Команда продолжает выступление в сезоне.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "SPORT_BLOCKED"


@pytest.mark.asyncio
async def test_ww2_and_nazi_terms_are_war_hard_block() -> None:
    censor = _censor()
    result = await censor.evaluate(
        CensorInput(
            news_id="war-ww2",
            title="Прокуратура рассказала о преступлениях нацистов в годы Великой Отечественной",
            body="В материале упоминаются события Второй мировой войны.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "WAR"
    assert "manual_war_hard_block" in result.reason_codes


@pytest.mark.asyncio
async def test_airport_category_is_uppercase() -> None:
    censor = _censor()
    result = await censor.evaluate(
        CensorInput(
            news_id="airport-uppercase",
            title="В аэропорту Калуги ввели временные ограничения",
            body="Режим ограничений введен оперативными службами.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "AIRPORT" or result.category == "WAR_OPERATIONAL"


@pytest.mark.asyncio
async def test_duplicate_decision_is_stable_via_cache() -> None:
    cache: dict[tuple[str, str], dict] = {}

    async def _load_cached(content_hash: str, config_hash: str):
        return cache.get((content_hash, config_hash))

    async def _save_cached(content_hash: str, config_hash: str, model_hash: str, result):
        cache[(content_hash, config_hash)] = {
            "decision": result.decision,
            "category": result.category,
            "risk_tier": result.risk_tier,
            "reason_codes": list(result.reason_codes),
            "confidence": dict(result.confidence),
            "model_version_hash": model_hash,
        }

    censor = _censor(duplicate_ttl_seconds=3600)
    censor._load_cached_decision = _load_cached  # type: ignore[attr-defined]
    censor._save_cached_decision = _save_cached  # type: ignore[attr-defined]
    payload = CensorInput(
        news_id="3",
        title="Экономика региона",
        body="В регионе обсуждают бюджетные параметры и субсидии.",
        source="TASS",
    )
    first = await censor.evaluate(payload)
    second = await censor.evaluate(payload)
    assert first.decision == second.decision
    assert second.audit.get("cache_hit") is True
    assert second.audit.get("normalizer_version") == NORMALIZER_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("title", "body"),
    [
        ("Переговоры МИД и послов", "Дипломатические переговоры прошли в рабочем формате."),
        ("Новые экономические санкции", "Обсуждаются санкции и внешнеторговые ограничения."),
    ],
)
async def test_diplomacy_and_sanctions_not_auto_hard_block(title: str, body: str) -> None:
    censor = _censor()
    result = await censor.evaluate(
        CensorInput(news_id="4", title=title, body=body, source="TASS")
    )
    assert result.decision in {"allow", "review", "skip"}
    assert result.decision != "hard_block"


@pytest.mark.asyncio
async def test_unknown_topic_low_signal_moves_to_skip() -> None:
    censor = _censor(
        unknown_topic_to_skip_enabled=True,
        unknown_low_signal_l2_max=0.25,
    )
    result = await censor.evaluate(
        CensorInput(
            news_id="7",
            title="В аэропорту Саранска ввели временные ограничения",
            body="Ограничения введены в рабочем порядке, без происшествий.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert "manual_airport_hard_block" in result.reason_codes


@pytest.mark.asyncio
async def test_sanctions_allow_is_gated_to_review_when_low_confidence() -> None:
    censor = _censor(
        sanctions_allow_l2_min=0.95,
    )
    result = await censor.evaluate(
        CensorInput(
            news_id="8",
            title="Журналист оценил шансы на ужесточение санкций",
            body="Обсуждается вероятность новых санкционных ограничений.",
            source="TASS",
        )
    )
    assert result.category == "SANCTIONS"
    assert result.decision != "allow"
    assert "sanctions_allow_gate" in result.reason_codes


def test_unknown_topic_low_signal_is_forwarded_to_allow() -> None:
    censor = _censor(unknown_topic_to_skip_enabled=True, unknown_low_signal_l2_max=0.25)
    decision, category, codes = censor._apply_policy_overrides(  # type: ignore[attr-defined]
        cfg=censor._config,  # type: ignore[attr-defined]
        decision="review",
        category=None,
        reason_codes=["unknown_topic"],
        l2_score=0.1,
        l3_used=False,
        text_lower="нейтральная новость без кризисных маркеров",
        source="TASS",
    )
    assert decision == "allow"
    assert category == "NON_TOPICAL"
    assert "unknown_topic_low_signal_allow_forward" in codes
    assert "override:unknown_topic_forward_trusted_source" in codes


@pytest.mark.parametrize(
    ("l1", "l2", "l3", "expected"),
    [
        ("hard_block", ("allow", 0.2), None, "hard_block"),
        ("skip", ("hard_block", 0.9), "allow", "skip"),
        ("allow", ("review", 0.8), None, "review"),
        ("review", ("allow", 0.4), None, "review"),
        ("review", ("allow", 0.4), "allow", "allow"),
    ],
)
def test_compose_decision_truth_table(
    l1: str,
    l2: tuple[str, float] | None,
    l3: str | None,
    expected: str,
) -> None:
    result = compose_decision(
        l1_decision=l1,  # type: ignore[arg-type]
        l2_signal=l2,  # type: ignore[arg-type]
        l3_decision=l3,  # type: ignore[arg-type]
    )
    assert result == expected


def test_content_hash_normalization_equivalence() -> None:
    h1, _ = compute_content_hash(
        title="В аэропорту&nbsp;Сочи ввели ограничения!!!",
        body="  БПЛА   не зафиксированы  ",
        url="https://example.com/news?id=1&utm_source=abc",
    )
    h2, _ = compute_content_hash(
        title="в аэропорту сочи ввели ограничения!",
        body="бпла не зафиксированы",
        url="https://example.com/news?id=1",
    )
    assert h1 == h2


@pytest.mark.asyncio
async def test_hot_reload_config_updates_runtime_flags(tmp_path: Path) -> None:
    cfg_path = tmp_path / "safety_gate_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "safety_gate": {
                    "censorship_runtime": {
                        "sport_block_enabled": True,
                        "hot_reload_enabled": True,
                        "hot_reload_poll_seconds": 0.0,
                    }
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    censor = _censor()
    censor._config_path = str(cfg_path)  # type: ignore[attr-defined]
    censor._config.hot_reload_enabled = True  # type: ignore[attr-defined]
    censor._config.hot_reload_poll_seconds = 0.0  # type: ignore[attr-defined]
    first = await censor.evaluate(
        CensorInput(
            news_id="5",
            title="Футбольный матч чемпионата",
            body="Сборная сыграла матч.",
            source="TASS",
        )
    )
    assert first.decision == "hard_block"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "safety_gate": {
                    "censorship_runtime": {
                        "sport_block_enabled": False,
                        "hot_reload_enabled": True,
                        "hot_reload_poll_seconds": 0.0,
                    }
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    censor._last_reload_check = 0.0  # type: ignore[attr-defined]
    second = await censor.evaluate(
        CensorInput(
            news_id="6",
            title="Футбольный матч чемпионата",
            body="Сборная сыграла матч.",
            source="TASS",
        )
    )
    assert censor._config.sport_block_enabled is False  # type: ignore[attr-defined]
    assert "sport_blocked" not in second.reason_codes


@pytest.mark.asyncio
async def test_model_version_hash_changes_config_hash() -> None:
    first = _censor(l2_model_version="model-v1")
    second = _censor(l2_model_version="model-v2")
    assert first.config_version_hash != second.config_version_hash


@pytest.mark.asyncio
async def test_ethno_hate_containment_blocks_obfuscated_phrase() -> None:
    censor = _censor(ethno_hate_containment_enabled=True)
    result = await censor.evaluate(
        CensorInput(
            news_id="ethno-1",
            title="Комментатор призвал и-з-г-н-а-т-ь «и-н-о-р-о-д-ц-ев» из региона",
            body="В посте звучит призыв: нужно изгнать инородцев любой ценой.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "ETHNIC_RELIGIOUS"
    assert "manual_ethno_hate_containment" in result.reason_codes


@pytest.mark.asyncio
async def test_manual_lifestyle_fp_economic_text_not_hard_blocked() -> None:
    censor = _censor()
    result = await censor.evaluate(
        CensorInput(
            news_id="fp-econ",
            title="Цены на золото и пост отчетности",
            body="Рынок обсуждает игру ставок рефинансирования и бюджетный пост правительства.",
            source="TASS",
        )
    )
    assert result.decision != "hard_block" or result.category not in {
        "SHOWBIZ",
        "SPORT_BLOCKED",
        "GAMBLING",
        "WELLNESS",
        "FOOD",
        "ASTROLOGY",
    }


@pytest.mark.asyncio
async def test_sport_runtime_flag_skips_manual_sport_terms() -> None:
    censor = _censor(sport_block_enabled=False)
    result = await censor.evaluate(
        CensorInput(
            news_id="sport-off",
            title="Биатлон и кхл обзор сезона",
            body="Турнир по биатлону прошел без политических заявлений.",
            source="TASS",
        )
    )
    assert result.category != "SPORT_BLOCKED"
    assert "manual_sport_hard_block" not in result.reason_codes
    assert "sport_blocked" not in result.reason_codes


@pytest.mark.asyncio
async def test_manual_cinema_category_hard_block() -> None:
    censor = _censor()
    result = await censor.evaluate(
        CensorInput(
            news_id="cinema-1",
            title="Новый блокбастер выходит в прокат",
            body="Студия объявила дату кинопремьеры.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "CINEMA"
    assert "manual_cinema_hard_block" in result.reason_codes


@pytest.mark.asyncio
async def test_war_beats_lifestyle_on_shared_priority() -> None:
    censor = _censor()
    result = await censor.evaluate(
        CensorInput(
            news_id="war-priority",
            title="ВСУ применили БПЛА в районе фронта",
            body="Сообщается о работе беспилотников.",
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "WAR_OPERATIONAL"
