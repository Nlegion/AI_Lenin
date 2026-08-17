"""Regression: VPS publication leaks must hard_block after term expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.safety.news_guard import NewsGuard
from src.core.safety.pre_rag_censor import CensorRuntimeConfig, PreRagCensor
from src.core.safety.pre_rag_censor_types import CensorInput
from src.core.safety.safety_gate import SafetyGate

ROOT = Path(__file__).resolve().parents[1]


def _censor(**overrides) -> PreRagCensor:
    cfg = CensorRuntimeConfig(**overrides)
    return PreRagCensor(
        safety_gate=SafetyGate.from_base_dir(ROOT),
        news_guard=NewsGuard.from_file(ROOT / "config" / "news_guard.yaml"),
        config=cfg,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("news_id", "title", "category", "reason_code"),
    [
        (
            "vps-ufc",
            "Глава UFC заявил, что его взбесило поведение Гэрри в поединке с Махачевым",
            "SPORT_BLOCKED",
            "manual_sport_hard_block",
        ),
        (
            "vps-hockey",
            'Новичок "Рейнджерс" хоккеист Дорофеев вызывает уважение у Гаврикова',
            "SPORT_BLOCKED",
            "manual_sport_hard_block",
        ),
        (
            "vps-sync",
            "Синхронистка Штатнова выиграла золото в произвольной программе ЮЧМ",
            "SPORT_BLOCKED",
            "manual_sport_hard_block",
        ),
        (
            "vps-concert",
            "В Санкт-Петербурге 10 и 11 октября пройдут концерты Канье Уэста",
            "MUSIC",
            "manual_music_hard_block",
        ),
        (
            "vps-boxoffice",
            '"Последний богатырь: Колобок" возглавил кинопрокат в России и СНГ за выходные',
            "CINEMA",
            "manual_cinema_hard_block",
        ),
        (
            "vps-flood",
            "В Забайкалье подтопило более 40 домов и 200 приусадебных участков",
            "NATURAL_DISASTER",
            "manual_natural_disaster_hard_block",
        ),
        (
            "vps-collapse",
            "В Нижегородской области обрушилась стена жилого дома",
            "FIRE",
            "manual_fire_hard_block",
        ),
        (
            "vps-auto",
            '"Автостат": доля электромобилей на авторынке РФ впервые превысила 2%',
            "AUTO",
            "manual_auto_hard_block",
        ),
    ],
)
async def test_vps_leak_titles_hard_block(
    news_id: str,
    title: str,
    category: str,
    reason_code: str,
) -> None:
    result = await _censor().evaluate(
        CensorInput(
            news_id=news_id,
            title=title,
            body=title,
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == category
    assert reason_code in result.reason_codes


@pytest.mark.asyncio
async def test_economy_politics_non_block_avoids_federac_liga() -> None:
    """Non-block control must not contain федерац/лига (sport regex FP stems)."""
    result = await _censor().evaluate(
        CensorInput(
            news_id="econ-ok",
            title="Минфин оценил динамику доходов бюджета и инфляции",
            body="Кабмин обсуждает налоговые меры и тарифы для промышленности.",
            source="TASS",
        )
    )
    assert result.decision in {"allow", "review"}
    assert result.category != "SPORT_BLOCKED"


@pytest.mark.asyncio
async def test_oms_dolgoletiya_remains_allow() -> None:
    result = await _censor().evaluate(
        CensorInput(
            news_id="oms-ok",
            title="Мишустин: центры здорового долголетия должны появиться до конца года",
            body="По ОМС россияне смогут пройти обследование в центрах медицины.",
            source="TASS",
        )
    )
    assert result.decision in {"allow", "review"}
    assert result.category != "WELLNESS"


@pytest.mark.asyncio
async def test_rejected_broad_film_stem_not_added() -> None:
    """Documentary/cult-film titles without кинопрокат must not hard_block as CINEMA."""
    result = await _censor().evaluate(
        CensorInput(
            news_id="film-control",
            title='От "Рокки" до "Брата": фильмы, которые стали культовыми вопреки всему',
            body="Обзор культовых картин в истории кино без кассовых сборов.",
            source="TASS",
        )
    )
    assert result.category != "CINEMA"


@pytest.mark.asyncio
async def test_rejected_obrushil_market_metaphor_not_fire() -> None:
    """Market metaphor обрушил must not hard_block as FIRE (stem intentionally not added)."""
    result = await _censor().evaluate(
        CensorInput(
            news_id="market-control",
            title="Аналитики: решение регулятора обрушило котировки на бирже",
            body="Инвесторы фиксируют снижение индексов без коммунальной аварии.",
            source="TASS",
        )
    )
    assert result.category != "FIRE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("news_id", "title"),
    [
        (
            "oly-swim",
            "Российские пловцы стали третьими в неофициальном медальном зачете ЧЕ",
        ),
        (
            "oly-wrestle",
            "Россиянин выиграл золото по вольной борьбе на чемпионате мира",
        ),
        ("oly-greco", "Сборная взяла медали в греко-римской борьбе"),
        ("oly-judo", "Дзюдоист сборной победил в финале Олимпиады"),
        ("oly-taekwondo", "Тхэквондистка вышла в полуфинал мирового первенства"),
        ("oly-waterpolo", "Сборная по водному поло сыграла вничью"),
        ("oly-rowing", "Экипаж академической гребли финишировал первым"),
        ("oly-kickbox", "Кикбоксер победил нокаутом в титульном бою"),
    ],
)
async def test_olympic_and_combat_sports_hard_block(
    news_id: str,
    title: str,
) -> None:
    result = await _censor().evaluate(
        CensorInput(
            news_id=news_id,
            title=title,
            body=title,
            source="TASS",
        )
    )
    assert result.decision == "hard_block"
    assert result.category == "SPORT_BLOCKED"


@pytest.mark.asyncio
async def test_class_struggle_not_blocked_as_sport() -> None:
    """Bare борьб is not a sport term — avoid FP on political phrasing."""
    result = await _censor().evaluate(
        CensorInput(
            news_id="politics-struggle",
            title="Эксперты обсудили классовую борьбу и роль профсоюзов",
            body="В материале о трудовых конфликтах и коллективных договорах.",
            source="TASS",
        )
    )
    assert result.category != "SPORT_BLOCKED"
