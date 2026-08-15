"""Tests for token-safe matcher, FIO toponym FP, and 2057-style regressions."""

from __future__ import annotations

from pathlib import Path

from src.core.safety.news_guard import NewsGuard, load_news_guard_config
from src.core.safety.pattern_match import pattern_hits
from src.core.safety.topic_routing import route_topic


def _guard() -> NewsGuard:
    return NewsGuard(config=load_news_guard_config(path=Path("config/news_guard.yaml")))


def test_export_not_matched_as_sport() -> None:
    hits = pattern_hits("экспорт нефти через ктк и порты", ["спорт"])
    assert hits == []
    result = _guard().evaluate_input(
        title="Экспорт нефти через КТК вырос",
        content="Порты увеличили перевалку; экономика и торговля.",
        source="TASS",
    )
    assert result.decision == "allow"


def test_svoi_not_matched_as_svo() -> None:
    hits = pattern_hits("работники отстаивают свои права на заводе audi", ["сво"])
    assert hits == []
    result = _guard().evaluate_input(
        title="Забастовка на заводе Audi",
        content="Работники отстаивают свои права и зарплату; профсоюз обсуждает условия труда.",
        source="TASS",
    )
    assert result.decision == "allow"


def test_nacionalnaya_kompaniya_not_quarantine() -> None:
    hits = pattern_hits("национальная компания строит аэс", ["национальн"])
    assert hits == []


def test_fio_toponym_china_tennis_not_deny() -> None:
    result = _guard().evaluate_input(
        title="Теннис: победа китаянки",
        content="Представительница Китая Ван Синьюй выиграла турнир.",
        source="TASS",
    )
    assert result.decision == "skip"
    assert (
        any("out_of_scope" in c or "sport" in c for c in result.reason_codes)
        or result.decision == "skip"
    )


def test_fio_franz_Josef_land_not_deny() -> None:
    result = _guard().evaluate_input(
        title="Палеонтология",
        content="На Земле Франца Иосифа нашли окаменелости динозавра.",
        source="TASS",
    )
    assert result.decision in {"skip", "allow"}
    assert result.reason != "private pii detected without public-interest context"


def test_charge_context_keeps_fio_deny() -> None:
    result = _guard().evaluate_input(
        title="Следствие",
        content="Иванов Иван Иванович обвинен в мошенничестве, задержан полицией.",
        source="UnknownBlog",
    )
    assert result.decision == "deny"


def test_putin_opened_world_cup_is_full() -> None:
    routed = route_topic(
        title="Путин открыл чемпионат мира по футболу",
        content="Торжественная церемония на стадионе.",
    )
    assert routed.route == "full"
    result = _guard().evaluate_input(
        title="Путин открыл чемпионат мира по футболу",
        content="Торжественная церемония на стадионе.",
        source="TASS",
    )
    assert result.decision == "allow"


def test_kremlin_alone_title_not_full() -> None:
    routed = route_topic(title="Кремль", content="В здании начался ремонт фасада.")
    assert routed.route != "full" or routed.primary != "policy"


def test_vs_rf_combat_deny() -> None:
    result = _guard().evaluate_input(
        title="Сводка",
        content="ВС РФ поразили склад; военные сообщили об обстреле.",
        source="TASS",
    )
    assert result.decision == "deny"


def test_army_of_consumers_not_combat() -> None:
    result = _guard().evaluate_input(
        title="Ритейл",
        content="Армия потребителей нанесла удар по скидкам в сезон распродаж; экономика ритейла.",
        source="TASS",
    )
    assert result.decision == "allow"
