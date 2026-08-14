"""Trial50 hotfix regressions: drone/combat soft-pass, sport, FIO, artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts._live_news_qa_gate import apply_live_pre_llm_gate
from scripts._quality_qa_io import QaItem
from src.core.generation.output_artifacts import (
    apply_artifact_pass,
    detect_encoding_artifacts,
)
from src.core.safety.drone_combat_guard import (
    combat_adjacent_hit,
    drone_air_raid_hit,
    soft_pass_allowed,
)
from src.core.safety.news_guard import NewsGuard
from src.core.safety.pattern_match import svo_token_hit
from src.core.settings.quality_postcheck_config import load_quality_postcheck_config

ROOT = Path(__file__).resolve().parents[1]
GUARD = NewsGuard.from_file(ROOT / "config" / "news_guard.yaml")
QC = load_quality_postcheck_config(ROOT / "config" / "quality_postcheck.yaml")


def test_drone_air_raid_deny() -> None:
    result = GUARD.evaluate_input(
        title="Над регионами сбили БПЛА",
        content="В Белгородской области силы ПВО сбили несколько БПЛА, объявлена опасность атаки.",
        source="TASS",
    )
    assert result.decision == "deny"
    assert result.risk_tier == "red"


def test_civil_drone_center_not_denied() -> None:
    result = GUARD.evaluate_input(
        title="Центр беспилотных систем и технологий",
        content="В регионе открыли центр разработки беспилотных систем и технологий для мониторинга.",
        source="TASS",
    )
    assert result.decision != "deny" or "drone" not in " ".join(result.reason_codes)


def test_training_alarm_not_combat_adjacent() -> None:
    hit = combat_adjacent_hit(
        "В Московской области прошли плановые учения МЧС и учебная тревога оповещения."
    )
    assert hit.hit is False


def test_oil_evac_not_combat_adjacent() -> None:
    hit = combat_adjacent_hit(
        "Эвакуация жителей из-за разлива нефти в регионе; МЧС проводит работы."
    )
    assert hit.hit is False


def test_svo_token_not_svoj() -> None:
    assert svo_token_hit(lowered="участники сво прибыли") is True
    assert svo_token_hit(lowered="своими силами жители") is False
    assert svo_token_hit(lowered="своя позиция") is False


def test_soft_pass_blocked_for_bpla_unknown() -> None:
    item = QaItem(
        id="t1",
        title="Опасность атаки БПЛА",
        content="В Курской области объявлена опасность атаки БПЛА и воздушная тревога.",
        question="Проанализируй новость",
        source="TASS",
    )
    # Force unknown path: if already deny, that also satisfies safety.
    gate = GUARD.evaluate_input(
        title=item.title, content=item.content, source=item.source
    )
    if gate.decision == "deny":
        return
    row: dict = {"id": item.id}
    blocked = apply_live_pre_llm_gate(
        guard=GUARD,
        item=item,
        row=row,
        unknown_as_allow=True,
    )
    assert blocked is not None
    assert blocked.get("gate_soft_pass_blocked") or blocked.get("skipped_llm_reason")


def test_soft_pass_contract_uses_shared_helper() -> None:
    text = "В Харькове прогремел мощный взрыв; власти сообщили о жертвах."
    allowed, codes = soft_pass_allowed(risk_tier="yellow", text=text)
    assert allowed is False
    assert codes
    assert combat_adjacent_hit(text).hit is True


def test_passport_not_sport_skip() -> None:
    result = GUARD.evaluate_input(
        title="Выдача паспортов УК в доме",
        content="Жителям многоквартирного дома сообщили о порядке выдачи паспортов и квитанций УК.",
        source="TASS",
    )
    assert "out_of_scope:sport" not in result.reason_codes
    assert result.decision != "skip" or "спорт" not in result.message.lower()


def test_sport_still_skips() -> None:
    result = GUARD.evaluate_input(
        title="Футбольный матч чемпионата",
        content="Сборная сыграла товарищеский матч чемпионата без политических заявлений.",
        source="TASS",
    )
    assert result.decision == "skip"
    assert any("out_of_scope:sport" in c for c in result.reason_codes)


def test_fio_festival_public_interest() -> None:
    result = GUARD.evaluate_input(
        title="Фестиваль молодежи в Госдуме",
        content="Замруководителя аппарата Госдумы Иван Иванов Иванович открыл молодежный фестиваль.",
        source="TASS",
    )
    assert result.decision != "deny" or "fio:" not in " ".join(result.reason_codes)


def test_siloviki_search_crime_carveout() -> None:
    result = GUARD.evaluate_input(
        title="Обыск на Украине",
        content="Российские силовики сообщили об обыске по делу о коррупции; возбуждено уголовное дело.",
        source="TASS",
    )
    assert "context:military_rf_forces" not in result.reason_codes
    assert (
        result.decision != "deny"
        or result.reason != "military/combat topic hard deny matched"
    )


def test_siloviki_obstrel_still_deny() -> None:
    result = GUARD.evaluate_input(
        title="Силовики отразили обстрел",
        content="Российские силовики отразили обстрел на границе; войска ведут ответный огонь.",
        source="TASS",
    )
    assert result.decision == "deny"


def test_artifact_encoding_detect_not_blind_repair() -> None:
    text = "Швейцарские СЃ сообщили о переговорах по экономике."
    assert "artifact:mojibake_sg" in detect_encoding_artifacts(text)
    res = apply_artifact_pass(text=text, config=QC, item_id="enc1")
    # Soft artifact mode: detect/keep, do not blind-repair and do not full-replace.
    assert res.used_fallback is False
    assert "США" not in res.text.replace("переговорах", "")
    assert any(c.startswith("artifact:") or c.startswith("detect:") for c in res.codes)


def test_redact_placeholder_not_bare_delete() -> None:
    text = "Посол России в [обезличено] отверг обвинения и дал развернутый комментарий по существу."
    res = apply_artifact_pass(text=text, config=QC, item_id="r1")
    assert "[обезличено]" not in res.text
    # Visible redaction markers are stripped by final_public_scrub; prose remains.
    assert "«[место]»" not in res.text
    assert "Посол России в" in res.text
    assert "отверг обвинения" in res.text
    assert (
        "strip:redact_placeholder" in res.codes
        or "strip:mesto_marker" in res.codes
        or res.used_fallback
    )


def test_short_loop_strip() -> None:
    text = "Анализ события. *Ленин, *:. принцип :. Далее нормальный текст о экономике и классах общества."
    res = apply_artifact_pass(text=text, config=QC, item_id="loop1")
    assert "*Ленин" not in res.text
    assert "принцип :." not in res.text


def test_disclaimer_footer_not_header() -> None:
    out = GUARD.guard_output(analysis="Краткий анализ экономики и санкций.")
    assert out.moderated_text.strip().startswith("Краткий")
    assert "образовательных целях" in out.moderated_text
    assert out.moderated_text.index("Краткий") < out.moderated_text.index(
        "образовательных"
    )


def test_safe_gold_false_deny_budget() -> None:
    path = ROOT / "data" / "eval" / "safe_gold_trial50.jsonl"
    if not path.is_file():
        return
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    false_deny = 0
    for row in rows:
        result = GUARD.evaluate_input(
            title=row["title"], content=row["content"], source="TASS"
        )
        if result.decision == "deny" and row.get("expect") != "deny":
            false_deny += 1
    assert false_deny <= max(1, int(0.02 * len(rows) + 1))


def test_sbil_matrix() -> None:
    assert drone_air_raid_hit("В области сбили БПЛА над регионом РФ.").hit is True
    assert (
        drone_air_raid_hit("Водитель сбил пешехода на переходе в городе.").hit is False
    )
    assert (
        drone_air_raid_hit("Его сбил с толку странный заголовок новости.").hit is False
    )
    assert (
        combat_adjacent_hit("Кибератака на сайт банка со стороны конкурентов.").hit
        is False
    )


def test_drone_red_gold_no_false_accept() -> None:
    path = ROOT / "data" / "eval" / "red_gold_drone.jsonl"
    if not path.is_file():
        return
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        hit = drone_air_raid_hit(f"{row['title']}\n{row['content']}")
        result = GUARD.evaluate_input(
            title=row["title"], content=row["content"], source="TASS"
        )
        assert hit.hit or result.decision == "deny"
