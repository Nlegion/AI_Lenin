"""Build curated config/censor_terms/*.yaml from models/words.txt draft."""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.safety.manual_terms_policy import ACRONYM_ALLOWLIST, AMBIGUITY_BANLIST  # noqa: E402

DRAFT = ROOT / "models" / "words.txt"
OUT_DIR = ROOT / "config" / "censor_terms"
ARTIFACT_DIR = ROOT / ".cursor" / "artifacts"

# Extra scrub drops (build-time only). Seeds may re-introduce precise production terms.
SCRUB_EXTRA_BANLIST = frozenset(
    {
        "встреч",
        "круг",
        "карточк",
        "судья",
        "капитан",
        "игрок",
        "аренд",
        "рекорд",
        "бронзов",
        "серебрян",
        "медал",
        "прыжк",
        "борьб",
        "ледов",
        "шашк",
        "парусн",
        "акробат",
        "синхронн",
        "дистан",
        "дорожк",
        "плав",
        "плов",
        "гребл",
        "копье",
        "фанат",
        "хулиган",
        "комментатор",
        "наставник",
        "арбитр",
        "рефери",
        "секундант",
        "легионер",
        "атлет",
        "спортсмен",
        "тренер",
        "чемпион",
        "призер",
        "квалификац",
        "гонка",
        "финал",
        "полуфинал",
        "фол",
        "штрафной",
        "камбэк",
        "овертайм",
        "буллит",
        "пенальти",
        "офсайд",
        "нокдаун",
        "нокаут",
        "хет-трик",
        "интер",
        "милан",
        "ростов",
        "рубин",
        "сибирь",
        "адмирал",
        "металлург",
        "трактор",
        "торпедо",
        "автомобилист",
        "барыс",
        "краснодар",
        "арсенал",
        "челси",
        "бавария",
        "манчестер",
        "ливерпуль",
        "барселона",
        "ювентус",
        "альянс",
        "оборона",
        "отступлен",
        "окружение",
        "котел",
        "окоп",
        "прилет",
        "отработк",
        "подавлен",
        "зачистк",
        "передов",
        "передок",
        "сирена",
        "обломк",
        "хлопок",
        "взрыв",
        "залп",
        "ранен",
        "ликвидир",
        "уничтож",
        "поражен",
        "полигон",
        "учения",
        "маневры",
        "эшелон",
        "учебка",
        "бомб",
        "битв",
        "сражен",
        "фронт",
        "наступлен",
        "контрнаступ",
        "штурм",
        "погиб",
        "повестк",
        "град",
        "смерч",
        "ураган",
        "бук",
        "тор",
        "оса",
        "тигр",
        "тайфун",
        "калибр",
        "герань",
        "орлан",
        "зала",
        "собор",
        "икон",
        "пасх",
        "рождеств",
        "молитв",
        "монах",
        "приход",
        "секта",
        "кирилл",
        "диагноз",
        "авария",
        "кома",
        "венок",
        "гроб",
        "прощан",
        "смерт",
        "умер",
        "скончал",
        "наезд",
        "терминал",
        "вылет",
        "багаж",
        "самолет",
        "ангар",
        "победа",
        "прокат",
        "павильон",
        "сериал",
        "аншлаг",
        "хайп",
        "стрим",
        "донат",
        "лайк",
        "мем",
        "бренд",
        "бутик",
        "подиум",
        "дефиле",
        "ботокс",
        "филлер",
        "аниме",
        "манга",
        "комикс",
        "сель",
        "лавин",
        "оползн",
        "обвал",
        "метео",
        "осадк",
        "мороз",
        "смог",
        "гроз",
        "гром",
        "молни",
        "циклон",
        "катаклизм",
        "маникюр",
        "педикюр",
        "стрижк",
        "кардио",
        "массаж",
        "саун",
        "баня",
        "седан",
        "купе",
        "пикап",
        "робот",
        "бензин",
        "дизель",
        "гибрид",
        "батарея",
        "дилер",
        "лизинг",
        "каско",
        "осаго",
        "флагман",
        "процессор",
        "корпус",
        "хостел",
        "круиз",
        "сафари",
        "ялта",
        "байкал",
        "алтай",
        "виз",
        "шенген",
        "гид",
        "сувенир",
        "самокат",
        "лавка",
        "виски",
        "суши",
        "роллы",
        "таро",
        "ведьма",
        "карма",
        "аура",
        "транс",
        "фишки",
        "фора",
        "ничья",
        "котенок",
        "щенок",
        "питомец",
        "ветеринар",
        "груминг",
        "гештальт",
        "мотивац",
        "ипотек",
        "риелтор",
        "циан",
        "пик",
        "ламинат",
        "паркет",
        "обои",
        "лофт",
        "санузел",
        "домофон",
        "мвд",
        "мчс",
        "омон",
        "путин",
        "фсб",
        "азов",
        "изюм",
        "лиман",
        "сумы",
        "львов",
        "воронеж",
    }
)

# Bare leisure toponyms — drop outside WAR; WAR uses FRONTLINE allowlist instead.
LEISURE_TOPONYMS = {
    "сочи",
    "крым",
    "ялта",
    "алушта",
    "судак",
    "евпатория",
    "феодосия",
    "анапа",
    "геленджик",
    "дивноморск",
    "байкал",
    "алтай",
    "камчатка",
    "мальдивы",
    "бали",
    "пхукет",
    "самуи",
    "гоа",
    "дубай",
    "абхазия",
}

WAR_FRONTLINE_TOPONYMS = {
    "курск",
    "белгород",
    "брянск",
    "шебекино",
    "грайворон",
    "мариуполь",
    "бахмут",
    "артемовск",
    "авдеевк",
    "угледар",
    "соледар",
    "купянск",
    "изюм",
    "лиман",
    "херсон",
    "николаев",
    "одесс",
    "харьков",
    "сумы",
    "чернигов",
    "киев",
    "львов",
    "запорож",
    "мелитополь",
    "энергодар",
    "геническ",
    "севастополь",
    "донбасс",
    "новоросс",
    "азов",
}

TRANSPORT_DROP = {
    "электричк",
    "сапсан",
    "ласточка",
    "плацкарт",
    "купе",
    "ржд",
    "метрополитен",
    "пробк",
    "затор",
    "дпс",
    "гибдд",
}

CRIME_DROP = {
    "маньяк",
    "серийн убийц",
    "насильник",
    "педофил",
    "похищен",
    "заложник",
}

ENGLISH_JUNK = {"подтоплен dwelling", "вулканологическ observatori", "таurus"}

# Map draft assignment names → category file ids (index order later).
DRAFT_TO_CATEGORY = {
    "_MANUAL_WAR_TERMS": "WAR_OPERATIONAL",
    "_MANUAL_WAR_GENERIC_TERMS": "WAR",
    "_MANUAL_SPORT_TERMS": "SPORT_BLOCKED",
    "_MANUAL_AIRPORT_TERMS": "AIRPORT",
    "_MANUAL_RELIGION_TERMS": "RELIGION",
    "_MANUAL_DEATH_TERMS": "DEATH",
    "_MANUAL_FIRE_TERMS": "FIRE",
    "_MANUAL_CINEMA_TERMS": "CINEMA",
    "_MANUAL_MUSIC_TERMS": "MUSIC",
    "_MANUAL_SHOWBIZ_TERMS": "SHOWBIZ",
    "_MANUAL_BLOGGERS_TERMS": "SHOWBIZ",
    "_MANUAL_GOSSIP_TERMS": "SHOWBIZ",
    "_MANUAL_POP_CULTURE_TERMS": "POP_CULTURE",
    "_MANUAL_NATURAL_TERMS": "NATURAL_DISASTER",
    "_MANUAL_FASHION_TERMS": "FASHION",
    "_MANUAL_WELLNESS_TERMS": "WELLNESS",
    "_MANUAL_AUTO_TERMS": "AUTO",
    "_MANUAL_GADGETS_TERMS": "GADGETS",
    "_MANUAL_TRAVEL_TERMS": "TRAVEL",
    "_MANUAL_FOOD_TERMS": "FOOD",
    "_MANUAL_ASTROLOGY_TERMS": "ASTROLOGY",
    "_MANUAL_GAMBLING_TERMS": "GAMBLING",
    "_MANUAL_PETS_TERMS": "PETS",
    "_MANUAL_SELFHELP_TERMS": "SELFHELP",
    "_MANUAL_REALESTATE_TERMS": "REALESTATE",
}

CATEGORY_FILES = {
    "WAR_OPERATIONAL": "war_operational.yaml",
    "WAR": "war.yaml",
    "SPORT_BLOCKED": "sport.yaml",
    "AIRPORT": "airport.yaml",
    "RELIGION": "religion.yaml",
    "DEATH": "death.yaml",
    "FIRE": "fire.yaml",
    "TERRACT": "terract.yaml",
    "CINEMA": "cinema.yaml",
    "MUSIC": "music.yaml",
    "SHOWBIZ": "showbiz.yaml",
    "POP_CULTURE": "pop_culture.yaml",
    "NATURAL_DISASTER": "natural_disaster.yaml",
    "FASHION": "fashion.yaml",
    "WELLNESS": "wellness.yaml",
    "AUTO": "auto.yaml",
    "GADGETS": "gadgets.yaml",
    "TRAVEL": "travel.yaml",
    "FOOD": "food.yaml",
    "ASTROLOGY": "astrology.yaml",
    "GAMBLING": "gambling.yaml",
    "PETS": "pets.yaml",
    "SELFHELP": "selfhelp.yaml",
    "REALESTATE": "realestate.yaml",
}

INDEX_ORDER = list(CATEGORY_FILES.keys())

REASON_CODES = {
    "WAR_OPERATIONAL": "manual_war_operational_hard_block",
    "WAR": "manual_war_hard_block",
    "SPORT_BLOCKED": "manual_sport_hard_block",
    "AIRPORT": "manual_airport_hard_block",
    "RELIGION": "manual_religion_hard_block",
    "DEATH": "manual_death_hard_block",
    "FIRE": "manual_fire_hard_block",
    "TERRACT": "manual_wildberries_terract",
    "CINEMA": "manual_cinema_hard_block",
    "MUSIC": "manual_music_hard_block",
    "SHOWBIZ": "manual_showbiz_hard_block",
    "POP_CULTURE": "manual_pop_culture_hard_block",
    "NATURAL_DISASTER": "manual_natural_disaster_hard_block",
    "FASHION": "manual_fashion_hard_block",
    "WELLNESS": "manual_wellness_hard_block",
    "AUTO": "manual_auto_hard_block",
    "GADGETS": "manual_gadgets_hard_block",
    "TRAVEL": "manual_travel_hard_block",
    "FOOD": "manual_food_hard_block",
    "ASTROLOGY": "manual_astrology_hard_block",
    "GAMBLING": "manual_gambling_hard_block",
    "PETS": "manual_pets_hard_block",
    "SELFHELP": "manual_selfhelp_hard_block",
    "REALESTATE": "manual_realestate_hard_block",
}

# Seed terms that must exist for regression tests / current production coverage.
# Must not intersect src.core.safety.manual_terms_policy.AMBIGUITY_BANLIST.
SEED_TERMS: dict[str, list[str]] = {
    "WAR_OPERATIONAL": [
        "бпла",
        "дрон",
        "беспилотн",
        "авиационн",
        "опасност",
        "бойцы",
        "аэропорт приостановил",
        "всу",
        "сво",
    ],
    "WAR": [
        "война",
        "вторая мировая",
        "великой отечественной",
        "великая отечественная",
        "нацист",
        "боестолкнов",
    ],
    "SPORT_BLOCKED": [
        "гимнаст",
        "фигурист",
        "isu",
        "роднин",
        "спартак",
        "цска",
        "зенит",
        "локомотив",
        "динамо",
        "первая лига",
        "рпл",
        "кхл",
        "футбол",
        "хоккей",
        "теннис",
        "волейбол",
        "баскетбол",
        "чемпионат",
        "турнир",
        "биатлон",
        "олимп",
        "паралимп",
        "гто",
    ],
    "AIRPORT": ["аэропорт"],
    "RELIGION": ["храм"],
    "DEATH": ["останк", "труп"],
    "FIRE": ["пожар", "пожары"],
    "TERRACT": ["wildberries"],
}


def _parse_draft(text: str) -> dict[str, list[str]]:
    pattern = re.compile(
        r"^(_MANUAL_\w+)\s*=\s*\((.*?)\)\s*$",
        re.MULTILINE | re.DOTALL,
    )
    result: dict[str, list[str]] = {}
    for match in pattern.finditer(text):
        name = match.group(1)
        body = match.group(2)
        # Last assignment wins (draft has duplicates).
        try:
            values = ast.literal_eval("(" + body + ")")
        except (SyntaxError, ValueError):
            # Fallback: extract quoted strings.
            values = re.findall(r'"([^"]+)"', body)
        terms = [str(v).strip().casefold() for v in values if str(v).strip()]
        result[name] = terms
    return result


_NATURAL_KEEP_MARKERS = (
    "землетряс",
    "сейсм",
    "вулкан",
    "изверж",
    "наводнен",
    "паводок",
    "подтоплен",
    "цунами",
    "ураган",
    "тайфун",
    "торнадо",
    "смерч",
    "шторм",
    "лавин",
    "оползн",
    "селев",
    "сель ",
    "лесн пожар",
    "торфян",
    "стихийн",
    "катаклизм",
    "магнитуд",
    "эпицентр",
    "затоплен",
    "прорыв дамб",
    "прорыв плотин",
    "эвакуац населен",
    "чрезвычайн ситуаци природн",
    "росгидромет",
    "гидрометцентр",
)


def _should_drop(term: str, category: str) -> str | None:
    if term in ENGLISH_JUNK:
        return "english_junk"
    if term in CRIME_DROP:
        return "crime_omit"
    if term in TRANSPORT_DROP:
        return "transport_drop"
    if category != "WAR_OPERATIONAL" and term in LEISURE_TOPONYMS:
        return "leisure_toponym"
    if (
        category == "WAR_OPERATIONAL"
        and term in LEISURE_TOPONYMS
        and term not in WAR_FRONTLINE_TOPONYMS
    ):
        return "non_frontline_toponym"
    # Single toponym outside frontline allowlist for WAR_OPERATIONAL city-like tokens:
    if (
        category == "WAR_OPERATIONAL"
        and " " not in term
        and term
        in {
            "воронеж",
        }
    ):
        return "non_frontline_toponym"
    if category == "NATURAL_DISASTER" and not any(
        marker in term for marker in _NATURAL_KEEP_MARKERS
    ):
        return "generic_weather"
    letters = re.sub(r"[^0-9a-zа-яё]", "", term, flags=re.IGNORECASE)
    if len(letters) <= 3 and term not in ACRONYM_ALLOWLIST:
        return "too_short"
    if term in AMBIGUITY_BANLIST:
        if category == "WAR_OPERATIONAL" and term in WAR_FRONTLINE_TOPONYMS:
            return None
        if category in SEED_TERMS and term in {
            t.casefold() for t in SEED_TERMS[category]
        }:
            return None
        return "ambiguity_banlist"
    if term in SCRUB_EXTRA_BANLIST:
        if category in SEED_TERMS and term in {
            t.casefold() for t in SEED_TERMS[category]
        }:
            return None
        if category == "WAR_OPERATIONAL" and term in WAR_FRONTLINE_TOPONYMS:
            return None
        return "scrub_extra"
    if category == "GADGETS" and term in {"дрон", "квадрокоптер"}:
        return "war_owns_token"
    if category == "AIRPORT" and term in {"борт", "пассажир"}:
        return "ambiguity_banlist"
    if category == "AUTO" and term in {
        "танк",
        "патриот",
        "газ",
        "автомат",
        "механик",
        "лада",
        "tank",
        "patriot",
    }:
        return "ambiguity_banlist"
    if category == "ASTROLOGY" and term in {
        "рак",
        "лев",
        "дева",
        "весы",
        "овен",
        "телец",
        "близнецы",
        "скорпион",
        "стрелец",
        "козерог",
        "водолей",
        "рыбы",
    }:
        return "zodiac_single"
    return None


def main() -> None:
    draft_text = DRAFT.read_text(encoding="utf-8")
    parsed = _parse_draft(draft_text)

    buckets: dict[str, list[str]] = defaultdict(list)
    drop_log: list[tuple[str, str, str]] = []
    for draft_name, terms in parsed.items():
        category = DRAFT_TO_CATEGORY.get(draft_name)
        if not category:
            continue
        for term in terms:
            reason = _should_drop(term, category)
            if reason:
                drop_log.append((category, term, reason))
                continue
            buckets[category].append(term)

    # Seeds (may re-introduce precise production terms even if scrub-extra).
    for category, seeds in SEED_TERMS.items():
        for term in seeds:
            normalized = term.casefold()
            if normalized in AMBIGUITY_BANLIST:
                drop_log.append(
                    (category, normalized, "seed_blocked_by_policy_banlist")
                )
                continue
            buckets[category].append(normalized)

    # First-wins global dedup by INDEX_ORDER.
    seen: dict[str, str] = {}
    dup_log: list[tuple[str, str, str]] = []
    final: dict[str, list[str]] = {cid: [] for cid in INDEX_ORDER}
    for category in INDEX_ORDER:
        for term in buckets.get(category, []):
            if term in seen:
                dup_log.append((term, seen[term], category))
                continue
            seen[term] = category
            final[category].append(term)
        # Stable unique within category
        final[category] = sorted(set(final[category]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for category, filename in CATEGORY_FILES.items():
        path = OUT_DIR / filename
        payload = {"terms": final[category]}
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    index = {
        "categories": [
            {
                "id": cid,
                "file": CATEGORY_FILES[cid],
                "enabled": True,
                "decision": "hard_block",
                "reason_code": REASON_CODES[cid],
            }
            for cid in INDEX_ORDER
        ]
    }
    (OUT_DIR / "index.yaml").write_text(
        yaml.safe_dump(index, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (OUT_DIR / "overrides.yaml").write_text(
        yaml.safe_dump(
            {"force_include": [], "force_exclude": []},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    artifact = ARTIFACT_DIR / f"{stamp}-censor-terms-scrub.md"
    lines = [
        "# Censor terms scrub",
        "",
        f"Generated: {stamp} UTC",
        "",
        "## Kept counts",
        "",
    ]
    for cid in INDEX_ORDER:
        lines.append(f"- {cid}: {len(final[cid])}")
    lines.extend(["", "## Dropped (sample by reason)", ""])
    by_reason: dict[str, int] = defaultdict(int)
    for _c, _t, reason in drop_log:
        by_reason[reason] += 1
    for reason, count in sorted(by_reason.items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", f"## Duplicates ignored (first-wins): {len(dup_log)}", ""])
    for term, first, later in dup_log[:80]:
        lines.append(f"- `{term}` kept in {first}, ignored in {later}")
    if len(dup_log) > 80:
        lines.append(f"- ... and {len(dup_log) - 80} more")
    artifact.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR}")
    print(f"Artifact {artifact}")
    for cid in INDEX_ORDER:
        print(f"  {cid}: {len(final[cid])}")


if __name__ == "__main__":
    main()
