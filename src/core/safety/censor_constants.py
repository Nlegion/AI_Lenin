"""Static regexes and term lists for pre-RAG censorship heuristics."""

from __future__ import annotations

import re

_RU_CHAR_RE = re.compile(r"[а-яёА-ЯЁ]")
_ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
_SPORT_TOKEN_RE = re.compile(
    r"\b(спорт|футбол|хоккей|теннис|волейбол|баскетбол|матч|турнир|чемпионат|"
    r"атлет|спортсмен|тренер|федерац|лига|кубок|олимп|паралимп|гто)\w*\b",
    re.IGNORECASE,
)
_SPORT_TEAM_RE = re.compile(
    r"\b(спартак|цска|зенит|локомотив|динамо|торпедо|ростов|рубин|крылья\s+советов|"
    r"ахмат|авангард|ак\s*барс|ска|трактор|металлург|салават\s+юлаев|"
    r"первая\s+лига|премьер-?лига|рпл|кхл|нхл)\b",
    re.IGNORECASE,
)
_AIRPORT_TEMPLATE_RE = re.compile(
    r"(аэропорт\w*).{0,60}(временн\w*\s+ограничени\w*|ограничени\w*|возобновил\w*\s+работ)",
    re.IGNORECASE,
)
_SPECULATIVE_TERMS = (
    "оценил",
    "шансы",
    "прогноз",
    "вероятность",
    "ожидается",
    "может",
    "угроза",
    "ужесточение",
    "обсудил",
    "предположил",
)
_CRISIS_TERMS = (
    "атака",
    "бпла",
    "дрон",
    "взрыв",
    "хлопок",
    "пожар",
    "чп",
    "авария",
    "пострадав",
    "погиб",
    "эвакуац",
    "опасност",
    "закрыт",
    "рейс",
    "угроза",
    "тревог",
)
_ETHNO_HATE_HARD_TERMS = (
    "русопет",
    "чурк",
    "хач",
    "черножоп",
    "узкоглаз",
    "инородц",
    "нацмен",
    "малоросс",
)
_ETHNO_HATE_ACTION_TERMS = (
    "ненав",
    "убива",
    "изгна",
    "депорт",
    "очист",
    "запрет",
    "уничтож",
)
_SEPARATOR_RE = re.compile(r"[\s\-\._:,;!?/\\|()\[\]{}\"'`~]+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\uFEFF]")
_CATEGORY_ALIASES = {
    "airport": "AIRPORT",
    "religion": "RELIGION",
    "death": "DEATH",
    "war": "WAR",
    "fire": "FIRE",
    "теракт": "TERRACT",
    "терракт": "TERRACT",
    "cinema": "CINEMA",
    "music": "MUSIC",
    "showbiz": "SHOWBIZ",
    "pop_culture": "POP_CULTURE",
    "natural_disaster": "NATURAL_DISASTER",
    "fashion": "FASHION",
    "wellness": "WELLNESS",
    "auto": "AUTO",
    "gadgets": "GADGETS",
    "travel": "TRAVEL",
    "food": "FOOD",
    "astrology": "ASTROLOGY",
    "gambling": "GAMBLING",
    "pets": "PETS",
    "selfhelp": "SELFHELP",
    "realestate": "REALESTATE",
}

_DEFAULT_L2_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "DIPLOMACY": ("дипломатия", "переговоры", "посол", "международные отношения"),
    "SANCTIONS": ("санкции", "ограничения", "экспорт", "импорт", "торговля"),
    "MILITARY_OFFICIAL_STATEMENT": ("минобороны", "заявление", "брифинг", "официально"),
    "PROTESTS": ("митинг", "протест", "демонстрация", "забастовка"),
    "ETHNIC_RELIGIOUS": ("национальн", "этническ", "религиозн"),
    "SENSITIVE_POLITICS": ("власть", "оппозиция", "политическ"),
}
