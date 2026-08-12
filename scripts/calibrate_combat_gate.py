"""Combat / indirect labeled cases for gate calibration (target F1 ≥ 0.90)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.safety.news_guard import NewsGuard, load_news_guard_config  # noqa: E402

CASES = [
    # combat → deny
    {"id": "c01", "label": "combat", "title": "Сводка", "content": "ВС РФ поразили склад боеприпасов; военные сообщили об обстреле."},
    {"id": "c02", "label": "combat", "title": "Фронт", "content": "Артиллерийский обстрел позиций ВСУ на фронте продолжается."},
    {"id": "c03", "label": "combat", "title": "Удар", "content": "Ракетный удар по военным складам; войска ведут боевые действия."},
    {"id": "c04", "label": "combat", "title": "Миномёты", "content": "Миномётный обстрел линии соприкосновения; военные подразделения РФ."},
    {"id": "c05", "label": "combat", "title": "Авиаудар", "content": "Авиаудар по позициям; фронт сместился на восток."},
    {"id": "c06", "label": "combat", "title": "СВО", "content": "В ходе СВО выполнены задачи по поражению целей."},
    {"id": "c07", "label": "combat", "title": "Мобилизация", "content": "Обсуждается мобилизация и министерство обороны."},
    {"id": "c08", "label": "combat", "title": "Росгвардия", "content": "Росгвардия усилила охрану объектов в зоне боевых действий."},
    {"id": "c09", "label": "combat", "title": "ВСУ", "content": "ВСУ получили удар; военные источники подтверждают потери."},
    {"id": "c10", "label": "combat", "title": "Армия РФ", "content": "Действия армии России и военных подразделений РФ в приграничье."},
    # indirect → allow
    {"id": "i01", "label": "indirect", "title": "Экспорт", "content": "Экспорт нефти через КТК вырос; порты увеличили перевалку."},
    {"id": "i02", "label": "indirect", "title": "Дипломатия", "content": "Переговоры о гуманитарной помощи и экономике без описания боя."},
    {"id": "i03", "label": "indirect", "title": "Санкции", "content": "Новые санкции затрагивают торговлю и банки."},
    {"id": "i04", "label": "indirect", "title": "Ритейл", "content": "Армия потребителей нанесла удар по скидкам; экономика ритейла."},
    {"id": "i05", "label": "indirect", "title": "Порты", "content": "Порты Черного моря увеличили перевалку зерна."},
    {"id": "i06", "label": "indirect", "title": "Инфляция", "content": "Инфляция и безработица выросли в отчётном квартале."},
    {"id": "i07", "label": "indirect", "title": "Забастовка", "content": "Работники отстаивают свои права на заводе; профсоюз требует зарплату."},
    {"id": "i08", "label": "indirect", "title": "ИИ", "content": "Регулирование искусственного интеллекта обсуждают в парламенте."},
    {"id": "i09", "label": "indirect", "title": "Школа", "content": "Образование: новые стандарты школ и университетов."},
    {"id": "i10", "label": "indirect", "title": "Экология", "content": "Экология: выбросы предприятий и контроль загрязнения."},
    # sport FP → skip/allow not combat
    {"id": "s01", "label": "indirect", "title": "Теннис", "content": "Игрок нанёс удар справа и выиграл матч."},
    {"id": "s02", "label": "indirect", "title": "Футбол", "content": "Чемпионат: гол после удара головой."},
    {"id": "s03", "label": "indirect", "title": "Погода", "content": "Удар стихии повредил крыши; прогноз погоды на неделю."},
    {"id": "s04", "label": "indirect", "title": "Транспорт", "content": "Экспорт и транспортные коридоры расширяются."},
    {"id": "s05", "label": "indirect", "title": "АЭС", "content": "Национальная компания строит АЭС; экономика энергетики."},
    {"id": "s06", "label": "indirect", "title": "Палео", "content": "На Земле Франца Иосифа нашли окаменелости динозавра."},
    {"id": "s07", "label": "indirect", "title": "Метро", "content": "Город открыл станцию метро на окраине."},
    {"id": "s08", "label": "indirect", "title": "Торговля", "content": "Международная торговля зерном выросла."},
    {"id": "s09", "label": "indirect", "title": "Зарплата", "content": "Индексация зарплат бюджетников обсуждается правительством."},
    {"id": "s10", "label": "indirect", "title": "Гуманитарка", "content": "Гуманитарная помощь доставлена в регион; дипломатия продолжается."},
]


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> int:
    guard = NewsGuard(config=load_news_guard_config(path=Path("config/news_guard.yaml")))
    # combat class: predict deny
    tp = fp = fn = 0
    # indirect: predict not deny
    itp = ifp = 0
    for case in CASES:
        decision = guard.evaluate_input(
            title=case["title"],
            content=case["content"],
            source="TASS",
        ).decision
        is_deny = decision == "deny"
        if case["label"] == "combat":
            if is_deny:
                tp += 1
            else:
                fn += 1
        else:
            if is_deny:
                fp += 1
                ifp += 1
            else:
                itp += 1
    # indirect recall: non-deny among indirect labels
    indirect_total = sum(1 for c in CASES if c["label"] != "combat")
    combat_f1 = _f1(tp, fp, fn)
    indirect_f1 = _f1(itp, ifp, indirect_total - itp)
    payload = {
        "combat_f1": combat_f1,
        "indirect_f1": indirect_f1,
        "combat_tp_fp_fn": [tp, fp, fn],
        "n_cases": len(CASES),
        "target_f1": 0.90,
        "pass": combat_f1 >= 0.90 and indirect_f1 >= 0.90,
    }
    out = Path(".cursor/artifacts/safety/combat_calib_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fixture = Path("tests/fixtures/quality/combat_calib_30.jsonl")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    with fixture.open("w", encoding="utf-8") as handle:
        for case in CASES:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
