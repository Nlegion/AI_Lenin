# Legal Disclaimer / Юридический дисклеймер

Last updated: 2026-08-15

This document is a notice, not a click-wrap contract. Using the software does not create an agreement beyond the MIT license terms in [LICENSE](LICENSE).

---

## English

### 1. Nature of the project

**AI_Lenin** is an educational / research tool. It uses natural-language processing and generative AI to imitate analytic style associated with V.I. Lenin, based on published works in a local research corpus.

The project is **not**:

- an official voice of any political party, movement, or state;
- a source of reliable information about current events;
- a call to any action (political, economic, military, or otherwise);
- a commercial product or affiliation with political, state, or commercial organizations.

### 2. Generated content

- All answers are produced by artificial intelligence and may contain errors, inaccuracies, or fiction.
- Generated text is not a call to action and must not be treated as an official position of the project authors or any organization.
- Users must verify facts independently and must not rely on system conclusions as authoritative truth.
- Responsibility for how generated content is used rests with the end user / instance operator.

### 3. Research corpus

- The corpus is **local** and **not distributed** with this repository (`/data/` is gitignored).
- Lenin materials follow the Soviet **Complete Works (PSS), 5th edition, 55 volumes** (Politizdat, Moscow, 1967; Institute of Marxism-Leninism under the CPSU Central Committee), as stated in the local volume-1 front matter.
- Local trees include an original-like path (`intellectual/Ленин/pss/`) and a cleaned derivative (`ultimate_cleaned_ontology/…`), plus Marx/Engels and other files used for stance-typed retrieval. Stance labels in config are classification for RAG, not a rights inventory.
- Digitization host / dataset URL is **not recorded** in this repository.
- Lenin’s own writings (d. 1924) are generally public domain in many jurisdictions. **Whole PSS volume files are not claimed as public domain**: prefaces, notes, indexes, pagination, and other editorial apparatus of the Soviet edition may remain protected. Do not assume PD status for translations, third-party authors, or registry `author` labels (path names are not legal attribution).

### 4. Software license and third parties

- Repository-authored code and documentation are licensed under the MIT License ([LICENSE](LICENSE)). That license does **not** relicense the corpus, news feeds, model weights, or third-party packages.
- Models, libraries, and external services have their own terms ([THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), [NOTICE](NOTICE)).
- Optional **DeepSeek** API, **Telegram** Bot API, and **TASS** RSS are external services. API keys and compliance with their terms of service are the operator’s responsibility. Secrets must not be committed. Telegram posts that quote a news title and a TASS link do not relicense the feed.

### 5. Intended use

The project is intended for lawful educational and research purposes related to historical texts and NLP methods. Operators are responsible for complying with applicable law in their jurisdiction.

---

## Русский

### 1. Характер проекта

**AI_Lenin** — образовательный / исследовательский инструмент. Он использует обработку естественного языка и генеративный ИИ для имитации аналитического стиля, связанного с В.И. Лениным, на основе опубликованных трудов в локальном исследовательском корпусе.

Проект **не является**:

- официальным голосом какой-либо политической партии, движения или государства;
- источником достоверной информации о текущих событиях;
- призывом к каким-либо действиям (политическим, экономическим, военным и иным);
- коммерческим продуктом и не аффилирован с политическими, государственными или коммерческими структурами.

### 2. Сгенерированный контент

- Все ответы созданы искусственным интеллектом и могут содержать ошибки, неточности или вымысел.
- Сгенерированный текст не является призывом к действию и не может считаться официальной позицией авторов проекта или какой-либо организации.
- Пользователь обязан самостоятельно проверять факты и не полагаться на выводы системы как на истину в последней инстанции.
- Ответственность за использование сгенерированного контента лежит на конечном пользователе / операторе инстанса.

### 3. Исследовательский корпус

- Корпус **локальный** и **не распространяется** вместе с этим репозиторием (`/data/` в `.gitignore`).
- Материалы Ленина соответствуют советскому **Полному собранию сочинений (ПСС), 5-е издание, 55 томов** (Политиздат, Москва, 1967; Институт марксизма-ленинизма при ЦК КПСС) — по выходным данным локального тома 1.
- Локально присутствуют дерево, близкое к исходной оцифровке (`intellectual/Ленин/pss/`), и очищенная производная копия (`ultimate_cleaned_ontology/…`), а также Маркс/Энгельс и другие файлы для stance-typed retrieval. Метки stance в конфигурации — классификация для RAG, не опись прав.
- URL оцифровки / датасета в репозитории **не зафиксирован**.
- Собственные тексты Ленина (ум. 1924) в большинстве юрисдикций обычно являются общественным достоянием. **Файлы томов ПСС целиком не объявляются общественным достоянием**: предисловия, примечания, указатели, пагинация и иной аппарат советского издания могут оставаться охраняемыми. Не предполагайте PD для переводов, чужих авторов и меток `author` в реестре (имя пути — не юридическая атрибуция).

### 4. Лицензия ПО и третьи стороны

- Авторский код и документация репозитория распространяются по лицензии MIT ([LICENSE](LICENSE)). Эта лицензия **не** перелицензирует корпус, новостные ленты, веса моделей и сторонние пакеты.
- Модели, библиотеки и внешние сервисы подчиняются своим условиям ([THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), [NOTICE](NOTICE)).
- Опциональный API **DeepSeek**, Bot API **Telegram** и RSS **ТАСС** — внешние сервисы. Ключи и соблюдение их ToS — ответственность оператора. Секреты нельзя коммитить. Посты в Telegram с заголовком новости и ссылкой на ТАСС не перелицензируют ленту.

### 5. Назначение

Проект предназначен для законных образовательных и исследовательских задач, связанных с историческими текстами и методами NLP. Оператор обязан соблюдать применимое право своей юрисдикции.
