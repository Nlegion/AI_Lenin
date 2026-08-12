import re
import logging
from pathlib import Path
import pymorphy3
from symspellpy import SymSpell, Verbosity

logger = logging.getLogger(__name__)


class TextCleaner:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Определяем путь к словарю
        base_dir = Path(__file__).parent.parent.parent
        dict_paths = [
            base_dir / "data" / "dictionaries" / "ru-100k.txt",
            base_dir / "src" / "core" / "data" / "ru-100k.txt",
            Path(__file__).parent / "ru-100k.txt"
        ]

        # Пытаемся загрузить словарь
        dictionary_loaded = False
        for dict_path in dict_paths:
            if dict_path.exists():
                try:
                    self.sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
                    dictionary_loaded = True
                    logger.info(f"Словарь загружен из: {dict_path}")
                    break
                except Exception as e:
                    logger.error(f"Ошибка загрузки словаря {dict_path}: {str(e)}")

        if not dictionary_loaded:
            logger.warning(
                "Не удалось загрузить словарь для SymSpell. Будет использоваться базовая проверка орфографии.")

        # Специфические исправления для марксистской терминологии
        self.special_corrections = {
            "капиталистическихого": "капиталистического",
            "буржуази": "буржуазии",
            "класовые": "классовые",
            "класа": "класса",
            "эксплуатаци": "эксплуатации",
            "расмотрена": "рассмотрена",
            "санкци": "санкции",
            "подчеркаивает": "подчеркивает",
            "иноваци": "инновации",
            "иноваций": "инноваций",
            "Франци": "Франции",
            "Парти": "Партии",
            "регистраци": "регистрации",
            "авиаци": "авиации",
            "современого": "современного",
            "касационую": "кассационную",
            "расмотрит": "рассмотрит",
            "иновациям": "инновациям",
            "расматривать": "рассматривать",
            "клас": "класс",
            "даных": "данных",
            "информаци": "информации",
            "особено": "особенно",
            "мошеники": "мошенники",
            "класическим": "классическим",
            "государственые": "государственные",
            "бесрочно": "бессрочно",
            "Нацгварди": "Нацгвардии",
            "размещени": "размещения",
            "истиными": "истинными",
            "военое": "военное",
            "капиталист": "капиталистов",
            "совремная": "современная",
        }

    def correct_spelling(self, text: str) -> str:
        """Корректирует орфографические ошибки в тексте"""
        # Сначала применяем специальные исправления
        for wrong, correct in self.special_corrections.items():
            text = re.sub(r'\b' + wrong + r'\b', correct, text)

        # Разбиваем текст на слова и проверяем каждое
        words = re.findall(r'\w+|\S+', text)
        corrected_words = []

        for word in words:
            if not word.isalpha():
                corrected_words.append(word)
                continue

            # Проверяем, есть ли слово в специальных исправлениях
            if word.lower() in self.special_corrections:
                corrected_word = self.special_corrections[word.lower()]
                # Сохраняем регистр оригинала
                if word.istitle():
                    corrected_word = corrected_word.capitalize()
                elif word.isupper():
                    corrected_word = corrected_word.upper()

                corrected_words.append(corrected_word)
                continue

            # Используем pymorphy3 для проверки нормальной формы
            parsed = self.morph.parse(word)
            if parsed and parsed[0].score > 0.5:
                # Слово распознано, используем его
                corrected_words.append(word)
            else:
                # Пытаемся исправить с помощью SymSpell
                suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    corrected_words.append(suggestions[0].term)
                else:
                    corrected_words.append(word)

        return ' '.join(corrected_words)

    def truncate_to_last_complete_sentence(self, text: str) -> str:
        """Обрезает текст до последнего законченного предложения"""
        if not text:
            return text

        # Ищем последнюю точку, восклицательный или вопросительный знак
        for i in range(len(text) - 1, -1, -1):
            if text[i] in '.!?':
                return text[:i + 1]

        # Если не нашли знаков препинания, возвращаем как есть
        return text

    def clean_text(self, text: str) -> str:
        """Полная очистка текста с исправлением опечаток"""
        if not text:
            return text

        # Корректируем орфографию
        text = self.correct_spelling(text)

        # Long typo fixes only (word-boundary). Never map short tokens like "Рё"→"ФРГ"
        # (SymSpell can turn possessive "её" into "Рё", which previously became "ФРГ").
        political_corrections = {
            "капиталистическихого": "капиталистического",
            "буржуази": "буржуазии",
        }

        for wrong, correct in political_corrections.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text)

        # Удаляем шаблонные фразы
        patterns = [
            r'Анализ новости с марксистско-ленинской точки зрения[:]?',
            r'Теперь[^.!?]*[.!?]', r'Рассмотрим[^.!?]*[.!?]',
            r'Анализируя[^.!?]*[.!?]', r'можно сделать вывод[^.!?]*[.!?]',
            r'данная ситуация[^.!?]*[.!?]', r'В контексте[^.!?]*[.!?]',
            r'Как отмечал[^.!?]*[.!?]', r'С точки зрения[^.!?]*[.!?]'
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text