import os
import re
import shutil
import logging
from pathlib import Path
from typing import List, Set, Dict
import unicodedata
from collections import Counter

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/logs/ultimate_cleaning.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UltimateOntologyCleaner:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

        # Ультра-специфичные паттерны для советских изданий
        self.patterns = [
            # 1. ИЗДАТЕЛЬСКАЯ ИНФОРМАЦИЯ (полные блоки)
            r'ПЕЧАТАЕТСЯ\s+ПО\s+ПОСТАНОВЛЕНИЮ.*?ЦК\s+КПСС',
            r'ИНСТИТУТ\s+МАРКСИЗМА-ЛЕНИНИЗМА.*?ПРИ\s+ЦК\s+КПСС',
            r'ПОЛНОЕ\s+СОБРАНИЕ\s+СОЧИНЕНИЙ.*?ИЗДАНИЕ\s+(ПЯТОЕ|ТРЕТЬЕ|ЧЕТВЕРТОЕ)',
            r'ГОСУДАРСТВЕННОЕ\s+ИЗДАТЕЛЬСТВО.*?ПОЛИТИЧЕСКОЙ\s+ЛИТЕРАТУРЫ',
            r'ИЗДАТЕЛЬСТВО\s+ПОЛИТИЧЕСКОЙ\s+ЛИТЕРАТУРЫ',
            r'МОСКВА[·\s]*\d{4}',
            r'Пролетарии\s+всех\s+стран,\s+соединяйтесь!*',

            # 2. ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ (полные строки)
            r'^Тираж.*\d+.*$',
            r'^Цена.*\d+.*$',
            r'^©.*$',
            r'^ISBN.*$',
            r'^Заведующий\s+редакцией.*$',
            r'^Редактор.*$',
            r'^Художественный\s+редактор.*$',
            r'^Технический\s+редактор.*$',
            r'^Корректор.*$',
            r'^Сдано\s+в\s+набор.*$',
            r'^Подписано\s+к\s+печати.*$',

            # 3. НОМЕРА ТОМОВ И СТРАНИЦ (агрессивно)
            r'Том\s*\d+',
            r'том\s*\d+',
            r'ТОМ\s*\d+',
            r'стр\.\s*\d+',
            r'с\.\s*\d+',
            r'—\s*\d+\s*—',
            r'Страница\s*\d+',
            r'Глава\s*[IVXLCDM]+',
            r'ГЛАВА\s*[IVXLCDM]+',

            # 4. OCR-АРТЕФАКТЫ
            r'\b[А-Яа-я]+0[А-Яа-я]+\b',
            r'\b[А-Яа-я]+1[А-Яа-я]+\b',
            r'\bр\s*е\s*д\s*\.',
            r'[А-Яа-я]+-\s*[А-Яа-я]+',

            # 5. СЛУЖЕБНЫЕ ПОМЕТКИ
            r'Прим\.\s*ред\.',
            r'Nota\s*bene',
            r'Ред\.',
            r'sic!?',
            r'Заметьте\.\s*Ред\.',

            # 6. ФОРМАТИРОВАНИЕ
            r'\*{3,}',
            r'·{3,}',
            r'—{3,}',
            r'_{3,}',
        ]

        # Дополнительные паттерны для многоуровневой очистки
        self.secondary_patterns = [
            r'^\s*[IVXLCDM]+\s*$',
            r'^\s*\d+\s*$',
            r'^\.\.\.\s*$',
            r'^-+\s*$',
            r'^_{3,}\s*$',
            r'^\s*$',
        ]

        # Ключевые слова для сохранения
        self.patterns_to_keep = [
            r'.{80,}',  # Сохранять строки длиннее 80 символов
            r'.*[а-яА-Я]{8,}.*',  # Сохранять строки с кириллическими словами
            r'.*[a-zA-Z]{8,}.*',  # Сохранять строки с латинскими словами
            r'.*\d{4,}.*',  # Сохранять строки с длинными числами
        ]

        # Маркеры начала содержания
        self.content_start_markers = [
            r"^ГЛАВА\s+\d+",
            r"^РАЗДЕЛ\s+\d+",
            r"^ЧАСТЬ\s+\d+",
            r"^§\s*\d+",
            r"^Статья\s+\d+",
            r"^ВВЕДЕНИЕ",
            r"^ПРЕДИСЛОВИЕ",
            r"^СОДЕРЖАНИЕ",
            r"^ОГЛАВЛЕНИЕ",
        ]

    def normalize_text(self, text: str) -> str:
        """Нормализация текста"""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def remove_patterns(self, text: str) -> str:
        """Удаление шаблонов технической информации"""
        # Первый проход - основные паттерны
        for pattern in self.patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Второй проход - дополнительные паттерны
        for pattern in self.secondary_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)

        return text

    def is_technical_line(self, line: str) -> bool:
        """Проверяет, является ли строка технической информацией"""
        line = line.strip()

        if not line:
            return True

        # Проверяем на шаблоны сохранения
        for pattern in self.patterns_to_keep:
            if re.match(pattern, line, re.IGNORECASE):
                return False

        # Короткие строки без букв считаем техническими
        if len(line) < 15 and not any(c.isalpha() for c in line):
            return True

        return False

    def remove_repeating_blocks(self, text: str) -> str:
        """Удаляет повторяющиеся блоки текста"""
        lines = text.split('\n')

        # Находим часто повторяющиеся строки
        line_counter = Counter(lines)
        common_lines = {line for line, count in line_counter.items()
                        if count > len(lines) * 0.03 and len(line.strip()) > 3}

        if common_lines:
            cleaned_lines = []
            for line in lines:
                if line not in common_lines or len(line.strip()) > 40:
                    cleaned_lines.append(line)
            text = '\n'.join(cleaned_lines)

        return text

    def find_content_start(self, lines: List[str]) -> int:
        """Находит начало основного содержания в тексте"""
        content_started = False
        content_start_line = 0

        for i, line in enumerate(lines):
            line_clean = line.strip()

            if not content_started:
                # Проверяем маркеры начала содержания
                for marker in self.content_start_markers:
                    if re.match(marker, line_clean, re.IGNORECASE):
                        content_started = True
                        content_start_line = i
                        break

                # Также начинаем содержание при наличии длинного текста
                if not content_started and len(line_clean) > 80 and any(c.isalpha() for c in line_clean):
                    content_started = True
                    content_start_line = i

            if content_started:
                # Ищем реальное начало содержания (пропускаем технические строки)
                if len(line_clean) > 30 and any(c.isalpha() for c in line_clean):
                    return i

        return content_start_line

    def clean_text(self, text: str) -> str:
        """Очищает текст от технической информации"""
        # Нормализация текста
        text = self.normalize_text(text)

        # Удаление шаблонов
        text = self.remove_patterns(text)

        # Удаление повторяющихся блоков
        text = self.remove_repeating_blocks(text)

        lines = text.split('\n')

        # Находим начало основного содержания
        content_start = self.find_content_start(lines)

        # Обрабатываем только содержание
        content_lines = lines[content_start:]

        cleaned_lines = []
        for line in content_lines:
            if not self.is_technical_line(line):
                cleaned_lines.append(line)

        # Удаляем повторяющиеся пустые строки
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)

        return cleaned_text.strip()

    def process_file(self, source_file: Path, target_file: Path):
        """Обрабатывает один файл"""
        try:
            # Создаем целевую директорию
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Читаем исходный файл
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Очищаем текст
            cleaned_content = self.clean_text(content)

            # Сохраняем только если достаточно содержания
            if cleaned_content and len(cleaned_content) > 1000:
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

                # Получаем статистику
                original_size = len(content)
                cleaned_size = len(cleaned_content)
                reduction = (1 - cleaned_size / original_size) * 100 if original_size > 0 else 0

                logger.info(f"Успешно обработан: {source_file} -> {target_file} "
                            f"({original_size} → {cleaned_size} символов, {reduction:.1f}% reduction)")
                return True
            else:
                logger.warning(f"Пропущен (мало содержания): {source_file}")
                return False

        except Exception as e:
            logger.error(f"Ошибка обработки {source_file}: {str(e)}")
            # Сохраняем оригинал в случае ошибки
            shutil.copy2(source_file, target_file)
            return False

    def process_directory(self):
        """Обрабатывает всю директорию"""
        logger.info(f"Начало обработки: {self.source_dir} -> {self.target_dir}")

        # Удаляем целевую директорию, если она существует
        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)
            logger.info("Удалена существующая целевая директория")

        # Создаем целевую директорию
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Находим все текстовые файлы
        text_files = list(self.source_dir.rglob("*.txt"))
        logger.info(f"Найдено {len(text_files)} текстовых файлов")

        # Обрабатываем файлы
        processed_count = 0
        success_count = 0

        for i, source_file in enumerate(text_files):
            if i % 10 == 0:
                logger.info(f"Обработано {i}/{len(text_files)} файлов")

            # Создаем путь к целевому файлу
            relative_path = source_file.relative_to(self.source_dir)
            target_file = self.target_dir / relative_path

            if self.process_file(source_file, target_file):
                success_count += 1
            processed_count += 1

        logger.info(f"Обработка завершена. Успешно обработано {success_count}/{processed_count} файлов.")


def main():
    # Пути к директориям
    source_dir = r"/data/books/intellectual"
    target_dir = r"/data/books/ultimate_cleaned_ontology"

    # Проверяем существование исходной директории
    if not Path(source_dir).exists():
        logger.error(f"Исходная директория не существует: {source_dir}")
        return

    # Создаем и запускаем очиститель
    cleaner = UltimateOntologyCleaner(source_dir, target_dir)
    cleaner.process_directory()


if __name__ == "__main__":
    main()