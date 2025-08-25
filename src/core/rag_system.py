import os
import logging
import chromadb
import torch
import re
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient
from src.core.settings.config import Settings

logger = logging.getLogger(__name__)


class EnhancedRAGSystem:
    def __init__(self, ontology_path: str = None):
        self.config = Settings()
        BASE_DIR = Path(__file__).parent.parent.parent

        # Пути к данным
        if ontology_path:
            self.ontology_path = Path(ontology_path)
        else:
            self.ontology_path = BASE_DIR / "data" / "books" / "intellectual"

        self.vector_db_path = BASE_DIR / "database" / "rag_db"

        # Создаем директории, если они не существуют
        os.makedirs(self.vector_db_path, exist_ok=True)

        # Инициализация моделей
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Инициализация ChromaDB
        self.chroma_client = PersistentClient(path=str(self.vector_db_path))
        self._init_collection()

    def _init_collection(self):
        try:
            self.collection = self.chroma_client.get_collection(
                name="philosophy_ontology",
                embedding_function=self.embedding_function
            )
            logger.info("RAG коллекция успешно загружена")
        except Exception:
            logger.info("Создание новой RAG коллекции")
            self.collection = self.chroma_client.create_collection(
                name="philosophy_ontology",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

    def _get_author_from_path(self, file_path: Path) -> str:
        """Извлекает имя автора из пути к файлу с улучшенной логикой"""
        try:
            # Получаем относительный путь от корня онтологии
            relative_path = file_path.relative_to(self.ontology_path)
            parts = relative_path.parts

            # Первая папка - это автор
            if len(parts) > 0:
                author = parts[0]

                # Специальные случаи
                if author == "pss":
                    return "Ленин"
                elif author == "single":
                    # Для файлов в папке single, автор определяется по родительской папке
                    if len(parts) > 1:
                        return parts[1]
                    else:
                        return "Ленин"  # По умолчанию для папки single

                return author

            return "Unknown"
        except Exception:
            # Если не можем определить автора из пути, попробуем извлечь из содержимого файла
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)  # Читаем первые 1000 символов

                # Ищем метаданные об авторе в начале файла
                author_match = re.search(r'АВТОР:\s*([^\n]+)', content)
                if author_match:
                    return author_match.group(1).strip()

                # Ищем упоминания известных авторов в тексте
                known_authors = ["Ленин", "Маркс", "Энгельс", "Гегель", "Аристотель", "Плеханов", "Богданов"]
                for author in known_authors:
                    if author in content:
                        return author

            except Exception:
                pass

            return "Unknown"

    def _clean_text(self, text: str) -> str:
        """Очистка текста от служебной информации и колонтитулов"""
        # Удаляем техническую информацию о публикации
        patterns_to_remove = [
            r'Гатчинская ул., 26',
            r'Главполиграфпрома?',
            r'Комитета по печати',
            r'имени А\. М\. Горького',
            r'Совета Министров СССР',
            r'Заведующий редакцией',
            r'Редактор',
            r'Художник',
            r'Художественный редактор',
            r'Технический редактор',
            r'Корректор',
            r'Сдано в набор',
            r'Подписано к печати',
            r'Тираж',
            r'Цена',
            r'©',
            r'ISBN',
            r'том \d+',
            r'Том \d+',
            r'\b\d{1,3}\b',  # Одиночные числа (номера страниц)
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Удаляем повторяющиеся пробелы и пустые строки
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)

        return text.strip()

    def _is_content_text(self, text: str) -> bool:
        """Проверяет, является ли текст содержательным (а не техническим)"""
        if len(text) < 50:  # Слишком короткий текст
            return False

        # Проверяем на наличие технических фраз
        technical_phrases = [
            'тираж', 'цена', 'редактор', 'корректор',
            'сдано в набор', 'подписано к печати'
        ]

        text_lower = text.lower()
        for phrase in technical_phrases:
            if phrase in text_lower:
                return False

        # Проверяем, что текст содержит осмысленные слова
        word_count = len(re.findall(r'\b\w{3,}\b', text))
        if word_count < 5:  # Мало осмысленных слов
            return False

        return True

    async def process_text_file(self, file_path: Path) -> List[Dict]:
        """Асинхронная обработка текстовых файлов с улучшенной очисткой"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()

            # Пропускаем технические файлы
            if "project_structure" in file_path.name.lower():
                return []

            # Извлекаем автора из метаданных в начале файла
            author = "Unknown"
            work = file_path.stem

            # Ищем метаданные в начале файла
            metadata_match = re.search(r'АВТОР:\s*([^\n]+)\nРАБОТА:\s*([^\n]+)\n\n', content)
            if metadata_match:
                author = metadata_match.group(1).strip()
                work = metadata_match.group(2).strip()
                # Удаляем метаданные из содержимого
                content = content[metadata_match.end():]
            else:
                # Если метаданных нет, определяем автора из пути
                author = self._get_author_from_path(file_path)

            # Очистка текста от остаточной технической информации
            content = self._clean_text(content)

            # Адаптивное разбиение на чанки
            chunk_size = 800
            overlap = 150
            chunks = []

            # Семантическое разбиение на абзацы
            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 50]

            for paragraph in paragraphs:
                # Пропускаем технические абзацы
                if not self._is_content_text(paragraph):
                    continue

                if len(paragraph) <= chunk_size:
                    chunks.append({
                        "text": paragraph,
                        "source": file_path.name,
                        "author": author,
                        "work": work
                    })
                else:
                    # Разбиваем длинные абзацы
                    for i in range(0, len(paragraph), chunk_size - overlap):
                        chunk = paragraph[i:i + chunk_size]
                        if len(chunk.strip()) > 100 and self._is_content_text(chunk):
                            chunks.append({
                                "text": chunk,
                                "source": file_path.name,
                                "author": author,
                                "work": work
                            })
            return chunks
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {str(e)}")
            return []

    async def build_ontology_index(self):
        """Асинхронное построение индекса с прогресс-баром"""
        logger.info("Начало построения индекса онтологии...")

        # Проверяем существование пути
        if not self.ontology_path.exists():
            logger.error(f"Путь {self.ontology_path} не существует!")
            return

        # Рекурсивный поиск текстовых файлов
        text_files = list(self.ontology_path.rglob("*.txt"))
        logger.info(f"Найдено {len(text_files)} текстовых файлов")

        if not text_files:
            logger.error("Не найдено текстовых файлов для индексации!")
            return

        documents = []
        metadatas = []
        ids = []

        for i, file_path in enumerate(text_files):
            if i % 10 == 0:
                logger.info(f"Обработано {i}/{len(text_files)} файлов")

            chunks = await self.process_text_file(file_path)
            for j, chunk in enumerate(chunks):
                documents.append(chunk["text"])
                metadatas.append({
                    "source": chunk["source"],
                    "author": chunk["author"],
                    "work": chunk["work"]
                })
                ids.append(f"{chunk['author']}_{file_path.stem}_{j}")

        # Пакетное добавление с оптимизацией
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            logger.info(f"Добавлено {end_idx}/{len(documents)} документов")

        logger.info(f"Индекс онтологии построен. Всего документов: {len(documents)}")

    def retrieve_relevant_context(self, query: str, k: int = 5, author_filter: Optional[str] = None) -> str:
        """Улучшенный поиск контекста"""
        try:
            # Исправляем фильтр для ChromaDB
            where_filter = None
            if author_filter:
                where_filter = {"author": {"$eq": author_filter}}

            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas"]
            )

            # Форматирование результатов
            context_parts = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    # Пропускаем технические фрагменты
                    if self._is_content_text(doc):
                        context_parts.append(
                            f"[Из {metadata['author']} - {metadata['work']}]: {doc}"
                        )

            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Ошибка поиска контекста: {str(e)}")
            return ""


# Глобальный экземпляр RAG системы
rag_system = None


def get_rag_system():
    global rag_system
    if rag_system is None:
        rag_system = EnhancedRAGSystem()
    return rag_system