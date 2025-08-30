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
            self.ontology_path = BASE_DIR / "data" / "books" / "ultimate_cleaned_ontology"

        self.vector_db_path = BASE_DIR / "database" / "rag_db"

        # Создаем директории, если они не существуют
        os.makedirs(self.vector_db_path, exist_ok=True)

        # Инициализация моделей - ФИКС: используем CPU для избежания ошибок с meta tensor
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu"  # Принудительно используем CPU вместо автоматического выбора
        )

        # Инициализация ChromaDB
        self.chroma_client = PersistentClient(path=str(self.vector_db_path))
        self._init_collection()

        # Паттерны для фильтрации технической информации
        self.technical_patterns = [
            r'Москва,.*Миусская площадь',
            r'Ленинградская типография',
            r'Печатный Двор',
            r'Ордена Трудового Красного Знамени',
            r'Главполиграфпрома',
            r'Комитета по печати',
            r'Совета Министров',
            r'Гатчинская ул',
            r'ISBN',
            r'©',
            r'тираж',
            r'цена',
            r'редактор',
            r'корректор',
            r'Сдано в набор',
            r'Подписано к печати',
            r'Заведующий редакцией',
            r'Художественный редактор',
            r'Технический редактор',
            r'ББК',
            r'УДК',
            r'Научное издание',
            r'Редакционная коллегия',
            r'Ответственный редактор'
        ]

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
        """Извлекает имя автора из пути к файлу"""
        try:
            # Получаем относительный путь от корня онтологии
            relative_path = file_path.relative_to(self.ontology_path)
            parts = relative_path.parts

            # Специальная обработка для структуры онтологии
            if len(parts) >= 2:
                author = parts[0]

                # Обработка специальных случаев
                author_mapping = {
                    "pss": "Ленин",
                    "single": "Ленин",  # Для отдельных работ Ленина
                    "МарксЭнгельс": "Маркс и Энгельс"
                }

                if author in author_mapping:
                    return author_mapping[author]

                # Для остальных случаев возвращаем имя автора
                return author

            return "Unknown"
        except Exception as e:
            logger.error(f"Ошибка определения автора для {file_path}: {str(e)}")
            return "Unknown"

    def _is_technical_text(self, text: str) -> bool:
        """Проверяет, является ли текст технической информацией"""
        text_lower = text.lower()

        # Проверяем на наличие технических фраз
        for pattern in self.technical_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        # Проверяем на наличие большого количества цифр и специальных символов
        digit_ratio = len(re.findall(r'\d', text)) / len(text) if len(text) > 0 else 0
        special_char_ratio = len(re.findall(r'[\[\]{}()<>|\\/*+=#^$@%]', text)) / len(text) if len(text) > 0 else 0

        if digit_ratio > 0.1 or special_char_ratio > 0.05:
            return True

        # Короткие строки без букв считаем техническими
        if len(text) < 20 and not any(c.isalpha() for c in text):
            return True

        return False

    async def process_text_file(self, file_path: Path) -> List[Dict]:
        """Асинхронная обработка текстовых файлов"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()

            # Пропускаем технические файлы
            if "project_structure" in file_path.name.lower():
                return []

            # Получаем автора из пути
            author = self._get_author_from_path(file_path)

            # Адаптивное разбиение на чанки
            chunk_size = 800
            overlap = 150
            chunks = []

            # Семантическое разбиение на абзацы
            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 50]

            for paragraph in paragraphs:
                # Пропускаем технические абзацы
                if self._is_technical_text(paragraph):
                    continue

                if len(paragraph) <= chunk_size:
                    chunks.append({
                        "text": paragraph,
                        "source": file_path.name,
                        "author": author,
                        "work": file_path.stem
                    })
                else:
                    # Разбиваем длинные абзацы
                    for i in range(0, len(paragraph), chunk_size - overlap):
                        chunk = paragraph[i:i + chunk_size]
                        if len(chunk.strip()) > 100 and not self._is_technical_text(chunk):
                            chunks.append({
                                "text": chunk,
                                "source": file_path.name,
                                "author": author,
                                "work": file_path.stem
                            })
            return chunks
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {str(e)}")
            return []

    async def build_ontology_index(self):
        """Асинхронное построение индекса"""
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

    def filter_technical_results(self, results: List[str]) -> List[str]:
        """Фильтрует техническую информацию из результатов поиска"""
        filtered_results = []

        for result in results:
            # Проверяем, является ли результат технической информацией
            if not self._is_technical_text(result):
                # Дополнительная очистка от технических фрагментов внутри текста
                cleaned_result = result
                for pattern in self.technical_patterns:
                    cleaned_result = re.sub(pattern, '', cleaned_result, flags=re.IGNORECASE)

                # Удаляем строки с большим количеством цифр и специальных символов
                lines = cleaned_result.split('\n')
                cleaned_lines = []

                for line in lines:
                    digit_ratio = len(re.findall(r'\d', line)) / len(line) if len(line) > 0 else 0
                    special_char_ratio = len(re.findall(r'[\[\]{}()<>|\\/*+=#^$@%]', line)) / len(line) if len(
                        line) > 0 else 0

                    if digit_ratio < 0.1 and special_char_ratio < 0.05 and len(line.strip()) > 20:
                        cleaned_lines.append(line.strip())

                if cleaned_lines:
                    filtered_result = ' '.join(cleaned_lines)
                    if len(filtered_result) > 50:  # Минимальная длина содержательного текста
                        filtered_results.append(filtered_result)

            # Ограничиваем количество результатов
            if len(filtered_results) >= 5:
                break

        return filtered_results

    def retrieve_relevant_context(self, query: str, k: int = 7, author_filter: Optional[str] = None) -> str:
        """Улучшенный поиск контекста с приоритетом для цитат Ленина"""
        try:
            # Добавляем ключевые слова политэкономии к запросу
            political_economy_keywords = [
                "империализм", "капитал", "политэкономия", "колониализм",
                "международная торговля", "санкции", "финансовый капитал",
                "международные отношения", "дипломатия", "гегемония"
            ]

            enhanced_query = f"{query} {' '.join(political_economy_keywords)}"

            # Исправляем фильтр для ChromaDB
            where_filter = None
            if author_filter:
                where_filter = {"author": {"$eq": author_filter}}

            results = self.collection.query(
                query_texts=[enhanced_query],  # Используем улучшенный запрос
                n_results=k * 3,  # Берем больше результатов для фильтрации
                where=where_filter,
                include=["documents", "metadatas"]
            )

            # Фильтруем техническую информацию
            filtered_documents = []
            filtered_metadatas = []

            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    # Приоритет для цитат Ленина
                    metadata = results['metadatas'][0][i]
                    is_lenin = metadata.get('author') == 'Ленин'

                    if not self._is_technical_text(doc):
                        # Добавляем цитаты Ленина в начало списка
                        if is_lenin:
                            filtered_documents.insert(0, doc)
                            filtered_metadatas.insert(0, metadata)
                        else:
                            filtered_documents.append(doc)
                            filtered_metadatas.append(metadata)

            # Если после фильтрации осталось мало результатов, берем оригинальные
            if len(filtered_documents) < k:
                filtered_documents = results['documents'][0][:k]
                filtered_metadatas = results['metadatas'][0][:k]
            else:
                filtered_documents = filtered_documents[:k]
                filtered_metadatas = filtered_metadatas[:k]

            # Форматирование результатов с акцентом на цитаты
            context_parts = []
            for i, doc in enumerate(filtered_documents):
                metadata = filtered_metadatas[i]
                work_name = metadata['work'].replace('_', ' ').title()

                # Для цитат Ленина добавляем специальное форматирование
                if metadata['author'] == 'Ленин':
                    context_parts.append(
                        f"[ЦИТАТА ЛЕНИНА из '{work_name}']: {doc}"
                    )
                else:
                    context_parts.append(
                        f"[Из {metadata['author']} - '{work_name}']: {doc}"
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