import logging
import asyncio
from pathlib import Path
from src.core.rag_system import EnhancedRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Перестроение индекса с улучшенной фильтрацией технической информации...")

    # Указываем путь к очищенной онтологии
    ontology_path = r"/data/books/ultimate_cleaned_ontology"

    # Проверяем существование пути
    path = Path(ontology_path)
    if not path.exists():
        logger.error(f"Путь {ontology_path} не существует!")
        return

    # Удаляем старую базу данных
    vector_db_path = Path(r"/database/rag_db")
    if vector_db_path.exists():
        import shutil
        shutil.rmtree(vector_db_path)
        logger.info("Старая база данных удалена")

    # Создаем экземпляр RAG системы
    rag_system = EnhancedRAGSystem(ontology_path=ontology_path)

    # Строим индекс
    await rag_system.build_ontology_index()
    logger.info("Индекс успешно построен с улучшенной фильтрацией технической информации")


if __name__ == "__main__":
    asyncio.run(main())