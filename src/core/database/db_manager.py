from typing import Any
from sqlalchemy import inspect
import structlog
from sqlalchemy import and_, delete, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select
from typing import Iterable, Type
from src.core.database.db_core import Base
from src.core.utils.decorators import handle_db_errors

logger = structlog.get_logger()


class AsyncDBManager:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()

    @handle_db_errors
    async def commit(self):
        await self._session.commit()

    @handle_db_errors
    async def rollback(self):
        await self._session.rollback()

    @handle_db_errors
    async def insert(self, instance: Base):
        self._session.add(instance)
        self.log_query(self._session.add(instance))
        await self.commit()

    @handle_db_errors
    async def upsert(self, instance: Base) -> None:
        try:
            mapper = inspect(type(instance))
            primary_keys = [key.name for key in mapper.primary_key]
            existing = await self._session.get(type(instance),
                                               {key: getattr(instance, key) for key in primary_keys})
            if existing:
                for key, value in instance.__dict__.items():
                    if not key.startswith('_'):
                        setattr(existing, key, value)
            else:
                self._session.add(instance)
        except Exception as e:
            await self._session.rollback()
            raise

    @handle_db_errors
    async def bulk_insert(self, model: type[Base], data: list[dict]) -> None:
        """
        Массовая вставка данных
        Пример: await bulk_insert(User, [{'name': 'John'}, {'name': 'Jane'}])
        """
        stmt = pg_insert(model).values(data)
        await self._session.execute(stmt)
        await self.commit()

    @handle_db_errors
    async def bulk_upsert(
            self,
            model: Type[Base],
            data: list[dict],
            conflict_columns: Iterable[str],
            batch_size: int = 1000
    ) -> None:
        """Улучшенный массовый upsert с учетом composite constraints"""
        if not data:
            logger.debug("No data provided for bulk upsert")
            return

            # Получаем таблицу из модели
        table = model.__table__

        # Анализ структуры модели
        composite_indexes = []
        if hasattr(table, 'indexes'):
            composite_indexes = [
                [col.name for col in index.columns]
                for index in table.indexes
                if index.unique and len(index.columns) > 1
            ]

        # Определение полей для уникальности
        unique_keys = set(conflict_columns)
        for index in composite_indexes:
            unique_keys.update(index)

        # Если не нашли composite индексов, используем первичный ключ
        if not unique_keys:
            unique_keys = [col.name for col in table.primary_key]

        # Остальная часть метода остается без изменений
        total = len(data)
        logger.debug(f"Total items to process: {total}")

        for index in range(0, total, batch_size):
            batch = data[index:index + batch_size]
            unique_records = {}

            for item in batch:
                # Формируем ключ уникальности
                key = tuple(item.get(key) for key in unique_keys)
                if None not in key and key not in unique_records:
                    unique_records[key] = item

            unique_batch = list(unique_records.values())
            logger.debug(f"Processing batch {index // batch_size + 1}, unique items: {len(unique_batch)}")

            # Проверка обязательных полей
            required_columns = [col.name for col in table.columns if not col.nullable and not col.autoincrement]
            for item in unique_batch:
                for col in required_columns:
                    if col not in item or item[col] is None:
                        logger.warning(f"Missing required column '{col}' in item: {item}")

            # Формирование запроса
            stmt = pg_insert(model).values(unique_batch)
            update_columns = {
                col.name: stmt.excluded[col.name]
                for col in table.columns
                if col.name not in unique_keys and not col.autoincrement
            }

            if update_columns:
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_keys,
                    set_=update_columns
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=unique_keys)

            try:
                await self._session.execute(stmt)
                await self._session.commit()
            except Exception as e:
                await self._session.rollback()
                logger.error(f"Batch failed: {str(e)}")
                raise

    @handle_db_errors
    async def bulk_delete(self, model: type[Base], ids: list[Any]) -> None:
        """
        Массовое удаление по списку ID
        Пример: await bulk_delete(User, [1, 2, 3])
        """
        stmt = delete(model).where(model.id.in_(ids))
        await self._session.execute(stmt)
        await self.commit()

    @handle_db_errors
    async def get_first_by_filter(self, model: type[Base], key: str, value: Any) -> Base | None:
        stmt = select(model).filter(getattr(model, key) == value)
        self.log_query(stmt)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    @handle_db_errors
    async def bulk_update(self, model: type[Base], data: list[dict], update_key: str = 'id') -> None:
        """
        Массовое обновление данных по ключу
        Пример: await bulk_update(User, [{'id': 1, 'name': 'John'}, {'id': 2, 'name': 'Jane'}])
        """
        stmt = update(model)
        await self._session.execute(
            stmt,
            [{'values': {k: v for k, v in item.items() if k != update_key},
              'sync_kwargs': {'synchronize_session': False}}
             for item in data]
        )
        await self.commit()

    @handle_db_errors
    async def get_all(self, model: type[Base]) -> list[Base]:
        stmt = select(model)
        self.log_query(stmt)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    @handle_db_errors
    async def get_all_by_filter(self, model: type[Base], key: str, value: Any) -> list[Base]:
        stmt = select(model).filter(getattr(model, key) == value)
        self.log_query(stmt)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    @handle_db_errors
    async def get_first(self, model: type[Base]) -> Base | None:
        stmt = select(model)
        self.log_query(stmt)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    @handle_db_errors
    async def get_first_by_conditions(
            self,
            model: type[Base],
            *conditions: Any,
            options: list = None  # Добавляем параметр options
    ) -> Base | None:
        stmt = select(model).where(and_(*conditions))

        # Добавляем опции, если они переданы
        if options:
            stmt = stmt.options(*options)

        self.log_query(stmt)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    @handle_db_errors
    async def delete_where(self, model: type[Base], *conditions: Any) -> None:
        stmt = delete(model).where(and_(*conditions))
        self.log_query(stmt)
        await self._session.execute(stmt)
        await self.commit()

    @handle_db_errors
    async def update(self, model: type[Base], attribute: str, data: dict) -> None:
        stmt = select(model).filter_by(id=attribute)
        self.log_query(stmt)
        result = await self._session.execute(stmt)
        obj = result.scalars().one()
        for key, value in data.items():
            setattr(obj, key, value)
        await self.commit()

    @handle_db_errors
    async def execute_(self, sql_command: str, params: dict | None = None) -> None:
        stmt = text(sql_command)
        self.log_query(stmt)
        await self._session.execute(text(sql_command), params or {})
        await self.commit()

    @handle_db_errors
    async def drop_table(self, model: type[Base]) -> None:
        stmt = model.__table__.drop()
        self.log_query(stmt)
        await self._session.execute(stmt)
        await self.commit()

    @handle_db_errors
    async def load_relationship(self, instance: Base, relationship: str) -> None:
        stmt = select(instance).options(selectinload(relationship))
        self.log_query(stmt)
        await self._session.execute(stmt)

    @handle_db_errors
    async def filter(self, model: type[Base], *conditions: Any, options=None) -> list[Base]:
        stmt = select(model).where(and_(*conditions))
        if options:
            stmt = stmt.options(*options)
        self.log_query(stmt)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    def log_query(self, statement: Select) -> None:
        logger = structlog.get_logger("sqlalchemy.query")
        try:
            compiled = statement.compile(
                dialect=self._session.bind.dialect,
                compile_kwargs={"literal_binds": True}
            )
            logger.debug("SQL query", query=str(compiled), params=compiled.params)
        except Exception as e:
            logger.error("Failed to log SQL query", error=str(e))
