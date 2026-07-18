from unittest.mock import patch

import pytest

from src.core.processor import NewsProcessor


@pytest.mark.asyncio
async def test_processor_initialization_defaults():
    def _close_created_coroutine(coroutine):
        coroutine.close()
        return None

    with patch("src.core.processor.asyncio.create_task", side_effect=_close_created_coroutine) as create_task_mock:
        processor = NewsProcessor()

    assert processor.fetch_interval == 300
    assert processor.stats["news_processed"] == 0
    assert "news_fetched" in processor.stats
    assert processor.analyzer_ready.is_set() is False
    assert callable(getattr(processor, "start_separated_processing", None))
    create_task_mock.assert_called_once()