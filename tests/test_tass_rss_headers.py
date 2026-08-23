from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.settings.config import Settings
from src.modules.news_system.fetcher import NewsFetcher


def _feed(*, status: int, entries: list, bozo: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        entries=entries,
        bozo=bozo,
        bozo_exception=None if not bozo else RuntimeError("html"),
    )


def _entry(*, link: str = "https://tass.ru/ekonomika/1") -> SimpleNamespace:
    item = SimpleNamespace(
        link=link,
        title="Заголовок",
        published_parsed=(2026, 8, 20, 3, 0, 0, 0, 0, 0),
    )
    item.get = lambda key, default=None: "лид" if key == "description" else default
    return item


def test_fetch_tass_sends_browser_user_agent(monkeypatch):
    captured: dict[str, object] = {}

    def fake_parse(url: str, request_headers: dict[str, str] | None = None):
        captured["url"] = url
        captured["headers"] = request_headers
        return _feed(status=200, entries=[_entry()])

    monkeypatch.setattr(
        "src.modules.news_system.fetcher.feedparser.parse",
        fake_parse,
    )
    items = NewsFetcher().fetch_tass()
    assert captured["url"] == Settings.TASS_RSS_URL
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["User-Agent"] == Settings.TASS_RSS_USER_AGENT
    assert headers["User-Agent"].startswith("Mozilla/5.0")
    assert len(items) == 1
    assert items[0]["source"] == "TASS"
    assert items[0]["url"] == "https://tass.ru/ekonomika/1"


def test_fetch_tass_http_403_returns_empty_and_logs(monkeypatch, caplog):
    monkeypatch.setattr(
        "src.modules.news_system.fetcher.feedparser.parse",
        MagicMock(return_value=_feed(status=403, entries=[], bozo=True)),
    )
    with caplog.at_level("ERROR"):
        items = NewsFetcher().fetch_tass()
    assert items == []
    assert "tass_rss_http_error" in caplog.text
    assert "403" in caplog.text
