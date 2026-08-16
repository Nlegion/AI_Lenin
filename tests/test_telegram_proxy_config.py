"""Unit tests for Telegram SOCKS proxy env gating (no live network)."""

from __future__ import annotations

import os

import pytest

from src.core.adapters.telegram import client as telegram_client


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_PROXY_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_PROXY_REQUIRED", raising=False)
    monkeypatch.setattr(telegram_client.Settings, "TELEGRAM_PROXY_URL", None)
    monkeypatch.setattr(telegram_client.Settings, "TELEGRAM_PROXY_REQUIRED", "false")


def test_resolve_proxy_optional_unset() -> None:
    assert telegram_client.resolve_telegram_proxy_url() is None


def test_resolve_proxy_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "TELEGRAM_PROXY_URL", "socks5://lenin:secret@host.docker.internal:1080"
    )
    assert (
        telegram_client.resolve_telegram_proxy_url()
        == "socks5://lenin:secret@host.docker.internal:1080"
    )


def test_required_without_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_PROXY_REQUIRED", "true")
    with pytest.raises(ValueError, match="TELEGRAM_PROXY_URL"):
        telegram_client.resolve_telegram_proxy_url()


def test_client_kwargs_trust_env_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_PROXY_URL", "socks5://127.0.0.1:1080")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy-token")
    instance = telegram_client.TelegramClient(token="dummy-token")
    kwargs = instance._client_kwargs()
    assert kwargs["trust_env"] is False
    assert kwargs["proxy"] == "socks5://127.0.0.1:1080"
