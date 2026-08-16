import asyncio
import os

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.core.settings.config import Settings


def telegram_proxy_required_from_env() -> bool:
    raw = os.getenv("TELEGRAM_PROXY_REQUIRED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_telegram_proxy_url() -> str | None:
    """Return SOCKS/HTTP proxy URL for Telegram Bot API, or None if optional/unset."""
    url = os.getenv("TELEGRAM_PROXY_URL", "").strip()
    if telegram_proxy_required_from_env() and not url:
        raise ValueError(
            "TELEGRAM_PROXY_URL is required when TELEGRAM_PROXY_REQUIRED=true"
        )
    return url or None


class TelegramClient:
    def __init__(self, token: str = Settings.TELEGRAM_BOT_TOKEN):
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.logger = structlog.get_logger()
        self.token = token
        self.proxy_url = resolve_telegram_proxy_url()
        self.logger.info(
            "telegram_proxy_enabled",
            enabled=bool(self.proxy_url),
            required=telegram_proxy_required_from_env(),
        )

    def _client_kwargs(self) -> dict:
        kwargs: dict = {"timeout": 30.0, "trust_env": False}
        if self.proxy_url:
            kwargs["proxy"] = self.proxy_url
        return kwargs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
        reraise=True,
    )
    async def send_request(self, method: str, params: dict = None) -> dict:
        await asyncio.sleep(0.05)
        url = f"{self.base_url}/{method}"
        try:
            async with httpx.AsyncClient(**self._client_kwargs()) as client:
                response = await client.post(url, json=params)
                response.raise_for_status()
                data = response.json()

                if not data.get("ok"):
                    error = data.get("description", "Unknown error")
                    self.logger.error("Telegram API error", method=method, error=error)

                    if "invalid token" in error.lower():
                        raise PermissionError("Invalid Telegram token")
                    elif "too many requests" in error.lower():
                        raise RuntimeError("Too many requests")

                return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 5))
                self.logger.warning("Rate limited, retrying after", seconds=retry_after)
                await asyncio.sleep(retry_after)
                raise
        except Exception as e:
            self.logger.error("Telegram request failed", error=str(e))
            raise
