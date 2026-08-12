"""Typed generation errors for degrade/retry decisions."""

from __future__ import annotations

from enum import Enum


class ErrorKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    EMPTY = "empty_response"
    INVALID = "invalid_response"


class GenerationError(Exception):
    """Raised (or wrapped) for recoverable/non-recoverable generation failures."""

    def __init__(self, message: str, *, kind: ErrorKind, cause: BaseException | None = None):
        super().__init__(message)
        self.kind = kind
        self.cause = cause

    @property
    def is_transient(self) -> bool:
        return self.kind == ErrorKind.TRANSIENT


def classify_exception(error: BaseException) -> ErrorKind:
    import asyncio

    import aiohttp

    if isinstance(error, GenerationError):
        return error.kind
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, aiohttp.ClientConnectorError)):
        return ErrorKind.TRANSIENT
    if isinstance(error, aiohttp.ServerDisconnectedError):
        return ErrorKind.TRANSIENT
    text = str(error).lower()
    if "empty choices" in text or "empty_response" in text or "empty model content" in text:
        return ErrorKind.EMPTY
    if "invalid" in text and "json" in text:
        return ErrorKind.INVALID
    if "429" in text or "http 5" in text or "timeout" in text or "temporarily" in text:
        return ErrorKind.TRANSIENT
    return ErrorKind.PERMANENT
