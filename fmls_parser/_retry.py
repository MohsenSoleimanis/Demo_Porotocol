"""Tiny retry helper for transient remote failures.

We retry only on network-level or 5xx errors — never on 4xx (those are
programming bugs, not transient).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

import httpx

log = logging.getLogger("fmls.retry")

T = TypeVar("T")


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError, httpx.PoolTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return 500 <= code < 600 or code == 429
    return False


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 8.0,
    op_name: str = "remote_call",
) -> T:
    """Call `fn()` with exponential backoff on transient errors."""
    last_exc: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as e:  # capture-then-decide
            if not is_transient(e) or i == attempts:
                raise
            last_exc = e
            delay = min(max_delay_s, base_delay_s * (2 ** (i - 1)))
            log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs",
                        op_name, i, attempts, type(e).__name__, delay)
            time.sleep(delay)
    assert last_exc is not None  # unreachable
    raise last_exc  # pragma: no cover
