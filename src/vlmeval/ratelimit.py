"""Per-model rate limiting: concurrency semaphore + RPM sliding window."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Awaitable, Callable


class RateLimiter:
    """Async context manager. Caps in-flight requests and requests-per-minute.

    `clock` and `sleep` are injectable for deterministic tests.
    """

    def __init__(
        self,
        concurrency: int = 1,
        rpm: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._sem = asyncio.Semaphore(concurrency)
        self._rpm = rpm
        self._stamps: deque[float] = deque()
        self._clock = clock
        self._sleep = sleep
        self._window_lock = asyncio.Lock()

    async def __aenter__(self) -> "RateLimiter":
        await self._sem.acquire()
        if self._rpm:
            try:
                async with self._window_lock:
                    while True:
                        now = self._clock()
                        while self._stamps and self._stamps[0] <= now - 60.0:
                            self._stamps.popleft()
                        if len(self._stamps) < self._rpm:
                            break
                        await self._sleep(self._stamps[0] + 60.0 - now)
                    self._stamps.append(self._clock())
            except BaseException:
                self._sem.release()
                raise
        return self

    async def __aexit__(self, *exc) -> None:
        self._sem.release()
