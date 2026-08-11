"""Unified model interface: local and API models share the same async surface."""

from __future__ import annotations

import asyncio
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vlmeval.cache import CachedRow, ResponseCache
from vlmeval.config import ModelConfig, RunConfig
from vlmeval.ratelimit import RateLimiter


@dataclass(frozen=True)
class GenParams:
    max_tokens: int
    temperature: float = 0.0  # greedy everywhere

    def cache_dict(self) -> dict:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    source: str = "api"  # "api" | "tokenizer" (local models)


@dataclass
class ModelResponse:
    text: str
    usage: Usage
    latency_s: float
    cost_usd: float | None
    cached: bool
    error: str | None = None


def _row_to_response(row: CachedRow) -> ModelResponse:
    return ModelResponse(
        text=row.response_text,
        usage=Usage(row.input_tokens, row.output_tokens, row.usage_source),
        latency_s=row.latency_s,
        cost_usd=row.cost_usd,
        cached=True,
    )


class BaseModel(ABC):
    def __init__(self, cfg: ModelConfig, run_cfg: RunConfig, cache: ResponseCache):
        self.cfg = cfg
        self.run_cfg = run_cfg
        self.cache = cache
        self.use_cache = True
        rl = cfg.rate_limit
        self.limiter = RateLimiter(concurrency=rl.concurrency, rpm=rl.rpm)

    # -- cache key ---------------------------------------------------------

    def cache_params(self, params: GenParams) -> dict:
        """GenParams + image-prep policy: changing either must miss the cache."""
        d = params.cache_dict()
        d["image_max_side"] = self.run_cfg.image_max_side
        d["jpeg_quality"] = self.run_cfg.jpeg_quality
        return d

    def cache_key(
        self, task: str, sample_id: str, prompt: str, image_jpeg: bytes, params: GenParams
    ) -> str:
        return ResponseCache.make_key(
            self.cfg.id,
            task,
            sample_id,
            prompt,
            image_jpeg,
            self.cache_params(params),
        )

    # -- main entry point ----------------------------------------------------

    async def generate(
        self, image_jpeg: bytes, prompt: str, params: GenParams, *, task: str, sample_id: str
    ) -> ModelResponse:
        key = self.cache_key(task, sample_id, prompt, image_jpeg, params)
        if self.use_cache and (row := self.cache.get(key)) is not None:
            return _row_to_response(row)

        async with self.limiter:
            try:
                text, usage, latency = await self._call_with_retry(image_jpeg, prompt, params)
                cost_usd = self.compute_cost(usage, latency)
                self._validate_cost(cost_usd)
            except Exception as e:  # exhausted retries — recorded, never cached
                return ModelResponse(
                    "", Usage(), 0.0, None, cached=False, error=f"{type(e).__name__}: {e}"
                )

        resp = ModelResponse(text, usage, latency, cost_usd, cached=False)
        self.cache.put(key, self.cfg.id, task, sample_id, prompt, self.cache_params(params), resp)
        return resp

    @staticmethod
    def _is_rate_error(e: Exception) -> bool:
        s = str(e)
        return (
            "429" in s
            or "RESOURCE_EXHAUSTED" in s
            or "rate limit" in s.lower()
            or "overloaded" in s.lower()
        )

    async def _call_with_retry(
        self, image_jpeg: bytes, prompt: str, params: GenParams
    ) -> tuple[str, Usage, float]:
        """Outer retry loop (the SDKs already retry 429/5xx internally).

        Latency covers the final successful attempt only, not queueing/backoff.
        Rate-limit errors back off much longer — quota windows are per-minute.
        """
        last: Exception | None = None
        for attempt in range(self.run_cfg.max_retries):
            t0 = time.perf_counter()
            try:
                text, usage = await self._call(image_jpeg, prompt, params)
                self._validate_provider_result(text, usage)
                return text, usage, time.perf_counter() - t0
            except Exception as e:
                last = e
                if attempt < self.run_cfg.max_retries - 1:
                    if self._is_rate_error(e):
                        delay = 20.0 * (attempt + 1) + random.random() * 5.0
                    else:
                        delay = min(2.0**attempt + random.random(), 30.0)
                    await asyncio.sleep(delay)
        assert last is not None
        raise last

    def _validate_provider_result(self, text: str, usage: Usage) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("provider returned empty response text")
        if not isinstance(usage, Usage):
            raise TypeError("provider returned invalid usage metadata")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(usage, field_name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise TypeError(f"provider returned invalid usage.{field_name}")
        if self.cfg.pricing is not None and (
            usage.input_tokens is None or usage.output_tokens is None
        ):
            raise ValueError("billable provider response is missing usage token counts")

    def _validate_cost(self, cost_usd: float | None) -> None:
        if self.cfg.pricing is not None and cost_usd is None:
            raise ValueError("billable provider response has no computed cost")
        if cost_usd is not None and (
            isinstance(cost_usd, bool)
            or not isinstance(cost_usd, (int, float))
            or not math.isfinite(cost_usd)
            or cost_usd < 0
        ):
            raise ValueError("computed cost must be a finite non-negative number")

    @abstractmethod
    async def _call(self, image_jpeg: bytes, prompt: str, params: GenParams) -> tuple[str, Usage]:
        """One provider call. Returns (text, usage from the provider's response)."""

    # -- cost ---------------------------------------------------------------

    def compute_cost(self, usage: Usage, latency_s: float) -> float | None:
        """Token-based cost from API-returned usage. Providers may override
        (OpenAI: image/text input split; local: GPU-rental imputation)."""
        p = self.cfg.pricing
        if p is None or usage.input_tokens is None or usage.output_tokens is None:
            return None
        return (
            usage.input_tokens / 1e6 * p.input_per_mtok
            + usage.output_tokens / 1e6 * p.output_per_mtok
        )

    # -- lifecycle ------------------------------------------------------------

    def unload(self) -> None:
        """Release heavyweight resources (local models override)."""
