"""OpenAI provider (Responses API, AsyncOpenAI).

Reasoning effort is set to "none" for fairness. OpenAI usage does not split
image vs text input tokens, but images bill at a different rate, so the image
share is computed from the documented patch formula (ceil(w/32)*ceil(h/32),
capped at 1536 patches, x1.62 multiplier for the mini family) and billed at the
image rate; the remainder bills at the text rate. Disclosed in the README.
"""

from __future__ import annotations

import base64
import math

from vlmeval.cache import ResponseCache
from vlmeval.config import ModelConfig, RunConfig
from vlmeval.images import image_dims
from vlmeval.models.base import BaseModel, GenParams, Usage

_PATCH = 32
_MAX_PATCHES = 1536
_MINI_MULTIPLIER = 1.62


def image_tokens(width: int, height: int) -> int:
    patches = math.ceil(width / _PATCH) * math.ceil(height / _PATCH)
    return int(min(patches, _MAX_PATCHES) * _MINI_MULTIPLIER)


class OpenAIModel(BaseModel):
    def __init__(self, cfg: ModelConfig, run_cfg: RunConfig, cache: ResponseCache):
        super().__init__(cfg, run_cfg, cache)
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()  # OPENAI_API_KEY from env
        self._last_image_tokens: int | None = None

    async def _call(self, image_jpeg: bytes, prompt: str, params: GenParams) -> tuple[str, Usage]:
        b64 = base64.b64encode(image_jpeg).decode()
        r = await self.client.responses.create(
            model=self.cfg.model_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            max_output_tokens=params.max_tokens,
            reasoning={
                "effort": "none"
            },  # gpt-5.4-mini: none|low|medium|high|xhigh; no reasoning = fairest
            temperature=params.temperature,
        )
        w, h = image_dims(image_jpeg)
        self._last_image_tokens = image_tokens(w, h)
        return r.output_text or "", Usage(
            input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens
        )

    def compute_cost(self, usage: Usage, latency_s: float) -> float | None:
        p = self.cfg.pricing
        if p is None or usage.input_tokens is None or usage.output_tokens is None:
            return None
        img_rate = (
            p.image_input_per_mtok if p.image_input_per_mtok is not None else p.input_per_mtok
        )
        img_tok = min(self._last_image_tokens or 0, usage.input_tokens)
        text_tok = usage.input_tokens - img_tok
        return (
            img_tok / 1e6 * img_rate
            + text_tok / 1e6 * p.input_per_mtok
            + usage.output_tokens / 1e6 * p.output_per_mtok
        )
