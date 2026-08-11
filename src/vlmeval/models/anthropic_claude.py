"""Anthropic provider (AsyncAnthropic messages API).

Image tokens (~ width*height/750) are already included in usage.input_tokens,
so cost needs no split. The SDK auto-retries 429/5xx internally.
"""

from __future__ import annotations

import base64

from vlmeval.cache import ResponseCache
from vlmeval.config import ModelConfig, RunConfig
from vlmeval.models.base import BaseModel, GenParams, Usage


class AnthropicModel(BaseModel):
    def __init__(self, cfg: ModelConfig, run_cfg: RunConfig, cache: ResponseCache):
        super().__init__(cfg, run_cfg, cache)
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic()  # ANTHROPIC_API_KEY from env

    async def _call(self, image_jpeg: bytes, prompt: str, params: GenParams) -> tuple[str, Usage]:
        b64 = base64.b64encode(image_jpeg).decode()
        r = await self.client.messages.create(
            model=self.cfg.model_id,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, Usage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)
