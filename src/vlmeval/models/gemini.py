"""Gemini provider (google-genai SDK, async client).

Thinking is minimized for fairness: gemini-3.x models accept
thinking_level="minimal" (cannot be fully disabled); 2.5-family models accept
thinking_budget=0. Thought tokens are billed as output tokens and included in
the reported output token count.
"""

from __future__ import annotations

import os

from vlmeval.cache import ResponseCache
from vlmeval.config import ModelConfig, RunConfig
from vlmeval.models.base import BaseModel, GenParams, Usage


class GeminiModel(BaseModel):
    def __init__(self, cfg: ModelConfig, run_cfg: RunConfig, cache: ResponseCache):
        super().__init__(cfg, run_cfg, cache)
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
        self.client = genai.Client(api_key=api_key)

    def _thinking_config(self):
        from google.genai import types

        if self.cfg.model_id and self.cfg.model_id.startswith("gemini-3"):
            return types.ThinkingConfig(thinking_level="minimal")
        return types.ThinkingConfig(thinking_budget=0)

    async def _call(self, image_jpeg: bytes, prompt: str, params: GenParams) -> tuple[str, Usage]:
        from google.genai import types

        r = await self.client.aio.models.generate_content(
            model=self.cfg.model_id,
            contents=[
                types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=params.temperature,
                max_output_tokens=params.max_tokens,
                thinking_config=self._thinking_config(),
            ),
        )
        u = r.usage_metadata
        out_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        return r.text or "", Usage(input_tokens=u.prompt_token_count, output_tokens=out_tokens)
