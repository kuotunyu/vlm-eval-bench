"""Local Qwen3-VL provider via Unsloth — the same load path the sibling
vlm-receipt-extractor project used for its original evaluation.

- `import unsloth` must precede any transformers import (it patches a
  transformers/bitsandbytes bug in the Qwen3-VL vision tower). This module is
  only imported for local_unsloth model configs, so API-only runs never pay it.
- Weights load lazily on first call, inside a single-worker thread executor;
  the GPU therefore serializes naturally while satisfying the async interface.
- Cost is imputed from the GPU rental rate x wall time and flagged as such;
  token counts come from the tokenizer (usage_source="tokenizer").
"""

from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor

from vlmeval.cache import ResponseCache
from vlmeval.config import ModelConfig, RunConfig
from vlmeval.models.base import BaseModel, GenParams, Usage


class LocalQwenModel(BaseModel):
    def __init__(self, cfg: ModelConfig, run_cfg: RunConfig, cache: ResponseCache):
        super().__init__(cfg, run_cfg, cache)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import unsloth  # noqa: F401  (must precede transformers — vision-tower patch)
        import torch
        from unsloth import FastVisionModel

        torch.manual_seed(self.run_cfg.seed)
        print(f"[local] loading {self.cfg.model_path} (4-bit) ...", flush=True)
        t0 = time.perf_counter()
        model, tokenizer = FastVisionModel.from_pretrained(self.cfg.model_path, load_in_4bit=True)
        FastVisionModel.for_inference(model)
        self._model, self._tokenizer = model, tokenizer
        print(f"[local] loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    async def _call(self, image_jpeg: bytes, prompt: str, params: GenParams) -> tuple[str, Usage]:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._infer, image_jpeg, prompt, params
        )

    def _infer(self, image_jpeg: bytes, prompt: str, params: GenParams) -> tuple[str, Usage]:
        self._ensure_loaded()
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_jpeg)).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        input_text = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._tokenizer(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=params.max_tokens,
                do_sample=False,
                use_cache=True,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.batch_decode(gen.unsqueeze(0), skip_special_tokens=True)[0]
        return text, Usage(
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(gen.shape[0]),
            source="tokenizer",
        )

    def compute_cost(self, usage: Usage, latency_s: float) -> float | None:
        """Imputed cost: GPU rental rate x inference wall time (flagged in report)."""
        return latency_s / 3600.0 * self.run_cfg.gpu_rent_usd_per_hour

    def unload(self) -> None:
        if self._model is None:
            return
        import gc

        import torch

        self._model = None
        self._tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[local] model unloaded, VRAM released", flush=True)
