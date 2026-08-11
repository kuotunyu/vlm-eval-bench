"""Model providers. Imported lazily so API-only runs never import unsloth/torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from vlmeval.cache import ResponseCache
    from vlmeval.config import ModelConfig, RunConfig
    from vlmeval.models.base import BaseModel


def build_model(cfg: "ModelConfig", run_cfg: "RunConfig", cache: "ResponseCache") -> "BaseModel":
    """Instantiate the provider implementation for a model config (lazy imports)."""
    if cfg.provider == "gemini":
        from vlmeval.models.gemini import GeminiModel

        return GeminiModel(cfg, run_cfg, cache)
    if cfg.provider == "openai":
        from vlmeval.models.openai_gpt import OpenAIModel

        return OpenAIModel(cfg, run_cfg, cache)
    if cfg.provider == "anthropic":
        from vlmeval.models.anthropic_claude import AnthropicModel

        return AnthropicModel(cfg, run_cfg, cache)
    if cfg.provider == "local_unsloth":
        from vlmeval.models.local_qwen import LocalQwenModel

        return LocalQwenModel(cfg, run_cfg, cache)
    raise ValueError(f"unknown provider: {cfg.provider!r}")
