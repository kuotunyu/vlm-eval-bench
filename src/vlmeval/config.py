"""Configuration loading: config.yaml -> typed dataclasses; .env -> environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class RunConfig:
    seed: int = 3407
    cost_cap_usd: float = 10.0
    output_dir: Path = Path("results")
    cache_db: Path = Path("results/cache.sqlite")
    image_max_side: int = 1280
    jpeg_quality: int = 90
    max_retries: int = 3
    bootstrap_iters: int = 2000
    gpu_rent_usd_per_hour: float = 0.35


@dataclass(frozen=True)
class TaskConfig:
    name: str
    hf_dataset: str
    split: str
    n_full: int
    n_mini: int
    max_output_tokens: int
    metric: str
    hf_config: str | None = None
    stratify_field: str | None = None

    def n_for(self, scale: str) -> int:
        return self.n_mini if scale == "mini" else self.n_full


@dataclass(frozen=True)
class RateLimitConfig:
    concurrency: int = 1
    rpm: int | None = None


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float
    image_input_per_mtok: float | None = None


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    enabled: bool = True
    model_id: str | None = None
    model_path: str | None = None
    pricing: Pricing | None = None
    est_image_tokens: int = 1000
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    @property
    def is_local(self) -> bool:
        return self.provider.startswith("local")


@dataclass(frozen=True)
class AppConfig:
    run: RunConfig
    tasks: tuple[TaskConfig, ...]
    models: tuple[ModelConfig, ...]

    def task(self, name: str) -> TaskConfig:
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(f"unknown task: {name!r}")

    def model(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model: {model_id!r}")

    def enabled_models(self) -> tuple[ModelConfig, ...]:
        return tuple(m for m in self.models if m.enabled)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Parse config.yaml into an AppConfig; also loads .env into the environment."""
    load_dotenv()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    run_raw = dict(raw.get("run", {}))
    for key in ("output_dir", "cache_db"):
        if key in run_raw:
            run_raw[key] = Path(run_raw[key])
    run = RunConfig(**run_raw)

    tasks = tuple(TaskConfig(**t) for t in raw.get("tasks", []))

    models = []
    for m in raw.get("models", []):
        m = dict(m)
        if m.get("pricing") is not None:
            m["pricing"] = Pricing(**m["pricing"])
        if m.get("rate_limit") is not None:
            m["rate_limit"] = RateLimitConfig(**m["rate_limit"])
        else:
            m.pop("rate_limit", None)
        models.append(ModelConfig(**m))

    return AppConfig(run=run, tasks=tasks, models=tuple(models))
