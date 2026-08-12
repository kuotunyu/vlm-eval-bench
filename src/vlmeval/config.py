"""Configuration loading: config.yaml -> typed dataclasses; .env -> environment."""

from __future__ import annotations

import math
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


def _require_finite_non_negative(value: float, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")


def validate_config(cfg: AppConfig) -> None:
    """Reject unsafe or ambiguous configuration before loading data or providers."""
    run = cfg.run
    _require_finite_non_negative(run.cost_cap_usd, "run.cost_cap_usd")
    _require_finite_non_negative(run.gpu_rent_usd_per_hour, "run.gpu_rent_usd_per_hour")
    if not isinstance(run.seed, int) or isinstance(run.seed, bool):
        raise ValueError("run.seed must be an integer")
    if run.image_max_side < 1:
        raise ValueError("run.image_max_side must be positive")
    if not 1 <= run.jpeg_quality <= 100:
        raise ValueError("run.jpeg_quality must be between 1 and 100")
    if run.max_retries < 1:
        raise ValueError("run.max_retries must be at least 1")
    if run.bootstrap_iters < 1:
        raise ValueError("run.bootstrap_iters must be at least 1")

    if not cfg.tasks:
        raise ValueError("at least one task is required")
    task_names = [task.name for task in cfg.tasks]
    if len(task_names) != len(set(task_names)):
        raise ValueError("duplicate task name")
    for task in cfg.tasks:
        if not task.name or not task.hf_dataset or not task.split:
            raise ValueError("task name, hf_dataset, and split must be non-empty")
        if task.n_full < 1:
            raise ValueError(f"task {task.name} n_full must be positive")
        if not 1 <= task.n_mini <= task.n_full:
            raise ValueError(f"task {task.name} n_mini must be between 1 and n_full")
        if task.max_output_tokens < 1:
            raise ValueError(f"task {task.name} max_output_tokens must be positive")

    if not cfg.models:
        raise ValueError("at least one model is required")
    model_ids = [model.id for model in cfg.models]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("duplicate model id")
    for model in cfg.models:
        if not model.id or not model.provider:
            raise ValueError("model id and provider must be non-empty")
        if model.is_local and not model.model_path:
            raise ValueError(f"local model {model.id} requires model_path")
        if not model.is_local and not model.model_id:
            raise ValueError(f"API model {model.id} requires model_id")
        if model.est_image_tokens < 0:
            raise ValueError(f"model {model.id} est_image_tokens must be non-negative")
        if model.rate_limit.concurrency < 1:
            raise ValueError(f"model {model.id} concurrency must be positive")
        if model.rate_limit.rpm is not None and model.rate_limit.rpm < 1:
            raise ValueError(f"model {model.id} rpm must be positive when set")
        if model.pricing is not None:
            _require_finite_non_negative(
                model.pricing.input_per_mtok, f"model {model.id} input_per_mtok"
            )
            _require_finite_non_negative(
                model.pricing.output_per_mtok, f"model {model.id} output_per_mtok"
            )
            if model.pricing.image_input_per_mtok is not None:
                _require_finite_non_negative(
                    model.pricing.image_input_per_mtok,
                    f"model {model.id} image_input_per_mtok",
                )


def load_config(path: str | Path = "config.yaml", *, load_environment: bool = True) -> AppConfig:
    """Parse config.yaml; optionally load the local dotenv for inference commands."""
    if load_environment:
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

    config = AppConfig(run=run, tasks=tasks, models=tuple(models))
    validate_config(config)
    return config
