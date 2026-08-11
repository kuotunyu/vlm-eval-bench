"""Task abstraction: dataset loading, seeded sampling with committed manifests,
prompt construction, scoring, aggregation."""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlmeval.config import RunConfig, TaskConfig
from vlmeval.images import prepare_image
from vlmeval.metrics.bootstrap import bootstrap_ci
from vlmeval.models.base import GenParams

MANIFEST_DIR = Path("data/samples")


def clean_answer(text: str) -> str:
    """Shared answer post-processing, identical for every model (fairness).

    Strips markdown fences, takes the first non-empty line, drops a leading
    "Answer:" prefix, surrounding quotes, and one trailing period.
    """
    t = text.strip()
    t = re.sub(r"^```(?:\w+)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    for line in t.splitlines():
        if line.strip():
            t = line.strip()
            break
    else:
        return ""
    t = re.sub(r"^answer\s*[:：]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip().strip('"').strip("'").strip()
    if t.endswith("."):
        t = t[:-1].strip()
    return t


@dataclass
class Sample:
    sample_id: str
    image_jpeg: bytes
    prompt: str
    reference: Any
    meta: dict = field(default_factory=dict)


class BaseTask(ABC):
    def __init__(self, cfg: TaskConfig, run_cfg: RunConfig):
        self.cfg = cfg
        self.run_cfg = run_cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    def gen_params(self) -> GenParams:
        return GenParams(max_tokens=self.cfg.max_output_tokens)

    # -- dataset & sampling --------------------------------------------------

    def load_dataset(self):
        from datasets import load_dataset

        if self.cfg.hf_config:
            return load_dataset(self.cfg.hf_dataset, self.cfg.hf_config, split=self.cfg.split)
        return load_dataset(self.cfg.hf_dataset, split=self.cfg.split)

    def select_indices(self, ds) -> list[int]:
        """Full-scale index list, drawn once with the fixed seed. The mini scale
        is always a PREFIX of this list, so scaling up reuses cached responses."""
        rng = random.Random(self.run_cfg.seed)
        return rng.sample(range(len(ds)), min(self.cfg.n_full, len(ds)))

    def manifest_path(self) -> Path:
        return MANIFEST_DIR / f"{self.name}_seed{self.run_cfg.seed}.json"

    def get_or_create_indices(self, ds) -> list[int]:
        path = self.manifest_path()
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["dataset"] != self.cfg.hf_dataset or manifest["split"] != self.cfg.split:
                raise RuntimeError(
                    f"manifest {path} does not match config; delete it to regenerate"
                )
            return manifest["indices"]
        indices = self.select_indices(ds)
        manifest = {
            "dataset": self.cfg.hf_dataset,
            "config": self.cfg.hf_config,
            "split": self.cfg.split,
            "seed": self.run_cfg.seed,
            "n_rows_in_split": len(ds),
            "indices": indices,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return indices

    def load_samples(self, scale: str) -> list[Sample]:
        ds = self.load_dataset()
        indices = self.get_or_create_indices(ds)
        n = self.cfg.n_for(scale)
        samples = [self.make_sample(ds[i], i) for i in indices[:n]]
        ids = [s.sample_id for s in samples]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{self.name}: duplicate sample ids in manifest selection")
        return samples

    def _prepare(self, image) -> bytes:
        return prepare_image(image, self.run_cfg.image_max_side, self.run_cfg.jpeg_quality)

    # -- per-task hooks --------------------------------------------------------

    @abstractmethod
    def make_sample(self, row: dict, index: int) -> Sample: ...

    @abstractmethod
    def score_one(self, pred_text: str, reference: Any) -> dict:
        """Score one prediction. Must return at least {"score": float};
        extra keys (pred_clean, parsed, ...) are persisted in the JSONL row."""

    # -- aggregation -----------------------------------------------------------

    def aggregate(self, rows: list[dict]) -> dict:
        """Aggregate persisted JSONL rows for this task (one model)."""
        scores = [r["score"] for r in rows]
        mean = sum(scores) / len(scores) if scores else 0.0
        ci = bootstrap_ci(
            scores,
            lambda xs: sum(xs) / len(xs),
            n_boot=self.run_cfg.bootstrap_iters,
            seed=self.run_cfg.seed,
        )
        n_err = sum(1 for r in rows if r.get("error"))
        return {
            "metric": self.cfg.metric,
            "n": len(rows),
            "score": round(mean, 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "error_rate": round(n_err / len(rows), 4) if rows else 0.0,
        }
