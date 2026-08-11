"""Cost estimation, the pre-run cap gate, and the mid-run cost meter."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from vlmeval.config import ModelConfig


@dataclass(frozen=True)
class EstimateRow:
    model_id: str
    task: str
    n: int
    n_cached: int
    est_usd: float

    @property
    def n_new(self) -> int:
        return self.n - self.n_cached


def estimate_row(
    model_cfg: ModelConfig,
    task_name: str,
    n: int,
    n_cached: int,
    avg_prompt_chars: float,
    max_output_tokens: int,
) -> EstimateRow:
    """Conservative pre-run estimate for the cap gate (~4 chars/token for text;
    est_image_tokens per image; full max_output_tokens for output). Actual
    accounting always comes from API-returned usage, never from this."""
    p = model_cfg.pricing
    if p is None:  # local models: no API spend
        return EstimateRow(model_cfg.id, task_name, n, n_cached, 0.0)
    n_new = max(0, n - n_cached)
    image_rate = p.image_input_per_mtok if p.image_input_per_mtok is not None else p.input_per_mtok
    per_q = (
        model_cfg.est_image_tokens / 1e6 * image_rate
        + (avg_prompt_chars / 4.0) / 1e6 * p.input_per_mtok
        + max_output_tokens / 1e6 * p.output_per_mtok
    )
    return EstimateRow(model_cfg.id, task_name, n, n_cached, n_new * per_q)


def format_estimate_table(
    rows: list[EstimateRow], cap: float, title: str = "Pre-run cost estimate"
) -> str:
    lines = [title, f"{'model':<28}{'task':<10}{'n':>5}{'cached':>8}{'new':>6}{'est $':>10}"]
    for r in rows:
        lines.append(
            f"{r.model_id:<28}{r.task:<10}{r.n:>5}{r.n_cached:>8}{r.n_new:>6}{r.est_usd:>10.4f}"
        )
    total = sum(r.est_usd for r in rows)
    lines.append(f"{'TOTAL':<57}{total:>10.4f}   (cap: ${cap:.2f})")
    return "\n".join(lines)


def gate(
    rows: list[EstimateRow],
    cap: float,
    yes: bool,
    confirm_fn: Callable[[str], str] = input,
) -> bool:
    """True = proceed. When the estimate exceeds the cap, proceed only with
    --yes AND an interactive 'yes' confirmation."""
    total = sum(r.est_usd for r in rows)
    if total <= cap:
        return True
    if not yes:
        return False
    answer = confirm_fn(
        f"Estimated ${total:.4f} exceeds the ${cap:.2f} cap. Type 'yes' to proceed: "
    )
    return answer.strip().lower() == "yes"


class CostMeter:
    """Running total of actual spend for one run; thread-safe."""

    def __init__(self, cap_usd: float):
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.cached_usd = 0.0  # original cost of cache-hit responses (informational)
        self._lock = threading.Lock()

    def add(self, cost_usd: float | None, cached: bool = False) -> bool:
        """Record one response's cost. Returns False once the cap is exceeded."""
        with self._lock:
            if cost_usd:
                if cached:
                    self.cached_usd += cost_usd
                else:
                    self.spent_usd += cost_usd
            return self.spent_usd <= self.cap_usd

    @property
    def exceeded(self) -> bool:
        return self.spent_usd > self.cap_usd
