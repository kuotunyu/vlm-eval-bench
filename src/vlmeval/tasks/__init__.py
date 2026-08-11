"""Evaluation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from vlmeval.config import RunConfig, TaskConfig
    from vlmeval.tasks.base import BaseTask


def build_task(cfg: "TaskConfig", run_cfg: "RunConfig") -> "BaseTask":
    """Instantiate the task implementation for a task config (lazy imports)."""
    if cfg.name == "docvqa":
        from vlmeval.tasks.docvqa import DocVQATask

        return DocVQATask(cfg, run_cfg)
    if cfg.name == "chartqa":
        from vlmeval.tasks.chartqa import ChartQATask

        return ChartQATask(cfg, run_cfg)
    if cfg.name == "cord":
        from vlmeval.tasks.cord import CordTask

        return CordTask(cfg, run_cfg)
    raise ValueError(f"unknown task: {cfg.name!r}")
