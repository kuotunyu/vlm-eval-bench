from vlmeval.config import AppConfig, ModelConfig, RunConfig, TaskConfig
from vlmeval.reporting import _analysis_lines, _complete_model_ids


def _cfg():
    tasks = tuple(
        TaskConfig(
            name=name,
            hf_dataset="fixture",
            split="test",
            n_full=n,
            n_mini=1,
            max_output_tokens=8,
            metric="fixture",
        )
        for name, n in (("docvqa", 2), ("chartqa", 2), ("cord", 1))
    )
    models = (
        ModelConfig(id="qwen3vl-8b-base", provider="local_fake"),
        ModelConfig(id="qwen3vl-8b-receipt-qlora", provider="local_fake"),
        ModelConfig(id="api", provider="fake"),
        ModelConfig(id="partial", provider="fake"),
    )
    return AppConfig(run=RunConfig(), tasks=tasks, models=models)


def _rows(n, *, cost=0.001, error=None):
    return [{"sample_id": str(i), "cost_usd": cost, "error": error} for i in range(n)]


def test_interrupted_model_is_excluded_from_leaderboard():
    cfg = _cfg()
    rows_by = {}
    for model in cfg.models[:3]:
        rows_by[(model.id, "docvqa")] = _rows(2)
        rows_by[(model.id, "chartqa")] = _rows(2)
        rows_by[(model.id, "cord")] = _rows(1)
    rows_by[("partial", "docvqa")] = _rows(2)
    rows_by[("partial", "chartqa")] = _rows(1)

    complete, incomplete = _complete_model_ids(cfg, rows_by)

    assert complete == ["qwen3vl-8b-base", "qwen3vl-8b-receipt-qlora", "api"]
    assert incomplete["partial"] == {
        "docvqa": (2, 2),
        "chartqa": (1, 2),
        "cord": (0, 1),
    }


def test_analysis_is_data_driven_and_has_no_placeholder():
    cfg = _cfg()
    model_ids = [model.id for model in cfg.models[:3]]
    task_names = [task.name for task in cfg.tasks]
    rows_by = {
        (mid, task): _rows(2 if task != "cord" else 1) for mid in model_ids for task in task_names
    }
    scores = {
        "qwen3vl-8b-base": (0.90, 0.80, 0.70),
        "qwen3vl-8b-receipt-qlora": (0.89, 0.81, 0.92),
        "api": (0.80, 0.70, 0.75),
    }
    aggs = {
        (mid, task): {"score": scores[mid][i], "ci95": [0.0, 1.0]}
        for mid in model_ids
        for i, task in enumerate(task_names)
    }

    text = "\n".join(
        _analysis_lines(
            task_names, model_ids, {"qwen3vl-8b-base", "qwen3vl-8b-receipt-qlora"}, aggs, rows_by
        )
    )

    assert "qwen3vl-8b-receipt-qlora" in text
    assert "CORD +0.220" in text
    assert "to be written" not in text
