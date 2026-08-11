from vlmeval.config import AppConfig, ModelConfig, RunConfig, TaskConfig
import json

from vlmeval.reporting import (
    _analysis_lines,
    _complete_model_ids,
    _dedup_rows,
    _make_charts,
    generate_report,
)


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


def test_report_input_order_is_canonical_after_latest_row_deduplication(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [
        {"sample_id": "b", "score": 0.5},
        {"sample_id": "a", "score": 1.0},
    ]
    first.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    second.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )

    assert _dedup_rows(first) == _dedup_rows(second)
    assert [row["sample_id"] for row in _dedup_rows(first)] == ["a", "b"]


def test_public_report_never_renders_dataset_or_prediction_text(tmp_path, monkeypatch):
    class FakeTask:
        name = "docvqa"

        @staticmethod
        def aggregate(rows):
            return {"score": 0.0, "ci95": [0.0, 0.0], "n": len(rows)}

    cfg = AppConfig(
        run=RunConfig(),
        tasks=(
            TaskConfig(
                name="docvqa",
                hf_dataset="fixture",
                split="test",
                n_full=1,
                n_mini=1,
                max_output_tokens=8,
                metric="fixture",
            ),
        ),
        models=(ModelConfig(id="api", provider="fake"),),
    )
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    row = {
        "sample_id": "secret-id",
        "score": 0.0,
        "prompt": "PRIVATE-QUESTION",
        "reference": "PRIVATE-REFERENCE",
        "pred_clean": "PRIVATE-PREDICTION",
        "cost_usd": 0.001,
        "input_tokens": 10,
        "output_tokens": 2,
        "latency_s": 0.5,
        "cached": False,
        "error": None,
    }
    (pred_dir / "api__docvqa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr("vlmeval.reporting.build_task", lambda *_: FakeTask())
    monkeypatch.setattr("vlmeval.reporting._make_charts", lambda *_: [])

    generate_report(cfg, tmp_path)

    report = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "PRIVATE-QUESTION" not in report
    assert "PRIVATE-REFERENCE" not in report
    assert "PRIVATE-PREDICTION" not in report
    assert "Representative error cases" not in report


def test_charts_do_not_compare_cross_task_average_with_cost(tmp_path):
    class FakeTask:
        name = "docvqa"

    rows_by = {
        ("api", "docvqa"): [{"cost_usd": 0.01, "latency_s": 0.5, "cached": False, "error": None}]
    }
    charts = _make_charts(
        _cfg(),
        [FakeTask()],
        ["api"],
        set(),
        {("api", "docvqa"): {"score": 0.5, "ci95": [0.4, 0.6]}},
        rows_by,
        tmp_path,
    )

    assert "cost_vs_score.png" not in {chart.name for chart in charts}
    assert not (tmp_path / "cost_vs_score.png").exists()
