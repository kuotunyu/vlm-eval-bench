import gzip
import json

import pytest

from vlmeval.config import AppConfig, ModelConfig, RunConfig, TaskConfig
from vlmeval.private_rescore import RescoreError, recompute_private_run


def _config() -> AppConfig:
    return AppConfig(
        run=RunConfig(seed=3407, bootstrap_iters=50),
        tasks=(
            TaskConfig("docvqa", "fixture", "test", 1, 1, 8, "anls"),
            TaskConfig("chartqa", "fixture", "test", 1, 1, 8, "relaxed_accuracy"),
            TaskConfig("cord", "fixture", "test", 1, 1, 8, "cord_f1"),
        ),
        models=(ModelConfig("model", "fake", model_id="model"),),
    )


def _row(task: str) -> dict:
    common = {
        "sample_id": f"{task}_0",
        "model": "model",
        "task": task,
        "pred_raw": "unused",
        "pred_clean": "unused",
        "score": 1.0,
        "input_tokens": 11,
        "output_tokens": 3,
        "usage_source": "api",
        "cost_usd": 0.012,
        "latency_s": 0.5,
        "cached": False,
        "error": None,
        "aux": {},
        "meta": {},
    }
    if task == "docvqa":
        common.update(pred_raw="ab", pred_clean="ab", reference=["ac"])
    elif task == "chartqa":
        common.update(pred_raw="14%", pred_clean="14%", reference="0.14")
    else:
        reference = {
            "items": [],
            "subtotal": None,
            "discount": None,
            "service": None,
            "tax": None,
            "total": 10,
        }
        common.update(pred_raw='{"total": 10}', reference=reference)
        common["aux"] = {"parsed": {**reference}}
    return common


def _write_fixture(input_dir, *, duplicate=False):
    for task in ("docvqa", "chartqa", "cord"):
        row = _row(task)
        rows = [row]
        if duplicate and task == "docvqa":
            failed = {**row, "score": 0.0, "error": "retry"}
            rows = [failed, row]
        (input_dir / f"model__{task}.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
        )


def test_private_rescore_recomputes_metrics_and_preserves_accounting(tmp_path):
    input_dir = tmp_path / "private"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    input_dir.mkdir()
    _write_fixture(input_dir, duplicate=True)

    first = recompute_private_run(_config(), input_dir, first_output)
    second = recompute_private_run(_config(), input_dir, second_output)

    comparison = {(row["model"], row["task"]): row for row in first["comparisons"]}
    assert comparison[("model", "docvqa")] == {
        "model": "model",
        "task": "docvqa",
        "changed_rows": 1,
        "historical_score": 1.0,
        "corrected_score": 0.0,
        "delta": -1.0,
    }
    assert comparison[("model", "chartqa")]["corrected_score"] == 1.0
    assert comparison[("model", "cord")]["delta"] == 0.0
    assert first == second

    for path in sorted(first_output.glob("*.jsonl.gz")):
        other = second_output / path.name
        assert path.read_bytes() == other.read_bytes()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == 1
        assert not (
            {"question", "prompt", "reference", "prediction", "pred_raw", "pred_clean"}
            & set(rows[0])
        )
        assert rows[0]["usage"] == {
            "input_tokens": 11,
            "output_tokens": 3,
            "source": "api",
        }
        assert rows[0]["cost_usd"] == 0.012
        assert rows[0]["latency"] == {"seconds": 0.5, "cached": False}
        assert rows[0]["error"] is None


def test_private_rescore_fails_closed_on_partial_matrix(tmp_path):
    input_dir = tmp_path / "private"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_fixture(input_dir)
    (input_dir / "model__cord.jsonl").unlink()

    with pytest.raises(RescoreError, match="missing private prediction"):
        recompute_private_run(_config(), input_dir, output_dir)

    assert not output_dir.exists()


def test_private_rescore_rejects_malformed_or_mismatched_rows(tmp_path):
    input_dir = tmp_path / "private"
    input_dir.mkdir()
    _write_fixture(input_dir)
    path = input_dir / "model__chartqa.jsonl"
    row = _row("chartqa")
    row["model"] = "wrong-model"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RescoreError, match="model/task mismatch"):
        recompute_private_run(_config(), input_dir, tmp_path / "output")


def test_private_rescore_records_but_does_not_read_out_of_scope_input(tmp_path):
    input_dir = tmp_path / "private"
    input_dir.mkdir()
    _write_fixture(input_dir)
    excluded = input_dir / "disabled-model__docvqa.jsonl"
    excluded.write_text("private content that is intentionally not parsed", encoding="utf-8")

    manifest = recompute_private_run(_config(), input_dir, tmp_path / "output")

    assert manifest["scope"]["excluded_input_files"] == [excluded.name]


def test_private_rescore_rejects_non_finite_historical_score(tmp_path):
    input_dir = tmp_path / "private"
    input_dir.mkdir()
    _write_fixture(input_dir)
    path = input_dir / "model__docvqa.jsonl"
    row = _row("docvqa")
    row["score"] = float("nan")
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RescoreError, match="historical score"):
        recompute_private_run(_config(), input_dir, tmp_path / "output")


def test_private_rescore_requires_complete_archived_provenance(tmp_path):
    input_dir = tmp_path / "private"
    input_dir.mkdir()
    _write_fixture(input_dir)
    archived_manifest = tmp_path / "archived.json"
    archived_manifest.write_text('{"files": []}\n', encoding="utf-8")

    with pytest.raises(RescoreError, match="archived model/task coverage"):
        recompute_private_run(
            _config(),
            input_dir,
            tmp_path / "output",
            archived_manifest,
        )
