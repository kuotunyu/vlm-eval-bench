import json

import pytest

from vlmeval.config import RunConfig, TaskConfig
from vlmeval.tasks.base import BaseTask


class FixtureTask(BaseTask):
    def make_sample(self, row, index):  # pragma: no cover - not used by manifest tests
        raise NotImplementedError

    def score_one(self, pred_text, reference):  # pragma: no cover - not used here
        raise NotImplementedError


def _task():
    return FixtureTask(
        TaskConfig(
            name="fixture",
            hf_dataset="fixture-dataset",
            hf_config="fixture-config",
            split="test",
            n_full=3,
            n_mini=1,
            max_output_tokens=8,
            metric="fixture",
        ),
        RunConfig(seed=3407),
    )


def _write_manifest(path, **overrides):
    manifest = {
        "dataset": "fixture-dataset",
        "config": "fixture-config",
        "split": "test",
        "seed": 3407,
        "n_rows_in_split": 5,
        "indices": [0, 1, 2],
    }
    manifest.update(overrides)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_seed_must_match_run_seed(tmp_path, monkeypatch):
    monkeypatch.setattr("vlmeval.tasks.base.MANIFEST_DIR", tmp_path)
    path = tmp_path / "fixture_seed3407.json"
    _write_manifest(path, seed=999)

    with pytest.raises(RuntimeError, match="seed"):
        _task().get_or_create_indices(list(range(5)))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_rows_in_split": 6}, "row count"),
        ({"indices": [0, 0, 2]}, "duplicate"),
        ({"indices": [0, 1, 5]}, "range"),
        ({"indices": [0, 1]}, "index count"),
        ({"indices": ["0", 1, 2]}, "integers"),
    ],
)
def test_manifest_indices_are_validated(tmp_path, monkeypatch, overrides, message):
    monkeypatch.setattr("vlmeval.tasks.base.MANIFEST_DIR", tmp_path)
    path = tmp_path / "fixture_seed3407.json"
    _write_manifest(path, **overrides)

    with pytest.raises(RuntimeError, match=message):
        _task().get_or_create_indices(list(range(5)))
