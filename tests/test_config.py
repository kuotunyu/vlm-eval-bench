from copy import deepcopy

import pytest
import yaml

from vlmeval.config import load_config


def _valid_config():
    return {
        "run": {
            "seed": 3407,
            "cost_cap_usd": 10.0,
            "image_max_side": 1280,
            "jpeg_quality": 90,
            "max_retries": 3,
            "bootstrap_iters": 2000,
            "gpu_rent_usd_per_hour": 0.35,
        },
        "tasks": [
            {
                "name": "fixture",
                "hf_dataset": "fixture/dataset",
                "split": "test",
                "n_full": 3,
                "n_mini": 1,
                "max_output_tokens": 8,
                "metric": "fixture",
            }
        ],
        "models": [
            {
                "id": "api",
                "provider": "gemini",
                "model_id": "fixture-model",
                "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
                "rate_limit": {"concurrency": 2, "rpm": 10},
            }
        ],
    }


def _load(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["run"].update(cost_cap_usd=-1), "cost_cap_usd"),
        (lambda raw: raw["tasks"][0].update(n_mini=4), "n_mini"),
        (lambda raw: raw.update(tasks=[]), "task"),
        (lambda raw: raw["models"].append(deepcopy(raw["models"][0])), "duplicate model"),
        (
            lambda raw: raw["models"][0]["rate_limit"].update(concurrency=0),
            "concurrency",
        ),
        (
            lambda raw: raw["models"][0]["pricing"].update(input_per_mtok=-1),
            "input_per_mtok",
        ),
    ],
)
def test_invalid_configuration_is_rejected_before_execution(tmp_path, mutate, message):
    raw = _valid_config()
    mutate(raw)

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, raw)


def test_offline_config_load_does_not_read_dotenv(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")
    calls = []
    monkeypatch.setattr("vlmeval.config.load_dotenv", lambda: calls.append(True))

    load_config(path, load_environment=False)

    assert calls == []
