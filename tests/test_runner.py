from types import SimpleNamespace

import pytest

from vlmeval.config import (
    AppConfig,
    ModelConfig,
    Pricing,
    RateLimitConfig,
    RunConfig,
    TaskConfig,
)
from vlmeval.cost import CostMeter
from vlmeval.cost import estimate_row as real_estimate_row
from vlmeval.models.base import GenParams
from vlmeval.runner import CostCapExceeded, EXIT_OK, _run_model_task, run


def test_no_cache_dry_run_does_not_credit_cache(tmp_path, monkeypatch):
    task_cfg = TaskConfig(
        name="fixture",
        hf_dataset="fixture",
        split="test",
        n_full=1,
        n_mini=1,
        max_output_tokens=8,
        metric="fixture",
    )
    model_cfg = ModelConfig(
        id="api",
        provider="fixture",
        pricing=Pricing(input_per_mtok=1.0, output_per_mtok=1.0),
    )
    cfg = AppConfig(
        run=RunConfig(cache_db=tmp_path / "cache.sqlite"),
        tasks=(task_cfg,),
        models=(model_cfg,),
    )

    class FakeCache:
        def __init__(self, _path):
            pass

        @staticmethod
        def make_key(*_args):
            return "cache-key"

        def has(self, _key):
            return True

    class FakeTask:
        name = "fixture"
        cfg = task_cfg

        def load_samples(self, _scale):
            return [
                SimpleNamespace(
                    sample_id="sample-1",
                    prompt="question",
                    image_jpeg=b"image",
                )
            ]

        def gen_params(self):
            return GenParams(max_tokens=8)

    observed_cached_counts = []

    def record_estimate(model, task_name, n, n_cached, avg_prompt_chars, max_output_tokens):
        observed_cached_counts.append(n_cached)
        return real_estimate_row(
            model,
            task_name,
            n,
            n_cached,
            avg_prompt_chars,
            max_output_tokens,
        )

    monkeypatch.setattr("vlmeval.runner.ResponseCache", FakeCache)
    monkeypatch.setattr("vlmeval.runner.build_task", lambda *_args: FakeTask())
    monkeypatch.setattr("vlmeval.runner.estimate_row", record_estimate)

    result = run(
        cfg,
        scale="mini",
        task_names=["fixture"],
        model_ids=["api"],
        dry_run=True,
        no_cache=True,
    )

    assert result == EXIT_OK
    assert observed_cached_counts == [0]


async def test_mid_run_cap_bounds_new_calls_to_provider_concurrency(tmp_path):
    started = 0

    class FakeModel:
        cfg = ModelConfig(
            id="api",
            provider="fixture",
            rate_limit=RateLimitConfig(concurrency=2),
        )

        async def generate(self, *_args, **_kwargs):
            nonlocal started
            started += 1
            import asyncio

            await asyncio.sleep(0)
            return SimpleNamespace(
                text="answer",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, source="fixture"),
                latency_s=0.01,
                cost_usd=0.10,
                cached=False,
                error=None,
            )

    class FakeTask:
        name = "fixture"

        def gen_params(self):
            return GenParams(max_tokens=8)

        def score_one(self, _text, _reference):
            return {"score": 1.0, "pred_clean": "answer"}

    samples = [
        SimpleNamespace(
            sample_id=f"sample-{index}",
            image_jpeg=b"image",
            prompt="question",
            reference="answer",
            meta={},
        )
        for index in range(10)
    ]

    with pytest.raises(CostCapExceeded):
        await _run_model_task(
            FakeModel(),
            FakeTask(),
            samples,
            tmp_path / "predictions.jsonl",
            CostMeter(cap_usd=0.05),
            "full",
        )

    assert started <= 2


async def test_local_imputed_cost_does_not_trip_paid_api_cap(tmp_path):
    started = 0

    class FakeLocalModel:
        cfg = ModelConfig(
            id="local",
            provider="local_fixture",
            rate_limit=RateLimitConfig(concurrency=1),
        )

        async def generate(self, *_args, **_kwargs):
            nonlocal started
            started += 1
            return SimpleNamespace(
                text="answer",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, source="tokenizer"),
                latency_s=0.01,
                cost_usd=0.10,
                cached=False,
                error=None,
            )

    class FakeTask:
        name = "fixture"

        def gen_params(self):
            return GenParams(max_tokens=8)

        def score_one(self, _text, _reference):
            return {"score": 1.0, "pred_clean": "answer"}

    samples = [
        SimpleNamespace(
            sample_id=f"sample-{index}",
            image_jpeg=b"image",
            prompt="question",
            reference="answer",
            meta={},
        )
        for index in range(2)
    ]
    meter = CostMeter(cap_usd=0.05)

    await _run_model_task(
        FakeLocalModel(),
        FakeTask(),
        samples,
        tmp_path / "predictions.jsonl",
        meter,
        "full",
    )

    assert started == 2
    assert meter.spent_usd == 0.0
