import pytest

from vlmeval.cache import ResponseCache
from vlmeval.config import ModelConfig, Pricing, RunConfig
from vlmeval.models.base import BaseModel, GenParams, Usage

RUN = RunConfig(max_retries=3)
PARAMS = GenParams(max_tokens=64)


class FakeModel(BaseModel):
    """Counts provider calls; optionally fails every call."""

    def __init__(self, cfg, run_cfg, cache, fail=False, text="42", usage=None):
        super().__init__(cfg, run_cfg, cache)
        self.calls = 0
        self.fail = fail
        self.text = text
        self.usage = usage or Usage(input_tokens=1000, output_tokens=10)

    async def _call(self, image_jpeg, prompt, params):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return self.text, self.usage


def make_model(tmp_path, **kwargs):
    cfg = ModelConfig(
        id="fake",
        provider="fake",
        pricing=Pricing(input_per_mtok=1.0, output_per_mtok=5.0),
    )
    cache = ResponseCache(tmp_path / "cache.sqlite")
    return FakeModel(cfg, RUN, cache, **kwargs), cache


def test_make_key_deterministic_and_sensitive():
    k = ResponseCache.make_key("m", "t", "s", "prompt", b"jpg", {"a": 1})
    assert k == ResponseCache.make_key("m", "t", "s", "prompt", b"jpg", {"a": 1})
    assert k != ResponseCache.make_key("m", "t", "s", "other prompt", b"jpg", {"a": 1})
    assert k != ResponseCache.make_key("m2", "t", "s", "prompt", b"jpg", {"a": 1})
    assert k != ResponseCache.make_key("m", "t", "s", "prompt", b"other", {"a": 1})
    assert k != ResponseCache.make_key("m", "t", "s", "prompt", b"jpg", {"a": 2})


async def test_second_call_is_cache_hit(tmp_path):
    model, cache = make_model(tmp_path)
    r1 = await model.generate(b"jpg", "q", PARAMS, task="t", sample_id="s1")
    r2 = await model.generate(b"jpg", "q", PARAMS, task="t", sample_id="s1")
    assert model.calls == 1
    assert not r1.cached and r2.cached
    assert r2.text == "42"
    assert r2.cost_usd == pytest.approx(r1.cost_usd)
    assert cache.stats() == {("fake", "t"): 1}


async def test_changed_image_bytes_miss_cache(tmp_path):
    model, _ = make_model(tmp_path)

    first = await model.generate(b"jpg-v1", "q", PARAMS, task="t", sample_id="s1")
    second = await model.generate(b"jpg-v2", "q", PARAMS, task="t", sample_id="s1")

    assert model.calls == 2
    assert not first.cached
    assert not second.cached


async def test_no_cache_bypasses_reads(tmp_path):
    model, _ = make_model(tmp_path)
    await model.generate(b"jpg", "q", PARAMS, task="t", sample_id="s1")
    model.use_cache = False
    r2 = await model.generate(b"jpg", "q", PARAMS, task="t", sample_id="s1")
    assert model.calls == 2
    assert not r2.cached


async def test_errors_are_returned_and_never_cached(tmp_path, monkeypatch):
    import asyncio

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    model, cache = make_model(tmp_path, fail=True)
    r = await model.generate(b"jpg", "q", PARAMS, task="t", sample_id="s1")
    assert r.error is not None and "boom" in r.error
    assert model.calls == RUN.max_retries
    assert cache.stats() == {}
    key = model.cache_key("t", "s1", "q", b"jpg", PARAMS)
    assert cache.get(key) is None


async def test_empty_response_is_retried_and_never_cached(tmp_path, monkeypatch):
    import asyncio

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    model, cache = make_model(tmp_path, text="")

    response = await model.generate(b"jpg", "q", PARAMS, task="t", sample_id="s1")

    assert response.error is not None and "empty" in response.error.lower()
    assert model.calls == RUN.max_retries
    assert cache.stats() == {}


async def test_billable_response_without_usage_is_retried_and_never_cached(tmp_path, monkeypatch):
    import asyncio

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    model, cache = make_model(tmp_path, usage=Usage())

    response = await model.generate(b"jpg", "q", PARAMS, task="t", sample_id="s1")

    assert response.error is not None and "usage" in response.error.lower()
    assert model.calls == RUN.max_retries
    assert cache.stats() == {}


def test_image_policy_is_part_of_the_key(tmp_path):
    cfg = ModelConfig(id="m", provider="fake")
    cache = ResponseCache(tmp_path / "c.sqlite")
    a = FakeModel(cfg, RunConfig(image_max_side=1280), cache)
    b = FakeModel(cfg, RunConfig(image_max_side=640), cache)
    assert a.cache_key("t", "s", "q", b"jpg", PARAMS) != b.cache_key("t", "s", "q", b"jpg", PARAMS)


def test_token_cost_computation(tmp_path):
    model, _ = make_model(tmp_path)
    cost = model.compute_cost(Usage(input_tokens=1000, output_tokens=10), latency_s=1.0)
    # 1000/1e6*$1 + 10/1e6*$5
    assert cost == pytest.approx(0.00105)
