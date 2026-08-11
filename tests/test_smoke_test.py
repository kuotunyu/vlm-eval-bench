import asyncio
import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import vlmeval.cache as cache_module
import vlmeval.config as config_module
import vlmeval.models as models_module
import vlmeval.tasks as tasks_module

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test.py"
SPEC = importlib.util.spec_from_file_location("smoke_test_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
smoke_test = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_test)


def test_smoke_run_reuses_one_event_loop(monkeypatch):
    loops = []
    unloaded = []

    class FakeTask:
        name = "cord"

        def load_samples(self, _scale):
            return [
                SimpleNamespace(
                    sample_id=f"sample-{index}",
                    image_jpeg=b"image",
                    prompt="prompt",
                    reference="answer",
                )
                for index in range(2)
            ]

        def gen_params(self):
            return object()

        def score_one(self, _text, _reference):
            return {"score": 1.0}

    class FakeModel:
        async def generate(self, *_args, **_kwargs):
            loops.append(asyncio.get_running_loop())
            return SimpleNamespace(
                text="answer",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                latency_s=0.01,
                cost_usd=0.0,
                cached=False,
                error=None,
            )

        def unload(self):
            unloaded.append(True)

    cfg = SimpleNamespace(
        run=SimpleNamespace(cache_db=Path("unused.sqlite")),
        task=lambda _name: object(),
        model=lambda _model_id: object(),
    )
    monkeypatch.setattr(config_module, "load_config", lambda _path: cfg)
    monkeypatch.setattr(cache_module, "ResponseCache", lambda _path: object())
    monkeypatch.setattr(tasks_module, "build_task", lambda *_args: FakeTask())
    monkeypatch.setattr(models_module, "build_model", lambda *_args: FakeModel())

    args = Namespace(models="model-a", tasks="cord", n=2, config="config.yaml")
    assert asyncio.run(smoke_test._run(args)) == 0
    assert len(loops) == 2
    assert loops[0] is loops[1]
    assert unloaded == [True]
