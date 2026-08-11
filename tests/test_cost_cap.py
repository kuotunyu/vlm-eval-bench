import pytest

from vlmeval.config import ModelConfig, Pricing
from vlmeval.cost import CostMeter, EstimateRow, estimate_row, format_estimate_table, gate

API_MODEL = ModelConfig(
    id="api-model",
    provider="fake",
    pricing=Pricing(input_per_mtok=1.0, output_per_mtok=5.0, image_input_per_mtok=0.8),
    est_image_tokens=1000,
)
LOCAL_MODEL = ModelConfig(id="local-model", provider="local_unsloth")


def test_estimate_row_math():
    r = estimate_row(
        API_MODEL, "docvqa", n=10, n_cached=4, avg_prompt_chars=400, max_output_tokens=100
    )
    # per question: image 1000tok@$0.8 + text 100tok@$1 + out 100tok@$5 = 0.0008+0.0001+0.0005
    assert r.n_new == 6
    assert r.est_usd == pytest.approx(6 * 0.0014)


def test_local_model_estimates_zero():
    r = estimate_row(
        LOCAL_MODEL, "cord", n=100, n_cached=0, avg_prompt_chars=500, max_output_tokens=768
    )
    assert r.est_usd == 0.0


def test_gate_under_cap_passes():
    rows = [EstimateRow("m", "t", 10, 0, 1.0)]
    assert gate(rows, cap=10.0, yes=False)


def test_gate_over_cap_blocks_without_yes():
    rows = [EstimateRow("m", "t", 10, 0, 20.0)]
    assert not gate(rows, cap=10.0, yes=False)


def test_gate_over_cap_requires_typed_confirmation():
    rows = [EstimateRow("m", "t", 10, 0, 20.0)]
    assert gate(rows, cap=10.0, yes=True, confirm_fn=lambda _: "yes")
    assert not gate(rows, cap=10.0, yes=True, confirm_fn=lambda _: "no")


def test_cost_meter_trips_at_cap():
    meter = CostMeter(cap_usd=2.5)
    assert meter.add(1.0)
    assert meter.add(1.0)
    assert not meter.add(1.0)  # 3.0 > 2.5
    assert meter.exceeded
    assert meter.spent_usd == pytest.approx(3.0)


def test_cost_meter_cached_spend_tracked_separately():
    meter = CostMeter(cap_usd=1.0)
    assert meter.add(0.9, cached=True)
    assert meter.spent_usd == 0.0
    assert meter.cached_usd == pytest.approx(0.9)
    assert not meter.exceeded


def test_format_estimate_table_smoke():
    rows = [EstimateRow("api-model", "docvqa", 20, 5, 0.021)]
    out = format_estimate_table(rows, cap=10.0)
    assert "api-model" in out and "TOTAL" in out and "10.00" in out
