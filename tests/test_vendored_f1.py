from vlmeval.vendored.metrics import compute_metrics


def gt(items=(), subtotal=None, discount=None, service=None, tax=None, total=None):
    return {
        "items": list(items),
        "subtotal": subtotal,
        "discount": discount,
        "service": service,
        "tax": tax,
        "total": total,
    }


def item(name, count=None, unit_price=None, price=None):
    return {"name": name, "count": count, "unit_price": unit_price, "price": price}


def test_identical_prediction_scores_one():
    g = gt(items=[item("latte", count=2, price=100)], subtotal=50, total=100)
    m = compute_metrics([{"gt": g, "parsed": g}])
    assert m["overall"]["f1"] == 1.0
    assert m["valid_json_rate"] == 1.0
    assert m["total_exact_match"] == 1.0


def test_one_wrong_scalar_is_fp_plus_fn():
    g = gt(subtotal=50, total=100)
    pred = gt(subtotal=60, total=100)  # subtotal wrong -> FP+FN; total TP
    m = compute_metrics([{"gt": g, "parsed": pred}])
    assert m["overall"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}


def test_unparseable_output_is_all_fn():
    g = gt(items=[item("latte", price=100)], total=100)
    m = compute_metrics([{"gt": g, "parsed": None}])
    assert m["overall"]["recall"] == 0.0
    assert m["overall"]["f1"] == 0.0
    assert m["valid_json_rate"] == 0.0


def test_item_alignment_survives_reordering():
    g = gt(items=[item("latte", price=100), item("mocha", price=200)], total=300)
    pred = gt(items=[item("mocha", price=200), item("latte", price=100)], total=300)
    m = compute_metrics([{"gt": g, "parsed": pred}])
    assert m["overall"]["f1"] == 1.0


def test_spurious_item_counts_fp():
    g = gt(items=[item("latte", price=100)], total=100)
    pred = gt(items=[item("latte", price=100), item("zzzz", price=5)], total=100)
    m = compute_metrics([{"gt": g, "parsed": pred}])
    # 3 TP (latte name+price, total), 2 FP (spurious name+price)
    assert m["overall"]["recall"] == 1.0
    assert m["overall"]["precision"] == 0.6


def test_numeric_string_prediction_coerced():
    g = gt(total=25000)
    pred = {"total": "25.000"}  # Indonesian thousands format
    m = compute_metrics([{"gt": g, "parsed": pred}])
    assert m["overall"]["f1"] == 1.0
