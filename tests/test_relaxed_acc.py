from vlmeval.metrics.relaxed_acc import relaxed_correct


def test_numeric_within_5_percent():
    assert relaxed_correct("12.5", "12")  # 4.17% off
    assert not relaxed_correct("12.7", "12")  # 5.83% off


def test_numeric_boundary_inclusive():
    assert relaxed_correct("10.5", "10")  # exactly 5%


def test_percentage_is_converted_to_a_fraction():
    assert relaxed_correct("14%", "0.14")
    assert not relaxed_correct("14%", "14")


def test_non_numeric_formatting_is_not_silently_removed():
    assert not relaxed_correct("$1,000", "1000")


def test_zero_target_requires_exact_zero():
    assert relaxed_correct("0", "0")
    assert not relaxed_correct("0.1", "0")
    assert not relaxed_correct("0.0", "0")


def test_string_casefold_exact():
    assert relaxed_correct("Yes", "yes")
    assert not relaxed_correct("increasing", "decreasing")
    assert not relaxed_correct("New  York", "new york")


def test_numeric_vs_word_falls_to_string_compare():
    assert not relaxed_correct("12", "twelve")


def test_negative_numbers():
    assert relaxed_correct("-10.2", "-10")
    assert not relaxed_correct("10", "-10")
