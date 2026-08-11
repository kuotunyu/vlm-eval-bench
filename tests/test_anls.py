import pytest

from vlmeval.metrics.anls import anls_score


def test_exact_match():
    assert anls_score("Paris", ["Paris"]) == 1.0


def test_case_and_whitespace_insensitive():
    assert anls_score("  paris ", ["Paris"]) == 1.0
    assert anls_score("new  york", ["New York"]) == 1.0


def test_near_match_above_threshold():
    # 1 edit over max-length 7 -> similarity 6/7 ~ 0.857, returned as-is
    assert anls_score("answer", ["answers"]) == pytest.approx(6 / 7)


def test_below_threshold_truncates_to_zero():
    # "abcdef" vs "abzzzz": 4 edits over 6 -> similarity 1/3 < 0.5 -> 0.0
    assert anls_score("abcdef", ["abzzzz"]) == 0.0
    assert anls_score("cat", ["dog"]) == 0.0


def test_multi_reference_takes_max():
    assert anls_score("paris", ["London", "Paris", "Rome"]) == 1.0


def test_empty_prediction():
    assert anls_score("", ["Paris"]) == 0.0


def test_no_references():
    assert anls_score("anything", []) == 0.0
