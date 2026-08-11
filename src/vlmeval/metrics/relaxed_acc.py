"""Relaxed accuracy — the ChartQA/Pix2Struct reference behavior.

Numeric answers count as correct within 5% relative error of a non-zero
target. Percentages are converted to fractions. All other cases use
case-insensitive exact string matching.
"""

from __future__ import annotations


def _norm(s: str) -> str:
    return s.lower()


def _to_float(s: str) -> float | None:
    try:
        if s.endswith("%"):
            return float(s.rstrip("%")) / 100.0
        return float(s)
    except ValueError:
        return None


def relaxed_correct(pred: str, target: str, tol: float = 0.05) -> bool:
    """True when `pred` matches `target` under the relaxed-accuracy rule."""
    p, t = _to_float(pred), _to_float(target)
    if p is not None and t:
        return abs(p - t) <= tol * abs(t)
    return _norm(pred) == _norm(target)
