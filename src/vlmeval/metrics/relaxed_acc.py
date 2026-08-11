"""Relaxed accuracy — the official ChartQA metric.

Numeric answers count as correct within 5% relative error of the target
(exact match required when the target is 0); non-numeric answers require an
exact match after casefolding and whitespace collapsing. `%`, `$` and
thousands separators are stripped before the numeric parse.
"""

from __future__ import annotations


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _to_float(s: str) -> float | None:
    t = s.strip().rstrip("%").strip()
    t = t.replace("$", "").replace(",", "")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def relaxed_correct(pred: str, target: str, tol: float = 0.05) -> bool:
    """True when `pred` matches `target` under the relaxed-accuracy rule."""
    p, t = _to_float(pred), _to_float(target)
    if p is not None and t is not None:
        if t == 0:
            return p == 0
        return abs(p - t) <= tol * abs(t)
    return _norm(pred) == _norm(target)
