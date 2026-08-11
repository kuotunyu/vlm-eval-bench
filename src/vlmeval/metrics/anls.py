"""ANLS (Average Normalized Levenshtein Similarity) — the official DocVQA metric.

Per-question score: max over reference answers of 1 - NL(pred, ans), where NL is
the Levenshtein distance normalized by the longer string; scores below the
threshold (0.5) are truncated to 0. Dataset ANLS is the mean over questions.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def anls_score(pred: str, answers: list[str], threshold: float = 0.5) -> float:
    """Score one prediction against a list of acceptable answers."""
    if not answers:
        return 0.0
    p = _norm(pred)
    best = 0.0
    for a in answers:
        a_n = _norm(a)
        if not p and not a_n:
            sim = 1.0
        else:
            sim = 1.0 - Levenshtein.normalized_distance(p, a_n)
        best = max(best, sim)
    return best if best >= threshold else 0.0
