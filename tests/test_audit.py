import json
from pathlib import Path

import pytest

from vlmeval.audit import (
    AuditError,
    read_audit_rows,
    read_latest_rows,
    serialize_audit_rows,
    verify_audit_pack,
)


def _row(sample_id, *, score=1.0, error=None):
    return {
        "sample_id": sample_id,
        "model": "model",
        "task": "task",
        "score": score,
        "pred_clean": f"answer-{sample_id}-{score}",
        "input_tokens": 10,
        "output_tokens": 2,
        "usage_source": "fixture",
        "cost_usd": 0.001,
        "latency_s": 0.5,
        "cached": False,
        "error": error,
    }


def test_latest_row_wins_without_changing_first_seen_order(tmp_path):
    source = tmp_path / "predictions.jsonl"
    rows = [_row("a", score=0.0, error="retry"), _row("b"), _row("a")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    deduplicated = read_latest_rows(source)

    assert [row["sample_id"] for row in deduplicated] == ["a", "b"]
    assert deduplicated[0]["score"] == 1.0
    assert deduplicated[0]["error"] is None


def test_audit_gzip_is_deterministic_and_excludes_dataset_fields(tmp_path):
    rows = [_row("a"), _row("b", score=0.5)]

    first = serialize_audit_rows(rows)
    second = serialize_audit_rows(rows)

    assert first == second
    path = tmp_path / "audit.jsonl.gz"
    path.write_bytes(first)
    published = read_audit_rows(path)
    assert len(published) == 2
    assert not ({"prompt", "reference", "image", "meta", "aux", "prediction"} & published[0].keys())


def test_unexpected_audit_field_is_rejected(tmp_path):
    row = _row("a")
    payload = serialize_audit_rows([row])
    path = tmp_path / "audit.jsonl.gz"
    path.write_bytes(payload)
    assert read_audit_rows(path)[0]["sample_id"] == "a"

    # The committed reader is strict; adding dataset material requires a schema review.
    import gzip

    published = read_audit_rows(path)[0]
    published["reference"] = "must not be published"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(published) + "\n")
    with pytest.raises(AuditError, match="unexpected keys"):
        read_audit_rows(path)

    published.pop("reference")
    published["usage"]["reference"] = "must not be nested"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(published) + "\n")
    with pytest.raises(AuditError, match="unexpected usage schema"):
        read_audit_rows(path)


def test_non_finite_audit_score_is_rejected(tmp_path):
    import gzip

    path = tmp_path / "audit.jsonl.gz"
    path.write_bytes(serialize_audit_rows([_row("a")]))
    published = read_audit_rows(path)[0]
    published["score"] = float("nan")
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(published) + "\n")

    with pytest.raises(AuditError, match="score"):
        read_audit_rows(path)


def test_committed_pack_has_expected_coverage_and_matches_leaderboard():
    summary = verify_audit_pack(
        audit_dir=Path("results/audit"),
        leaderboard_path=Path("results/archived_leaderboard.md"),
    )

    assert summary == {
        "models": 4,
        "tasks": 3,
        "files": 12,
        "rows": 2000,
        "leaderboard_checked": True,
        "readme_checked": False,
    }


def test_leaderboard_confidence_interval_tampering_is_rejected(tmp_path):
    leaderboard = Path("results/archived_leaderboard.md").read_text(encoding="utf-8")
    tampered = leaderboard.replace("[0.905, 0.964]", "[0.000, 0.000]", 1)
    path = tmp_path / "leaderboard.md"
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(AuditError, match="confidence interval"):
        verify_audit_pack(Path("results/audit"), path)
