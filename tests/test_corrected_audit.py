import json
import shutil
from pathlib import Path

import pytest

from vlmeval.private_rescore import RescoreError, verify_corrected_pack


def test_committed_corrected_pack_matches_archived_evidence_and_claims():
    summary = verify_corrected_pack(
        corrected_dir=Path("results/corrected"),
        archived_audit_dir=Path("results/audit"),
        leaderboard_path=Path("results/leaderboard.md"),
        readme_path=Path("README.md"),
    )

    assert summary == {
        "models": 4,
        "tasks": 3,
        "files": 12,
        "rows": 2000,
        "leaderboard_checked": True,
        "readme_checked": True,
    }


def test_corrected_verifier_rejects_delta_tampering(tmp_path):
    source = Path("results/corrected/rescore_manifest.json")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["comparisons"][0]["delta"] = 0.999
    path = tmp_path / "rescore_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RescoreError, match="comparison mismatch"):
        verify_corrected_pack(
            corrected_dir=Path("results/corrected"),
            archived_audit_dir=Path("results/audit"),
            manifest_path=path,
        )


def test_corrected_verifier_rejects_private_source_paths(tmp_path):
    source = Path("results/corrected/rescore_manifest.json")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["files"][0]["source"]["filename"] = "../private/predictions.jsonl"
    path = tmp_path / "rescore_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RescoreError, match="source filename"):
        verify_corrected_pack(
            corrected_dir=Path("results/corrected"),
            archived_audit_dir=Path("results/audit"),
            manifest_path=path,
        )


def test_corrected_verifier_rejects_undeclared_bundle_files(tmp_path):
    corrected_dir = tmp_path / "corrected"
    shutil.copytree("results/corrected", corrected_dir)
    (corrected_dir / "private-notes.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RescoreError, match="unexpected corrected bundle files"):
        verify_corrected_pack(corrected_dir, Path("results/audit"))
