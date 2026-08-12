import subprocess
import gzip
import io
import json
import tarfile
import zipfile

import pytest

from vlmeval.release import ReleaseError, verify_distribution, verify_release_tree


def test_release_scan_rejects_private_files_and_local_paths(tmp_path):
    (tmp_path / "plan.md").write_text("internal plan", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "built at C:" + r"\Users\private\workspace",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseError) as exc_info:
        verify_release_tree(tmp_path)

    message = str(exc_info.value)
    assert "plan.md" in message
    assert "absolute local path" in message


def test_release_scan_rejects_credentials(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "OPENAI_API_KEY=" + "sk-" + "abcdefghijklmnopqrstuvwxyz123456",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseError, match="credential-like value"):
        verify_release_tree(tmp_path)


def test_release_scan_accepts_blank_environment_example(tmp_path):
    (tmp_path / ".env.example").write_text(
        "OPENAI_API_KEY=\nANTHROPIC_API_KEY=\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("public", encoding="utf-8")

    summary = verify_release_tree(tmp_path)

    assert summary["files"] == 2
    assert summary["bytes"] > 0


def test_release_scan_includes_untracked_nonignored_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("public", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    (tmp_path / "notes.txt").write_text(
        "OPENAI_API_KEY=" + "sk-" + "abcdefghijklmnopqrstuvwxyz123456",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseError, match="credential-like value"):
        verify_release_tree(tmp_path)


def test_distribution_scan_rejects_private_artifact_path(tmp_path):
    archive = tmp_path / "bad.whl"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("vlmeval/__init__.py", "")
        bundle.writestr("results/predictions/private.jsonl", "{}")

    with pytest.raises(ReleaseError, match="private artifact path"):
        verify_distribution(archive)


def test_release_scan_validates_corrected_gzip_schema(tmp_path):
    corrected_dir = tmp_path / "results" / "corrected"
    corrected_dir.mkdir(parents=True)
    row = {
        "sample_id": "opaque",
        "model": "model",
        "task": "task",
        "score": 1.0,
        "usage": {"input_tokens": 1, "output_tokens": 1, "source": "api"},
        "cost_usd": 0.01,
        "latency": {"seconds": 0.1, "cached": False},
        "error": None,
        "prediction": "must not ship",
    }
    with gzip.open(corrected_dir / "bad.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")

    with pytest.raises(ReleaseError, match="invalid public audit rows"):
        verify_release_tree(tmp_path)


def test_sdist_requires_corrected_verifier_and_manifest(tmp_path):
    archive = tmp_path / "source.tar.gz"
    required_before_correction = (
        "pkg/LICENSE",
        "pkg/README.md",
        "pkg/pyproject.toml",
        "pkg/results/audit/run_manifest.json",
        "pkg/scripts/verify_audit.py",
        "pkg/scripts/verify_release.py",
    )
    with tarfile.open(archive, "w:gz") as bundle:
        for name in required_before_correction:
            payload = b"public"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ReleaseError, match="results/corrected/rescore_manifest.json"):
        verify_distribution(archive)
