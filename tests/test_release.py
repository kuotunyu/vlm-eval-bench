import subprocess
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
