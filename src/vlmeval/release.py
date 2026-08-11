"""Repository-level checks for a public, privacy-safe release tree."""

from __future__ import annotations

import re
import subprocess
import tarfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable

from vlmeval.audit import read_audit_rows

MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
IGNORED_WORK_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
FORBIDDEN_NAMES = {
    ".env",
    "agents.md",
    "claude.md",
    "gemini.md",
    "plan.md",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".key",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
}
FORBIDDEN_PARTS = {
    ("data", "raw"),
    ("results", "predictions"),
    ("results", "raw"),
    ("results", "summaries"),
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".example",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|/" + "home/" + r"[^/\s]+/)",
    re.IGNORECASE,
)
CREDENTIAL_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?im)^[ \t]*(?:ANTHROPIC_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|HF_TOKEN|OPENAI_API_KEY)"
        r"[ \t]*=[ \t]*\S+"
    ),
)


class ReleaseError(ValueError):
    """Raised when a release tree contains unsafe or unauditable material."""


def _tracked_or_walked_files(root: Path) -> Iterable[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        for raw_path in result.stdout.split(b"\0"):
            if raw_path:
                yield root / raw_path.decode("utf-8")
        return

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in IGNORED_WORK_DIRS for part in relative.parts):
            continue
        if path.is_file():
            yield path


def _has_forbidden_part(parts: tuple[str, ...]) -> bool:
    lowered = tuple(part.lower() for part in parts)
    return any(
        lowered[index : index + len(forbidden)] == forbidden
        for forbidden in FORBIDDEN_PARTS
        for index in range(len(lowered) - len(forbidden) + 1)
    )


def verify_release_tree(root: Path) -> dict[str, int]:
    """Fail closed on private files, secrets, workstation paths, and oversized files."""
    root = root.resolve()
    if not root.is_dir():
        raise ReleaseError(f"release root is not a directory: {root}")

    findings: list[str] = []
    files = sorted(_tracked_or_walked_files(root), key=lambda path: path.as_posix())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        lowered_name = path.name.lower()
        size = path.stat().st_size
        total_bytes += size

        if lowered_name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden public file: {relative_text}")
        if _has_forbidden_part(relative.parts):
            findings.append(f"forbidden private artifact path: {relative_text}")
        if size > MAX_PUBLIC_FILE_BYTES:
            findings.append(f"oversized public file: {relative_text} ({size} bytes)")

        if path.suffix.lower() not in TEXT_SUFFIXES and lowered_name != ".env.example":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 text file: {relative_text}")
            continue
        if LOCAL_PATH_RE.search(content):
            findings.append(f"absolute local path in {relative_text}")
        if any(pattern.search(content) for pattern in CREDENTIAL_RES):
            findings.append(f"credential-like value in {relative_text}")

    leaderboard = root / "results" / "leaderboard.md"
    if leaderboard.is_file():
        markdown = leaderboard.read_text(encoding="utf-8")
        if re.search(r"(?im)^## Representative error cases\s*$", markdown) or re.search(
            r"(?im)^\s*-\s+(?:Q|expected|got):", markdown
        ):
            findings.append("dataset or prediction text in results/leaderboard.md")

    audit_dir = root / "results" / "audit"
    if audit_dir.is_dir():
        for path in sorted(audit_dir.glob("*.jsonl.gz")):
            try:
                read_audit_rows(path)
            except (OSError, ValueError) as exc:
                findings.append(f"invalid public audit rows in {path.relative_to(root)}: {exc}")

    if findings:
        raise ReleaseError("release verification failed:\n- " + "\n- ".join(findings))
    return {"files": len(files), "bytes": total_bytes}


def _verify_archive_names(names_and_sizes: list[tuple[str, int]], *, wheel: bool) -> dict[str, int]:
    findings: list[str] = []
    files = 0
    total_bytes = 0
    normalized_names: list[str] = []
    for name, size in names_and_sizes:
        path = PurePosixPath(name)
        if name.endswith("/"):
            continue
        files += 1
        total_bytes += size
        normalized_names.append(name)
        if path.is_absolute() or ".." in path.parts:
            findings.append(f"unsafe archive path: {name}")
            continue
        public_parts = path.parts if wheel else path.parts[1:]
        if not public_parts:
            findings.append(f"unexpected archive member: {name}")
            continue
        public_name = public_parts[-1].lower()
        public_path = "/".join(public_parts)
        if (
            public_name in FORBIDDEN_NAMES
            or PurePosixPath(public_name).suffix in FORBIDDEN_SUFFIXES
        ):
            findings.append(f"forbidden public file: {public_path}")
        if _has_forbidden_part(tuple(public_parts)):
            findings.append(f"forbidden private artifact path: {public_path}")
        if size > MAX_PUBLIC_FILE_BYTES:
            findings.append(f"oversized public file: {public_path} ({size} bytes)")
        if wheel and public_parts[0] != "vlmeval" and ".dist-info" not in public_parts[0]:
            findings.append(f"unexpected wheel content: {public_path}")

    required_suffixes = (
        ("vlmeval/__init__.py", ".dist-info/METADATA", ".dist-info/licenses/LICENSE")
        if wheel
        else (
            "/LICENSE",
            "/README.md",
            "/pyproject.toml",
            "/results/audit/run_manifest.json",
            "/scripts/verify_audit.py",
            "/scripts/verify_release.py",
        )
    )
    for suffix in required_suffixes:
        if not any(name == suffix or name.endswith(suffix) for name in normalized_names):
            findings.append(f"required distribution file missing: {suffix}")
    if findings:
        raise ReleaseError("distribution verification failed:\n- " + "\n- ".join(findings))
    return {"files": files, "bytes": total_bytes}


def verify_distribution(path: Path) -> dict[str, int]:
    """Inspect wheel/sdist member names, sizes, and required public content."""
    path = path.resolve()
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            members = [(entry.filename, entry.file_size) for entry in archive.infolist()]
        return _verify_archive_names(members, wheel=True)
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            links = [entry.name for entry in archive.getmembers() if entry.issym() or entry.islnk()]
            if links:
                raise ReleaseError(f"distribution contains links: {links}")
            members = [(entry.name, entry.size) for entry in archive.getmembers() if entry.isfile()]
        return _verify_archive_names(members, wheel=False)
    raise ReleaseError(f"unsupported distribution type: {path.name}")
