"""Build and verify the privacy-safe, deterministic result audit bundle."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = 1
PORTFOLIO_MODELS = (
    "qwen3vl-8b-base",
    "qwen3vl-8b-receipt-qlora",
    "gemini-3.1-flash-lite",
    "gpt-5.4-mini",
)
AUDIT_ROW_KEYS = {
    "sample_id",
    "model",
    "task",
    "score",
    "prediction",
    "usage",
    "cost_usd",
    "latency",
    "error",
}


class AuditError(ValueError):
    """Raised when an audit bundle is incomplete or internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_latest_rows(path: Path) -> list[dict[str, Any]]:
    """Read the last row per sample ID, preserving first-seen ID order.

    This is intentionally the same rule used by the leaderboard reporter: a
    retry replaces its earlier failed row without moving that sample in the
    deterministic run order.
    """
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                sample_id = str(row["sample_id"])
            except KeyError as exc:
                raise AuditError(f"{path}:{line_number}: missing sample_id") from exc
            by_id[sample_id] = row
    return list(by_id.values())


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    prediction = row.get("pred_clean")
    if prediction is None:
        prediction = row.get("pred_raw")
    return {
        "sample_id": str(row["sample_id"]),
        "model": row["model"],
        "task": row["task"],
        "score": row["score"],
        "prediction": prediction,
        "usage": {
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "source": row.get("usage_source"),
        },
        "cost_usd": row.get("cost_usd"),
        "latency": {
            "seconds": row.get("latency_s"),
            "cached": bool(row.get("cached")),
        },
        "error": row.get("error"),
    }


def serialize_audit_rows(rows: list[dict[str, Any]]) -> bytes:
    """Return byte-stable gzip JSONL (fixed gzip timestamp and canonical JSON)."""
    payload = b"".join(
        (
            json.dumps(_audit_row(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
        stream.write(payload)
    return output.getvalue()


def read_audit_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != AUDIT_ROW_KEYS:
                raise AuditError(
                    f"{path}:{line_number}: unexpected keys "
                    f"{sorted(set(row) - AUDIT_ROW_KEYS)}; missing {sorted(AUDIT_ROW_KEYS - set(row))}"
                )
            nested_schemas = {
                "usage": {"input_tokens", "output_tokens", "source"},
                "latency": {"seconds", "cached"},
            }
            for field, allowed_keys in nested_schemas.items():
                value = row.get(field)
                if not isinstance(value, dict) or set(value) != allowed_keys:
                    actual_keys = sorted(value) if isinstance(value, dict) else type(value).__name__
                    raise AuditError(
                        f"{path}:{line_number}: unexpected {field} schema {actual_keys}"
                    )
            rows.append(row)
    return rows


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def public_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Statistics that can be recomputed without dataset references."""
    scores = [float(row["score"]) for row in rows]
    costs = [float(row.get("cost_usd") or 0.0) for row in rows]
    input_tokens = [int(row["usage"].get("input_tokens") or 0) for row in rows]
    output_tokens = [int(row["usage"].get("output_tokens") or 0) for row in rows]
    uncached_latencies = [
        float(row["latency"]["seconds"])
        for row in rows
        if not row["latency"].get("cached")
        and not row.get("error")
        and row["latency"].get("seconds")
    ]
    return {
        "row_count": len(rows),
        "unique_sample_count": len({row["sample_id"] for row in rows}),
        "per_sample_mean_score": round(sum(scores) / len(scores), 8) if scores else 0.0,
        "error_count": sum(bool(row.get("error")) for row in rows),
        "cost_usd": round(sum(costs), 12),
        "input_tokens": sum(input_tokens),
        "output_tokens": sum(output_tokens),
        "uncached_latency": {
            "count": len(uncached_latencies),
            "mean_s": round(sum(uncached_latencies) / len(uncached_latencies), 8)
            if uncached_latencies
            else 0.0,
            "p50_s": round(_percentile(uncached_latencies, 50), 8),
            "p95_s": round(_percentile(uncached_latencies, 95), 8),
        },
    }


def build_audit_pack(
    config_path: Path,
    predictions_dir: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    """Generate the committed bundle from complete, local raw prediction files."""
    from vlmeval.config import load_config
    from vlmeval.tasks import build_task

    config_path = config_path.resolve()
    project_root = config_path.parent
    predictions_dir = predictions_dir.resolve()
    audit_dir = audit_dir.resolve()
    cfg = load_config(config_path)
    tasks = [build_task(task_cfg, cfg.run) for task_cfg in cfg.tasks]
    task_by_name = {task.name: task for task in tasks}
    expected_counts = {task_cfg.name: task_cfg.n_full for task_cfg in cfg.tasks}
    configured_models = {model.id for model in cfg.models}
    missing_models = set(PORTFOLIO_MODELS) - configured_models
    if missing_models:
        raise AuditError(f"models missing from config: {sorted(missing_models)}")

    audit_dir.mkdir(parents=True, exist_ok=True)
    for old_file in audit_dir.glob("*.jsonl.gz"):
        old_file.unlink()

    files: list[dict[str, Any]] = []
    total_rows = 0
    for model_id in PORTFOLIO_MODELS:
        for task_cfg in cfg.tasks:
            task_name = task_cfg.name
            source_path = predictions_dir / f"{model_id}__{task_name}.jsonl"
            if not source_path.exists():
                raise AuditError(f"missing source predictions: {source_path}")
            rows = read_latest_rows(source_path)
            expected = expected_counts[task_name]
            if len(rows) != expected:
                raise AuditError(
                    f"{source_path}: expected {expected} unique rows, found {len(rows)}"
                )
            for row in rows:
                if row.get("model") != model_id or row.get("task") != task_name:
                    raise AuditError(
                        f"{source_path}: row {row.get('sample_id')} has mismatched model/task"
                    )

            output_name = f"{model_id}__{task_name}.jsonl.gz"
            output_path = audit_dir / output_name
            output_path.write_bytes(serialize_audit_rows(rows))
            safe_rows = read_audit_rows(output_path)
            aggregate = task_by_name[task_name].aggregate(rows)
            files.append(
                {
                    "path": output_name,
                    "sha256": sha256_file(output_path),
                    "source_prediction_path": source_path.relative_to(project_root).as_posix(),
                    "source_prediction_sha256": sha256_file(source_path),
                    "model": model_id,
                    "task": task_name,
                    "aggregate_from_raw_rows": aggregate,
                    "public_stats": public_stats(safe_rows),
                }
            )
            total_rows += len(rows)

    sample_manifests: dict[str, dict[str, str]] = {}
    for task_cfg in cfg.tasks:
        path = project_root / "data" / "samples" / f"{task_cfg.name}_seed{cfg.run.seed}.json"
        if not path.exists():
            raise AuditError(f"missing committed sample manifest: {path}")
        sample_manifests[task_cfg.name] = {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
        }

    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generator": {
            "command": "uv run python scripts/build_audit_pack.py",
            "deduplication": (
                "Last JSONL row wins for each sample_id; first-seen sample order is preserved "
                "(identical to the leaderboard reporter)."
            ),
        },
        "configuration": {
            "path": config_path.name,
            "sha256": sha256_file(config_path),
            "seed": cfg.run.seed,
            "bootstrap_iterations": cfg.run.bootstrap_iters,
        },
        "sample_manifests": sample_manifests,
        "scope": {
            "models": list(PORTFOLIO_MODELS),
            "task_order": [task_cfg.name for task_cfg in cfg.tasks],
            "tasks": expected_counts,
            "total_rows": total_rows,
            "excluded_models": [
                "gemini-2.5-flash-lite (disabled, incomplete free-tier run)",
                "claude-haiku-4-5 (disabled, no complete billed run)",
            ],
        },
        "files": files,
        "hardware_and_method": {
            "local_models": "RTX 4090, WSL2, 4-bit, batch size 1; no network latency.",
            "api_models": "Provider round-trip latency and returned billed usage from the run.",
            "cost": (
                "Local dollars are imputed at $0.35/hour from measured inference time; "
                "API dollars use provider-returned usage and config pricing."
            ),
        },
        "verification_boundary": [
            "Dataset images, prompts, and references are intentionally excluded from this pack.",
            (
                "DocVQA and ChartQA mean scores, all row counts, costs, token totals, errors, "
                "and uncached latency summaries are independently recomputable from the pack."
            ),
            (
                "CORD uses corpus-level micro F1 and bootstrap intervals, which require references. "
                "Those aggregate values are recorded with hashes of the ignored raw sources, but "
                "cannot be independently rescored from this license-safe pack alone."
            ),
        ],
    }
    manifest_path = audit_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _close(actual: float, expected: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _markdown_table(markdown: str, heading: str) -> dict[str, list[str]]:
    lines = markdown.splitlines()
    try:
        start = lines.index(heading)
    except ValueError as exc:
        raise AuditError(f"leaderboard is missing {heading}") from exc
    rows: dict[str, list[str]] = {}
    found_header = False
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not found_header:
            found_header = True
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows[cells[0]] = cells[1:]
    return rows


def _find_model_row(table: dict[str, list[str]], model_id: str) -> list[str]:
    for label, cells in table.items():
        if label.startswith(model_id):
            return cells
    raise AuditError(f"leaderboard table is missing model {model_id}")


def _first_number(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        raise AuditError(f"could not parse numeric value from {value!r}")
    return float(match.group())


def verify_audit_pack(audit_dir: Path, leaderboard_path: Path | None = None) -> dict[str, Any]:
    """Verify bundle hashes/content and consistency with the committed leaderboard."""
    audit_dir = audit_dir.resolve()
    manifest_path = audit_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise AuditError(f"unsupported schema version in {manifest_path}")

    project_root = (
        leaderboard_path.resolve().parent.parent
        if leaderboard_path is not None
        else audit_dir.parent.parent
    )

    def verify_project_file(entry: dict[str, str], label: str) -> None:
        path = (project_root / entry["path"]).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise AuditError(f"{label} path escapes the project root: {entry['path']}") from exc
        if not path.is_file():
            raise AuditError(f"missing {label}: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise AuditError(f"SHA-256 mismatch for {label}: {path}")

    verify_project_file(manifest["configuration"], "configuration")
    for task_name, entry in manifest["sample_manifests"].items():
        verify_project_file(entry, f"sample manifest {task_name}")

    expected_models = manifest["scope"]["models"]
    expected_tasks = manifest["scope"]["tasks"]
    file_entries = manifest["files"]
    declared_files = {entry["path"] for entry in file_entries}
    bundled_files = {
        path.relative_to(audit_dir).as_posix() for path in audit_dir.rglob("*.jsonl.gz")
    }
    if bundled_files != declared_files:
        raise AuditError(f"undeclared or missing audit files: {bundled_files ^ declared_files}")
    allowed_bundle_files = declared_files | {
        "README.md",
        "evaluation_config.yaml",
        "run_manifest.json",
    }
    actual_bundle_files = {
        path.relative_to(audit_dir).as_posix() for path in audit_dir.rglob("*") if path.is_file()
    }
    if actual_bundle_files != allowed_bundle_files:
        raise AuditError(
            f"unexpected or missing files in audit directory: "
            f"{actual_bundle_files ^ allowed_bundle_files}"
        )
    expected_pairs = {(model, task) for model in expected_models for task in expected_tasks}
    actual_pairs = {(entry["model"], entry["task"]) for entry in file_entries}
    if actual_pairs != expected_pairs:
        raise AuditError(f"model/task coverage mismatch: {actual_pairs ^ expected_pairs}")

    verified_rows = 0
    entries_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    rows_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in file_entries:
        path = audit_dir / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise AuditError(f"SHA-256 mismatch: {path}")
        rows = read_audit_rows(path)
        pair = (entry["model"], entry["task"])
        if any((row["model"], row["task"]) != pair for row in rows):
            raise AuditError(f"model/task mismatch inside {path}")
        sample_ids = [row["sample_id"] for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise AuditError(f"duplicate sample IDs inside {path}")
        if len(rows) != expected_tasks[entry["task"]]:
            raise AuditError(f"unexpected row count inside {path}: {len(rows)}")
        recomputed = public_stats(rows)
        if recomputed != entry["public_stats"]:
            raise AuditError(f"public statistics mismatch: {path}")
        if entry["task"] in {"docvqa", "chartqa"} and not _close(
            round(recomputed["per_sample_mean_score"], 4),
            entry["aggregate_from_raw_rows"]["score"],
        ):
            raise AuditError(f"independently recomputed score mismatch: {path}")
        entries_by_pair[pair] = entry
        rows_by_pair[pair] = rows
        verified_rows += len(rows)

    if verified_rows != manifest["scope"]["total_rows"]:
        raise AuditError(
            f"total row mismatch: verified {verified_rows}, "
            f"manifest says {manifest['scope']['total_rows']}"
        )

    if leaderboard_path is not None:
        markdown = leaderboard_path.read_text(encoding="utf-8")
        score_table = _markdown_table(markdown, "## Scores")
        cost_table = _markdown_table(markdown, "## Cost")
        task_names = manifest["scope"].get("task_order") or [
            entry["task"] for entry in file_entries if entry["model"] == expected_models[0]
        ]
        for model_id in expected_models:
            score_cells = _find_model_row(score_table, model_id)
            task_scores = []
            for index, task_name in enumerate(task_names):
                aggregate = entries_by_pair[(model_id, task_name)]["aggregate_from_raw_rows"]
                displayed_score = _first_number(score_cells[index])
                if not _close(displayed_score, round(aggregate["score"], 3)):
                    raise AuditError(f"leaderboard score mismatch: {model_id}/{task_name}")
                task_scores.append(float(aggregate["score"]))
            displayed_average = _first_number(score_cells[len(task_names)])
            if not _close(displayed_average, round(sum(task_scores) / len(task_scores), 3)):
                raise AuditError(f"leaderboard average mismatch: {model_id}")

            model_rows = [
                row for task_name in task_names for row in rows_by_pair[(model_id, task_name)]
            ]
            totals = public_stats(model_rows)
            cost_cells = _find_model_row(cost_table, model_id)
            if not _close(_first_number(cost_cells[0]), round(totals["cost_usd"], 4)):
                raise AuditError(f"leaderboard total cost mismatch: {model_id}")
            expected_per_100 = totals["cost_usd"] / len(model_rows) * 100
            if not _close(_first_number(cost_cells[1]), round(expected_per_100, 4)):
                raise AuditError(f"leaderboard cost/100 mismatch: {model_id}")
            if int(_first_number(cost_cells[2])) != totals["input_tokens"]:
                raise AuditError(f"leaderboard input-token mismatch: {model_id}")
            if int(_first_number(cost_cells[3])) != totals["output_tokens"]:
                raise AuditError(f"leaderboard output-token mismatch: {model_id}")

    return {
        "models": len(expected_models),
        "tasks": len(expected_tasks),
        "files": len(file_entries),
        "rows": verified_rows,
        "leaderboard_checked": leaderboard_path is not None,
    }
