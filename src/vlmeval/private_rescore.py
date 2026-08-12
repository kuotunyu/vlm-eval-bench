"""Offline rescoring of complete private prediction matrices.

Private prompts, references, and predictions are read in place and never copied
to the output. Only strict audit rows and a machine-readable comparison manifest
are written.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from vlmeval.audit import (
    _audit_row,
    _close,
    _find_model_row,
    _first_number,
    _markdown_table,
    _validate_audit_row,
    public_stats,
    read_audit_rows,
    serialize_audit_rows,
    sha256_file,
)
from vlmeval.config import AppConfig
from vlmeval.metrics.anls import anls_score
from vlmeval.metrics.bootstrap import bootstrap_ci
from vlmeval.metrics.relaxed_acc import relaxed_correct
from vlmeval.tasks import build_task
from vlmeval.vendored.metrics import compute_metrics
from vlmeval.vendored.schema import tolerant_json_parse

RESCORE_SCHEMA_VERSION = 1
PRIVATE_REQUIRED_KEYS = {
    "sample_id",
    "model",
    "task",
    "pred_raw",
    "reference",
    "score",
    "input_tokens",
    "output_tokens",
    "usage_source",
    "cost_usd",
    "latency_s",
    "cached",
    "error",
}


class RescoreError(ValueError):
    """Raised when private evidence is incomplete, malformed, or ambiguous."""


def _published_numbers_match(text: str, expected: list[float]) -> bool:
    tokens = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if len(tokens) != len(expected):
        return False
    for token, value in zip(tokens, expected):
        decimals = len(token.partition(".")[2]) if "." in token else 0
        quantum = Decimal(1).scaleb(-decimals)
        displayed = Decimal(token)
        wanted = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
        if displayed != wanted:
            return False
    return True


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_private_latest(path: Path) -> tuple[list[dict[str, Any]], int]:
    by_id: dict[str, dict[str, Any]] = {}
    physical_rows = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            physical_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RescoreError(f"{path}:{line_number}: malformed JSON") from exc
            if not isinstance(row, dict):
                raise RescoreError(f"{path}:{line_number}: row must be an object")
            missing = PRIVATE_REQUIRED_KEYS - set(row)
            if missing:
                raise RescoreError(f"{path}:{line_number}: missing keys {sorted(missing)}")
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                raise RescoreError(f"{path}:{line_number}: sample_id must be a non-empty string")
            by_id[sample_id] = row
    return list(by_id.values()), physical_rows


def _validate_accounting(row: dict[str, Any], context: str) -> None:
    for field in ("input_tokens", "output_tokens"):
        value = row[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise RescoreError(f"{context}: {field} must be a non-negative integer or null")
    if not isinstance(row["usage_source"], str) or not row["usage_source"]:
        raise RescoreError(f"{context}: usage_source must be a non-empty string")
    for field in ("cost_usd", "latency_s"):
        value = row[field]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise RescoreError(f"{context}: {field} must be finite and non-negative or null")
    if not isinstance(row["cached"], bool):
        raise RescoreError(f"{context}: cached must be a boolean")
    if row["error"] is not None and not isinstance(row["error"], str):
        raise RescoreError(f"{context}: error must be a string or null")


def _correct_row(row: dict[str, Any], task_name: str, context: str) -> dict[str, Any]:
    corrected = dict(row)
    _validate_accounting(corrected, context)
    historical_score = corrected["score"]
    if (
        isinstance(historical_score, bool)
        or not isinstance(historical_score, (int, float))
        or not math.isfinite(historical_score)
        or not 0.0 <= historical_score <= 1.0
    ):
        raise RescoreError(f"{context}: historical score must be finite and in [0, 1]")
    pred_raw = corrected["pred_raw"]
    if not isinstance(pred_raw, str):
        raise RescoreError(f"{context}: pred_raw must be a string")

    if corrected["error"]:
        corrected["score"] = 0.0
        return corrected
    if task_name == "docvqa":
        prediction = corrected.get("pred_clean")
        reference = corrected["reference"]
        if (
            not isinstance(prediction, str)
            or not isinstance(reference, list)
            or any(not isinstance(answer, str) for answer in reference)
        ):
            raise RescoreError(f"{context}: invalid DocVQA prediction/reference")
        corrected["score"] = anls_score(prediction, reference)
    elif task_name == "chartqa":
        prediction = corrected.get("pred_clean")
        reference = corrected["reference"]
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise RescoreError(f"{context}: invalid ChartQA prediction/reference")
        corrected["score"] = 1.0 if relaxed_correct(prediction, reference) else 0.0
    elif task_name == "cord":
        reference = corrected["reference"]
        if not isinstance(reference, dict):
            raise RescoreError(f"{context}: invalid CORD reference")
        parsed = tolerant_json_parse(pred_raw)
        corrected["aux"] = {**corrected.get("aux", {}), "parsed": parsed}
        corrected["score"] = compute_metrics([{"gt": reference, "parsed": parsed}])["overall"]["f1"]
    else:  # pragma: no cover - config validation owns the supported task set
        raise RescoreError(f"{context}: unsupported task {task_name}")
    return corrected


def _load_archived_entries(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    try:
        return {(entry["model"], entry["task"]): entry for entry in manifest["files"]}
    except (KeyError, TypeError) as exc:
        raise RescoreError(f"malformed archived manifest: {path}") from exc


def recompute_private_run(
    config: AppConfig,
    input_dir: Path,
    output_dir: Path,
    archived_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Rescore one complete matrix without changing inference/accounting evidence."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise RescoreError(f"private input directory does not exist: {input_dir}")
    if (
        input_dir == output_dir
        or _is_within(output_dir, input_dir)
        or _is_within(input_dir, output_dir)
    ):
        raise RescoreError("input and output directories must be isolated")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RescoreError(f"output directory must be absent or empty: {output_dir}")
    models = [model.id for model in config.enabled_models()]
    task_configs = list(config.tasks)
    expected_pairs = {(model_id, task.name) for model_id in models for task in task_configs}
    expected_input_files = {
        f"{model_id}__{task.name}.jsonl" for model_id in models for task in task_configs
    }
    excluded_input_files = sorted(
        path.name for path in input_dir.glob("*.jsonl") if path.name not in expected_input_files
    )
    task_by_name = {task.name: build_task(task, config.run) for task in task_configs}
    archived_entries = _load_archived_entries(
        archived_manifest_path.resolve() if archived_manifest_path is not None else None
    )
    if archived_manifest_path is not None and set(archived_entries) != expected_pairs:
        raise RescoreError("archived model/task coverage does not match enabled config")

    file_entries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    pending_files: dict[str, bytes] = {}
    total_rows = 0
    for model_id in models:
        for task_config in task_configs:
            task_name = task_config.name
            source_path = input_dir / f"{model_id}__{task_name}.jsonl"
            if not source_path.is_file():
                raise RescoreError(f"missing private prediction: {source_path.name}")
            source_sha256 = sha256_file(source_path)
            archived = archived_entries.get((model_id, task_name))
            if archived is not None and archived.get("source_prediction_sha256") != source_sha256:
                raise RescoreError(f"private source hash mismatch: {source_path.name}")

            rows, physical_rows = _read_private_latest(source_path)
            if len(rows) != task_config.n_full:
                raise RescoreError(
                    f"{source_path.name}: expected {task_config.n_full} latest rows, found {len(rows)}"
                )
            for row in rows:
                if row["model"] != model_id or row["task"] != task_name:
                    raise RescoreError(
                        f"{source_path.name}: row {row['sample_id']} has model/task mismatch"
                    )

            if archived is not None:
                archived_audit_path = archived_manifest_path.resolve().parent / archived["path"]
                archived_ids = {row["sample_id"] for row in read_audit_rows(archived_audit_path)}
                if {row["sample_id"] for row in rows} != archived_ids:
                    raise RescoreError(f"private sample IDs mismatch: {source_path.name}")

            corrected_rows = [
                _correct_row(row, task_name, f"{source_path.name}:{row['sample_id']}")
                for row in rows
            ]
            task = task_by_name[task_name]
            corrected_aggregate = task.aggregate(corrected_rows)
            historical_aggregate = (
                archived["aggregate_from_raw_rows"]
                if archived is not None
                else task.aggregate(rows)
            )
            changed_rows = sum(
                not math.isclose(
                    float(before["score"]),
                    float(after["score"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for before, after in zip(rows, corrected_rows)
            )
            comparison = {
                "model": model_id,
                "task": task_name,
                "changed_rows": changed_rows,
                "historical_score": historical_aggregate["score"],
                "corrected_score": corrected_aggregate["score"],
                "delta": round(corrected_aggregate["score"] - historical_aggregate["score"], 4),
            }
            comparisons.append(comparison)

            output_name = f"{model_id}__{task_name}.jsonl.gz"
            payload = serialize_audit_rows(corrected_rows)
            public_rows = [_audit_row(row) for row in corrected_rows]
            for index, row in enumerate(public_rows, start=1):
                _validate_audit_row(row, f"{output_name}:{index}")
            pending_files[output_name] = payload
            file_entries.append(
                {
                    "path": output_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "model": model_id,
                    "task": task_name,
                    "source": {
                        "filename": source_path.name,
                        "sha256": source_sha256,
                        "physical_rows": physical_rows,
                        "latest_rows": len(rows),
                    },
                    "historical_aggregate": historical_aggregate,
                    "corrected_aggregate": corrected_aggregate,
                    "public_stats": public_stats(public_rows),
                }
            )
            total_rows += len(rows)

    manifest: dict[str, Any] = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "kind": "corrected_offline_rescore",
        "inference": "Unchanged 2026-07-10 predictions; scoring-only recomputation.",
        "deduplication": (
            "Last JSONL row wins per sample_id; first-seen sample order is preserved."
        ),
        "scoring": {
            "docvqa": "case-insensitive, space-sensitive ANLS; normalized distance < 0.5",
            "chartqa": "Pix2Struct-compatible relaxed correctness; 5% numeric tolerance",
            "cord": "unchanged reference-dependent corpus-level field micro F1",
            "bootstrap_iterations": config.run.bootstrap_iters,
            "bootstrap_seed": config.run.seed,
        },
        "scope": {
            "models": models,
            "tasks": {task.name: task.n_full for task in task_configs},
            "task_order": [task.name for task in task_configs],
            "total_rows": total_rows,
            "excluded_input_files": excluded_input_files,
        },
        "comparisons": comparisons,
        "files": file_entries,
        "public_boundary": (
            "Rows exclude questions, prompts, references, predictions, raw provider responses, "
            "dataset images, and credentials."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_name, payload in pending_files.items():
        (output_dir / output_name).write_bytes(payload)
    (output_dir / "rescore_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _verify_score_tables(
    markdown: str,
    heading: str,
    manifest: dict[str, Any],
    files_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> None:
    table = _markdown_table(markdown, heading)
    task_order = manifest["scope"]["task_order"]
    for model_id in manifest["scope"]["models"]:
        cells = _find_model_row(table, model_id)
        task_scores: list[float] = []
        for index, task_name in enumerate(task_order):
            aggregate = files_by_pair[(model_id, task_name)]["corrected_aggregate"]
            expected = [
                float(aggregate["score"]),
                float(aggregate["ci95"][0]),
                float(aggregate["ci95"][1]),
            ]
            if not _published_numbers_match(cells[index], expected):
                raise RescoreError(f"corrected score claim mismatch: {model_id}/{task_name}")
            task_scores.append(float(aggregate["score"]))
        expected_average = sum(task_scores) / len(task_scores)
        if not _published_numbers_match(cells[len(task_order)], [expected_average]):
            raise RescoreError(f"corrected average claim mismatch: {model_id}")


def _verify_comparison_table(
    markdown: str,
    manifest: dict[str, Any],
) -> None:
    table = _markdown_table(markdown, "## Historical vs corrected")
    for comparison in manifest["comparisons"]:
        label = f"{comparison['model']} / {comparison['task']}"
        cells = table.get(label)
        if cells is None:
            raise RescoreError(f"comparison row missing: {label}")
        displayed = [_first_number(cell) for cell in cells[:4]]
        expected = [
            round(float(comparison["historical_score"]), 4),
            round(float(comparison["corrected_score"]), 4),
            round(float(comparison["delta"]), 4),
            int(comparison["changed_rows"]),
        ]
        if any(not _close(actual, wanted) for actual, wanted in zip(displayed, expected)):
            raise RescoreError(f"comparison claim mismatch: {label}")


def _verify_accounting_tables(
    markdown: str,
    manifest: dict[str, Any],
    rows_by_pair: dict[tuple[str, str], list[dict[str, Any]]],
    files_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> None:
    cost_table = _markdown_table(markdown, "## Cost")
    reliability_table = _markdown_table(markdown, "## Reliability")
    task_order = manifest["scope"]["task_order"]
    models = manifest["scope"]["models"]
    for model_id in models:
        rows = [row for task in task_order for row in rows_by_pair[(model_id, task)]]
        totals = public_stats(rows)
        cost_cells = _find_model_row(cost_table, model_id)
        expected_cost = [
            round(totals["cost_usd"], 4),
            round(totals["cost_usd"] / len(rows) * 100, 4),
            totals["input_tokens"],
            totals["output_tokens"],
        ]
        displayed_cost = [_first_number(cell) for cell in cost_cells[:4]]
        if any(
            not _close(actual, expected) for actual, expected in zip(displayed_cost, expected_cost)
        ):
            raise RescoreError(f"corrected cost claim mismatch: {model_id}")

        reliability_cells = _find_model_row(reliability_table, model_id)
        expected_error = round(totals["error_count"] / len(rows) * 100, 2)
        expected_valid = round(
            float(files_by_pair[(model_id, "cord")]["corrected_aggregate"]["valid_json_rate"])
            * 100,
            2,
        )
        if not _close(_first_number(reliability_cells[0]), expected_error) or not _close(
            _first_number(reliability_cells[1]), expected_valid
        ):
            raise RescoreError(f"corrected reliability claim mismatch: {model_id}")

    latency_rows: dict[tuple[str, str], list[str]] = {}
    lines = markdown.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith("## Latency")), None)
    if start is None:
        raise RescoreError("corrected leaderboard is missing latency table")
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] == "Model" or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        model_id = next((model for model in models if cells[0].startswith(model)), None)
        if model_id is not None:
            latency_rows[(model_id, cells[1])] = cells[2:]
    for pair, entry in files_by_pair.items():
        cells = latency_rows.get(pair)
        if cells is None:
            raise RescoreError(f"corrected latency row missing: {pair[0]}/{pair[1]}")
        stats = entry["public_stats"]["uncached_latency"]
        expected = [round(stats["mean_s"], 2), round(stats["p50_s"], 2), round(stats["p95_s"], 2)]
        displayed = [_first_number(cell) for cell in cells]
        if any(not _close(actual, wanted) for actual, wanted in zip(displayed, expected)):
            raise RescoreError(f"corrected latency claim mismatch: {pair[0]}/{pair[1]}")


def verify_corrected_pack(
    corrected_dir: Path,
    archived_audit_dir: Path,
    leaderboard_path: Path | None = None,
    readme_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify corrected rows, archived equivalence, and published corrected claims."""
    corrected_dir = corrected_dir.resolve()
    archived_audit_dir = archived_audit_dir.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else corrected_dir / "rescore_manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archived_manifest = json.loads(
            (archived_audit_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RescoreError(f"could not read corrected/archived manifest: {exc}") from exc
    if manifest.get("schema_version") != RESCORE_SCHEMA_VERSION:
        raise RescoreError("unsupported corrected schema version")

    models = manifest["scope"]["models"]
    tasks = manifest["scope"]["tasks"]
    expected_pairs = {(model, task) for model in models for task in tasks}
    expected_input_files = {f"{model}__{task}.jsonl" for model, task in expected_pairs}
    archived_entries = {
        (entry["model"], entry["task"]): entry for entry in archived_manifest["files"]
    }
    if expected_pairs != set(archived_entries):
        raise RescoreError("corrected model/task scope differs from archived evidence")

    files_by_pair = {(entry["model"], entry["task"]): entry for entry in manifest["files"]}
    if set(files_by_pair) != expected_pairs or len(manifest["files"]) != len(expected_pairs):
        raise RescoreError("corrected model/task file coverage mismatch")
    for pair, entry in files_by_pair.items():
        expected_source_name = f"{pair[0]}__{pair[1]}.jsonl"
        if entry.get("source", {}).get("filename") != expected_source_name:
            raise RescoreError(f"invalid private source filename: {pair[0]}/{pair[1]}")
    excluded_inputs = manifest["scope"].get("excluded_input_files")
    if not isinstance(excluded_inputs, list) or len(excluded_inputs) != len(set(excluded_inputs)):
        raise RescoreError("invalid excluded input filename list")
    for filename in excluded_inputs:
        if (
            not isinstance(filename, str)
            or not filename.endswith(".jsonl")
            or Path(filename).name != filename
            or filename in expected_input_files
        ):
            raise RescoreError(f"invalid excluded input filename: {filename!r}")
    declared_files = {entry["path"] for entry in manifest["files"]}
    actual_files = {
        path.relative_to(corrected_dir).as_posix() for path in corrected_dir.glob("*.jsonl.gz")
    }
    if declared_files != actual_files:
        raise RescoreError(f"corrected artifact coverage mismatch: {declared_files ^ actual_files}")
    allowed_bundle_files = declared_files | {"README.md", "rescore_manifest.json"}
    actual_bundle_files = {
        path.relative_to(corrected_dir).as_posix()
        for path in corrected_dir.rglob("*")
        if path.is_file()
    }
    if actual_bundle_files != allowed_bundle_files:
        raise RescoreError(
            f"unexpected corrected bundle files: {actual_bundle_files ^ allowed_bundle_files}"
        )

    comparisons = {(entry["model"], entry["task"]): entry for entry in manifest["comparisons"]}
    if set(comparisons) != expected_pairs or len(manifest["comparisons"]) != len(expected_pairs):
        raise RescoreError("corrected comparison coverage mismatch")

    rows_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    verified_rows = 0
    for pair in sorted(expected_pairs):
        entry = files_by_pair[pair]
        corrected_path = corrected_dir / entry["path"]
        if sha256_file(corrected_path) != entry["sha256"]:
            raise RescoreError(f"corrected SHA-256 mismatch: {entry['path']}")
        corrected_rows = read_audit_rows(corrected_path)
        archived_entry = archived_entries[pair]
        archived_rows = read_audit_rows(archived_audit_dir / archived_entry["path"])
        if len(corrected_rows) != tasks[pair[1]] or len(archived_rows) != len(corrected_rows):
            raise RescoreError(f"corrected row count mismatch: {pair[0]}/{pair[1]}")
        corrected_by_id = {row["sample_id"]: row for row in corrected_rows}
        archived_by_id = {row["sample_id"]: row for row in archived_rows}
        if set(corrected_by_id) != set(archived_by_id):
            raise RescoreError(f"corrected sample coverage mismatch: {pair[0]}/{pair[1]}")
        for sample_id, corrected_row in corrected_by_id.items():
            archived_row = archived_by_id[sample_id]
            if {key: value for key, value in corrected_row.items() if key != "score"} != {
                key: value for key, value in archived_row.items() if key != "score"
            }:
                raise RescoreError(f"inference/accounting changed: {pair[0]}/{pair[1]}/{sample_id}")

        stats = public_stats(corrected_rows)
        if stats != entry["public_stats"]:
            raise RescoreError(f"corrected public statistics mismatch: {pair[0]}/{pair[1]}")
        historical = archived_entry["aggregate_from_raw_rows"]
        if entry["historical_aggregate"] != historical:
            raise RescoreError(f"historical aggregate mismatch: {pair[0]}/{pair[1]}")

        corrected_aggregate = entry["corrected_aggregate"]
        if pair[1] in {"docvqa", "chartqa"}:
            scores = [float(row["score"]) for row in corrected_rows]
            mean = round(sum(scores) / len(scores), 4)
            ci = bootstrap_ci(
                scores,
                lambda values: sum(values) / len(values),
                n_boot=manifest["scoring"]["bootstrap_iterations"],
                seed=manifest["scoring"]["bootstrap_seed"],
            )
            if (
                not _close(mean, corrected_aggregate["score"])
                or [
                    round(ci[0], 4),
                    round(ci[1], 4),
                ]
                != corrected_aggregate["ci95"]
            ):
                raise RescoreError(f"corrected score/CI mismatch: {pair[0]}/{pair[1]}")
        else:
            if corrected_aggregate != historical or any(
                not _close(corrected["score"], archived_by_id[corrected["sample_id"]]["score"])
                for corrected in corrected_rows
            ):
                raise RescoreError("CORD changed even though its scoring semantics are unchanged")

        expected_comparison = {
            "model": pair[0],
            "task": pair[1],
            "changed_rows": sum(
                not _close(row["score"], archived_by_id[row["sample_id"]]["score"])
                for row in corrected_rows
            ),
            "historical_score": historical["score"],
            "corrected_score": corrected_aggregate["score"],
            "delta": round(corrected_aggregate["score"] - historical["score"], 4),
        }
        if comparisons[pair] != expected_comparison:
            raise RescoreError(f"comparison mismatch: {pair[0]}/{pair[1]}")
        if entry["source"]["sha256"] != archived_entry["source_prediction_sha256"]:
            raise RescoreError(f"private provenance hash mismatch: {pair[0]}/{pair[1]}")
        rows_by_pair[pair] = corrected_rows
        verified_rows += len(corrected_rows)

    if verified_rows != manifest["scope"]["total_rows"]:
        raise RescoreError("corrected total row count mismatch")

    if leaderboard_path is not None:
        markdown = leaderboard_path.read_text(encoding="utf-8")
        _verify_score_tables(markdown, "## Corrected scores", manifest, files_by_pair)
        _verify_comparison_table(markdown, manifest)
        _verify_accounting_tables(markdown, manifest, rows_by_pair, files_by_pair)
    if readme_path is not None:
        markdown = readme_path.read_text(encoding="utf-8")
        _verify_score_tables(markdown, "## Corrected results", manifest, files_by_pair)

    return {
        "models": len(models),
        "tasks": len(tasks),
        "files": len(files_by_pair),
        "rows": verified_rows,
        "leaderboard_checked": leaderboard_path is not None,
        "readme_checked": readme_path is not None,
    }
