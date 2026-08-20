"""Two-seed execution helpers and descriptive A/B report aggregation."""

from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping
import zipfile

from .artifacts import read_json_file, write_json_file

REPLICATE_LABELS = ("A", "B")
RUN_B_TRAINING_SEED_OFFSET = 100_000
_TRAINING_SEED_KEYS = (
    "dir_source_seed",
    "dir_same_task_seed",
    "dir_different_task_head_seed",
    "dense_source_seed",
    "dense_same_task_seed",
    "dense_different_task_head_seed",
    "dir_source_data_order_seed",
    "dir_same_task_data_order_seed",
    "different_task_data_order_seed",
    "dense_source_data_order_seed",
    "dense_same_task_data_order_seed",
)
_AGGREGATE_SECTIONS = (
    "condition_comparisons",
    "primary_results",
    "supporting_results",
    "jacobian_rank32_results",
)
_EXCLUDED_AGGREGATE_FRAGMENTS = (
    "p_value",
    "pvalue",
    "permutation",
    "bootstrap",
    "seed",
    "tolerance",
    "threshold",
    "sample_count",
    "valid_count",
    "invalid_count",
    "module_count",
    "warning_count",
    "variant_count",
    "total_variants",
    "usable_variants",
    "epoch_count",
    "probe_count",
    "iterations",
    "bytes",
    "confidence_interval",
    "standard_error",
    "stderr",
)


def apply_training_seed_offset(base_config: Mapping[str, Any], offset: int) -> dict[str, Any]:
    """Offset only training randomness; measurement and dictionary-basis seeds stay shared."""

    config = deepcopy(dict(base_config))
    offset = int(offset)
    if offset == 0:
        return config
    plan = dict(config.get("functional_correspondence", {}) or {})
    training = dict(plan.get("training", {}) or {})
    for key in _TRAINING_SEED_KEYS:
        training[key] = int(training[key]) + offset
    plan["training"] = training
    config["functional_correspondence"] = plan

    role = dict(config.get("dictionary_reuse", {}) or {})
    runtime = dict(role.get("runtime", {}) or {})
    runtime["base_seed"] = int(runtime.get("base_seed", 0)) + offset
    runtime["data_order_seed"] = int(runtime.get("data_order_seed", 0)) + offset
    role["runtime"] = runtime
    source_run = dict(role.get("source_run", {}) or {})
    if source_run:
        source_run["seed"] = int(training["dir_source_seed"])
        role["source_run"] = source_run
    config["dictionary_reuse"] = role
    return config


def replicate_runtime_paths(
    results_root: Path,
    checkpoints_root: Path,
    *,
    batch_name: str,
    replicate_label: str,
) -> dict[str, str]:
    if replicate_label not in REPLICATE_LABELS:
        raise ValueError(f"Unknown replicate label: {replicate_label!r}")
    output_dir = (results_root / batch_name / replicate_label).resolve()
    checkpoint_dir = (checkpoints_root / batch_name / replicate_label).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": str(output_dir),
        "work_dir": str(output_dir / ".work"),
        "checkpoint_dir": str(checkpoint_dir),
        "raw_report_zip": str(output_dir / "DiR_RAW_REPORT.zip"),
        "summary_report_zip": str(output_dir / "DiR_SUMMARY_REPORT.zip"),
    }


def _base_config_sha256(base_config: Mapping[str, Any]) -> str:
    canonical = json.dumps(base_config, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_batch_name(
    results_root: Path,
    checkpoints_root: Path,
    base_config: Mapping[str, Any],
) -> tuple[str, bool]:
    """Resume the newest compatible incomplete A/B batch, otherwise allocate a new one."""

    expected_hash = _base_config_sha256(base_config)
    for candidate in sorted(results_root.glob("run_*"), reverse=True):
        if not candidate.is_dir() or not ((candidate / "A").exists() or (candidate / "B").exists()):
            continue
        if (candidate / "DiR_SUMMARY_REPORT.zip").is_file() and (candidate / "DiR_RAW_REPORT.zip").is_file():
            continue
        state_path = checkpoints_root / candidate.name / "batch_state.json"
        if not state_path.is_file():
            continue
        try:
            state = read_json_file(state_path)
        except Exception:
            continue
        if str(state.get("base_config_sha256", "")) == expected_hash:
            return candidate.name, True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"
    batch_name = base_name
    index = 2
    while (results_root / batch_name).exists() or (checkpoints_root / batch_name).exists():
        batch_name = f"{base_name}_{index:02d}"
        index += 1
    write_json_file(
        checkpoints_root / batch_name / "batch_state.json",
        {
            "batch_name": batch_name,
            "base_config_sha256": expected_hash,
            "replicates": list(REPLICATE_LABELS),
            "run_B_training_seed_offset": RUN_B_TRAINING_SEED_OFFSET,
        },
    )
    return batch_name, False


def replicate_reports_complete(
    batch_dir: Path, checkpoint_batch_dir: Path, label: str
) -> bool:
    """Return True only for a fully packaged, computationally finished replicate.

    Scientific inconclusiveness may still be a completed deterministic result,
    but runtime/module exceptions, incomplete archives, or missing final completion
    receipts must remain resumable.
    """

    directory = batch_dir / label
    receipt_path = checkpoint_batch_dir / label / "completion_receipt.json"
    required_members = {
        "DiR_RAW_REPORT.zip": {
            "RESULTS_OVERVIEW.md",
            "training/training_metrics.csv",
            "measurements/core_measurements.json",
            "metadata/parameter_ownership.json",
            "metadata/sample_manifest.json",
            "metadata/measurement_module_manifest.json",
            "provenance/run_identity.json",
            "archive_manifest.json",
        },
        "DiR_SUMMARY_REPORT.zip": {
            "RESULTS_OVERVIEW.md",
            "summary.json",
            "manifest.json",
            "metadata/parameter_ownership.json",
            "provenance/run_identity.json",
            "archive_manifest.json",
        },
    }
    try:
        for name, members in required_members.items():
            path = directory / name
            if not path.is_file():
                return False
            with zipfile.ZipFile(path, "r") as archive:
                names = set(archive.namelist())
                if not members.issubset(names) or archive.testzip() is not None:
                    return False

        summary = _zip_json(directory / "DiR_SUMMARY_REPORT.zip", "summary.json")
        if not bool(summary.get("measurement_execution_complete", False)):
            return False
        if not str(summary.get("run_status", "")).startswith("completed"):
            return False

        if not receipt_path.is_file():
            return False
        receipt = read_json_file(receipt_path)
        if not isinstance(receipt, Mapping):
            return False
        if str(receipt.get("status", "")) not in {"completed", "completed_with_warnings"}:
            return False
        if dict(receipt.get("archive_statuses", {}) or {}) != {
            "raw_report": "completed",
            "summary_report": "completed",
        }:
            return False
        if str(receipt.get("artifact_run_id", "")) != str(summary.get("artifact_run_id", "")):
            return False
        return True
    except (OSError, zipfile.BadZipFile, KeyError, ValueError, json.JSONDecodeError):
        return False


def _zip_text(path: Path, member: str) -> str:
    with zipfile.ZipFile(path, "r") as archive:
        return archive.read(member).decode("utf-8")


def _zip_json(path: Path, member: str) -> dict[str, Any]:
    payload = json.loads(_zip_text(path, member))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path.name}:{member}")
    return payload


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _has_aggregatable_numeric_leaf(value: Any, path: str) -> bool:
    if _finite_number(value):
        return not any(
            fragment in path.lower() for fragment in _EXCLUDED_AGGREGATE_FRAGMENTS
        )
    if isinstance(value, Mapping):
        return any(
            _has_aggregatable_numeric_leaf(
                child, f"{path}.{key}" if path else str(key)
            )
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(
            _has_aggregatable_numeric_leaf(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    return False


def _collect_numeric_leaf_paths(
    value: Any, path: str, missing_in: str, records: list[dict[str, str]]
) -> None:
    """Record finite numeric leaves hidden by a missing/incompatible counterpart."""

    if _finite_number(value):
        if not any(fragment in path.lower() for fragment in _EXCLUDED_AGGREGATE_FRAGMENTS):
            records.append({
                "path": path,
                "missing_in": missing_in,
                "reason": "missing_nonfinite_or_incompatible_counterpart",
            })
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _collect_numeric_leaf_paths(child, child_path, missing_in, records)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _collect_numeric_leaf_paths(child, f"{path}[{index}]", missing_in, records)


def _collect_missing_paths(
    a: Any, b: Any, path: str, records: list[dict[str, str]]
) -> None:
    """Record A/B asymmetry whenever it hides an aggregatable numeric metric."""

    a_finite, b_finite = _finite_number(a), _finite_number(b)
    if a_finite or b_finite:
        if a_finite and b_finite:
            return
        if a_finite:
            _collect_numeric_leaf_paths(a, path, "B", records)
        if b_finite:
            _collect_numeric_leaf_paths(b, path, "A", records)
        return

    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys_a, keys_b = set(a), set(b)
        for key in sorted(keys_a - keys_b):
            child_path = f"{path}.{key}" if path else str(key)
            _collect_numeric_leaf_paths(a[key], child_path, "B", records)
        for key in sorted(keys_b - keys_a):
            child_path = f"{path}.{key}" if path else str(key)
            _collect_numeric_leaf_paths(b[key], child_path, "A", records)
        for key in sorted(keys_a.intersection(keys_b)):
            child_path = f"{path}.{key}" if path else str(key)
            _collect_missing_paths(a[key], b[key], child_path, records)
        return

    if isinstance(a, list) and isinstance(b, list):
        common = min(len(a), len(b))
        for index in range(common):
            _collect_missing_paths(a[index], b[index], f"{path}[{index}]", records)
        for index in range(common, len(a)):
            _collect_numeric_leaf_paths(a[index], f"{path}[{index}]", "B", records)
        for index in range(common, len(b)):
            _collect_numeric_leaf_paths(b[index], f"{path}[{index}]", "A", records)
        return

    # Container-vs-None/string/nonfinite mismatches can otherwise silently hide
    # every numeric metric beneath the container. Enumerate those leaves.
    _collect_numeric_leaf_paths(a, path, "B", records)
    _collect_numeric_leaf_paths(b, path, "A", records)


def _collect_statistics(a: Any, b: Any, path: str, records: list[dict[str, Any]]) -> None:
    if _finite_number(a) and _finite_number(b):
        if not any(fragment in path.lower() for fragment in _EXCLUDED_AGGREGATE_FRAGMENTS):
            a, b = float(a), float(b)
            records.append(
                {
                    "path": path,
                    "run_A": a,
                    "run_B": b,
                    "mean": (a + b) / 2.0,
                    "sample_std": abs(a - b) / math.sqrt(2.0),
                    "n": 2,
                }
            )
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a).intersection(b)):
            _collect_statistics(a[key], b[key], f"{path}.{key}" if path else str(key), records)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (child_a, child_b) in enumerate(zip(a, b)):
            _collect_statistics(child_a, child_b, f"{path}[{index}]", records)


def _paired_tree(a: Any, b: Any, path: str = "") -> Any:
    """Mirror paired summary structure with A/B/mean/sample-std leaves."""

    if _finite_number(a) and _finite_number(b):
        if any(fragment in path.lower() for fragment in _EXCLUDED_AGGREGATE_FRAGMENTS):
            return None
        a_value, b_value = float(a), float(b)
        return {
            "run_A": a_value,
            "run_B": b_value,
            "mean": (a_value + b_value) / 2.0,
            "sample_std": abs(a_value - b_value) / math.sqrt(2.0),
            "n": 2,
        }
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(set(a).intersection(b)):
            child_path = f"{path}.{key}" if path else str(key)
            child = _paired_tree(a[key], b[key], child_path)
            if child is not None and child != {} and child != []:
                output[str(key)] = child
        return output or None
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        output_list = []
        any_value = False
        for index, (child_a, child_b) in enumerate(zip(a, b)):
            child = _paired_tree(child_a, child_b, f"{path}[{index}]")
            output_list.append(child)
            any_value = any_value or child is not None
        return output_list if any_value else None
    return None


def _final_training(raw_zip: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in csv.DictReader(io.StringIO(_zip_text(raw_zip, "training/training_metrics.csv"))):
        run_id = str(row.get("run_id", "") or "").strip()
        if not run_id:
            continue
        epoch = int(float(str(row.get("epoch", "-1") or "-1")))
        if run_id not in latest or epoch >= int(latest[run_id]["epoch"]):
            latest[run_id] = {
                "epoch": epoch,
                "task_id": str(row.get("task_id", "") or ""),
                "eval_accuracy": float(row.get("eval_accuracy", "nan")),
                "eval_loss": float(row.get("eval_loss", "nan")),
            }
    return latest


def _training_statistics(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for run_id in sorted(set(a).intersection(b)):
        item: dict[str, Any] = {"task_id": a[run_id]["task_id"]}
        for metric in ("eval_accuracy", "eval_loss"):
            va, vb = float(a[run_id][metric]), float(b[run_id][metric])
            statistic = {
                "run_A": va,
                "run_B": vb,
                "mean": (va + vb) / 2.0,
                "sample_std": abs(va - vb) / math.sqrt(2.0),
                "n": 2,
            }
            item[metric] = statistic
            records.append({"path": f"training_final.{run_id}.{metric}", **statistic})
        output[run_id] = item
    return output, records


def _csv_text(records: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fields = ("path", "run_A", "run_B", "mean", "sample_std", "n")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row.get(key, "") for key in fields} for row in records)
    return buffer.getvalue()


def _overview(
    training: Mapping[str, Any],
    *,
    missing_metrics: list[dict[str, str]] | None = None,
    artifact_completion_statuses: Mapping[str, str] | None = None,
) -> str:
    lines = [
        "# DiR Results — Run A + Run B",
        "",
        "Cross-seed values are mean ± sample standard deviation (n=2).",
        "Per-run permutation p-values remain in A/B and are not averaged.",
        "",
        "| run | A accuracy | B accuracy | mean | sample std |",
        "|---|---:|---:|---:|---:|",
    ]
    for run_id, item in training.items():
        value = item["eval_accuracy"]
        lines.append(
            f"| {run_id} | {value['run_A']:.4f} | {value['run_B']:.4f} | "
            f"{value['mean']:.4f} | {value['sample_std']:.4f} |"
        )
    missing_metrics = list(missing_metrics or [])
    if missing_metrics:
        lines.extend(
            [
                "",
                f"WARNING: {len(missing_metrics)} A/B metric path(s) are missing, non-finite, or incompatible; see `AGGREGATE_SUMMARY.json` → `aggregation_completeness.missing_metrics`.",
            ]
        )
    artifact_completion_statuses = dict(artifact_completion_statuses or {})
    artifact_warnings = [
        f"Run {label}={status}"
        for label, status in sorted(artifact_completion_statuses.items())
        if status != "completed"
    ]
    if artifact_warnings:
        lines.extend(
            [
                "",
                "WARNING: final artifact completion is not warning-free: " + ", ".join(artifact_warnings) + ".",
            ]
        )
    lines.extend(["", "Full paired statistics are in `AGGREGATE_STATISTICS.csv` and mirrored under `aggregate/statistics/aggregate_statistics.csv`.", ""])
    return "\n".join(lines)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"


def _write_combined_zip(
    path: Path,
    *,
    source_archives: Mapping[str, Path],
    generated_members: Mapping[str, str],
) -> None:
    """Combine complete per-run archives under A/B with generated aggregate files."""

    temporary = path.with_name(path.name + ".temporary")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, text in generated_members.items():
            output.writestr(name, text)
        for prefix, source_path in source_archives.items():
            with zipfile.ZipFile(source_path, "r") as source:
                for info in source.infolist():
                    if info.is_dir():
                        continue
                    output.writestr(f"{prefix}/{info.filename}", source.read(info.filename))
    with zipfile.ZipFile(temporary, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt combined report archive member: {bad_member}")
    temporary.replace(path)


def write_aggregate_reports(batch_dir: Path, checkpoint_batch_dir: Path) -> None:
    """Write exactly two final ZIPs, each containing A/, B/, and aggregate/."""

    reports = {
        label: {
            "raw": batch_dir / label / "DiR_RAW_REPORT.zip",
            "summary": batch_dir / label / "DiR_SUMMARY_REPORT.zip",
        }
        for label in REPLICATE_LABELS
    }
    for label, pair in reports.items():
        for kind, path in pair.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing Run {label} {kind} report: {path}")

    summaries = {label: _zip_json(reports[label]["summary"], "summary.json") for label in REPLICATE_LABELS}
    completion_receipts = {
        label: read_json_file(checkpoint_batch_dir / label / "completion_receipt.json")
        for label in REPLICATE_LABELS
    }
    if not all(isinstance(receipt, Mapping) for receipt in completion_receipts.values()):
        raise ValueError("Completion receipts must be JSON objects")
    records: list[dict[str, Any]] = []
    paired_results: dict[str, Any] = {}
    missing_metrics: list[dict[str, str]] = []
    for section in _AGGREGATE_SECTIONS:
        in_a, in_b = section in summaries["A"], section in summaries["B"]
        if in_a and in_b:
            _collect_missing_paths(summaries["A"][section], summaries["B"][section], section, missing_metrics)
            _collect_statistics(summaries["A"][section], summaries["B"][section], section, records)
            paired = _paired_tree(summaries["A"][section], summaries["B"][section], section)
            if paired is not None:
                paired_results[section] = paired
        elif in_a and _has_aggregatable_numeric_leaf(summaries["A"][section], section):
            missing_metrics.append({"path": section, "missing_in": "B"})
        elif in_b and _has_aggregatable_numeric_leaf(summaries["B"][section], section):
            missing_metrics.append({"path": section, "missing_in": "A"})

    final_training = {
        label: _final_training(reports[label]["raw"])
        for label in REPLICATE_LABELS
    }
    training_run_ids_a = set(final_training["A"])
    training_run_ids_b = set(final_training["B"])
    for run_id in sorted(training_run_ids_a - training_run_ids_b):
        missing_metrics.append({"path": f"training_final.{run_id}", "missing_in": "B"})
    for run_id in sorted(training_run_ids_b - training_run_ids_a):
        missing_metrics.append({"path": f"training_final.{run_id}", "missing_in": "A"})
    training, training_records = _training_statistics(final_training["A"], final_training["B"])
    records = sorted([*training_records, *records], key=lambda row: str(row["path"]))
    missing_metrics = sorted(
        {
            (str(row["path"]), str(row["missing_in"]), str(row.get("reason", "missing"))): row
            for row in missing_metrics
        }.values(),
        key=lambda row: (row["path"], row["missing_in"], str(row.get("reason", ""))),
    )
    aggregation_complete = not missing_metrics
    artifact_completion_clean = all(
        str(completion_receipts[label].get("status", "")) == "completed"
        and dict(completion_receipts[label].get("archive_statuses", {}) or {})
        == {"raw_report": "completed", "summary_report": "completed"}
        and str(completion_receipts[label].get("artifact_run_id", ""))
        == str(summaries[label].get("artifact_run_id", ""))
        for label in REPLICATE_LABELS
    )
    replicates_overall_valid = all(
        bool(summaries[label].get("overall_valid")) for label in REPLICATE_LABELS
    )
    aggregate_overall_valid = bool(
        replicates_overall_valid and aggregation_complete and artifact_completion_clean
    )
    aggregate_run_status = "completed" if aggregate_overall_valid else "completed_with_warnings"
    summary = {
        "schema_version": "functional_correspondence_two_seed_aggregate_v4",
        "run_status": aggregate_run_status,
        "replicate_count": 2,
        "statistics_contract": {
            "mean": "arithmetic_mean",
            "sample_std": "ddof_1",
            "n": 2,
            "measurement_seeds": "shared",
            "run_B_training_seed_offset": RUN_B_TRAINING_SEED_OFFSET,
            "dictionary_basis_bank_seed": "shared_by_design",
            "per_run_inferential_statistics": "not_averaged_or_pooled",
        },
        "overall_valid": aggregate_overall_valid,
        "overall_valid_scope": (
            "both_replicates_scientifically_valid_plus_complete_A_B_numeric_aggregation_"
            "plus_warning_free_final_artifact_completion"
        ),
        "aggregation_completeness": {
            "complete": aggregation_complete,
            "missing_metric_count": len(missing_metrics),
            "missing_metrics": missing_metrics,
        },
        "replicates": {
            label: {
                "artifact_run_id": summaries[label].get("artifact_run_id"),
                "run_status": summaries[label].get("run_status"),
                "training_status": summaries[label].get("training_status"),
                "measurement_status": summaries[label].get("measurement_status"),
                "measurement_execution_complete": summaries[label].get("measurement_execution_complete"),
                "measurement_fully_valid": summaries[label].get("measurement_fully_valid"),
                "overall_valid": summaries[label].get("overall_valid"),
                "artifact_completion_status": completion_receipts[label].get("status"),
                "artifact_post_training_warning_count": completion_receipts[label].get("post_training_warning_count"),
                "archive_statuses": completion_receipts[label].get("archive_statuses"),
            }
            for label in REPLICATE_LABELS
        },
        "training_final": training,
        "paired_results": paired_results,
        "descriptive_statistics": records,
    }
    artifact_completion_statuses = {
        label: str(completion_receipts[label].get("status", "unknown"))
        for label in REPLICATE_LABELS
    }
    overview = _overview(
        training,
        missing_metrics=missing_metrics,
        artifact_completion_statuses=artifact_completion_statuses,
    )
    statistics_csv = _csv_text(records)

    aggregate_common = {
        "aggregate/RESULTS_OVERVIEW.md": overview,
        "aggregate/summary.json": _json_text(summary),
        "aggregate/statistics/aggregate_statistics.csv": statistics_csv,
    }

    # Summary is intentionally aggregate-first for quick paper-result inspection.
    completion_members = {
        f"{label}/completion_receipt.json": _json_text(completion_receipts[label])
        for label in REPLICATE_LABELS
    }
    summary_generated = {
        "RESULTS_OVERVIEW.md": overview,
        "AGGREGATE_SUMMARY.json": _json_text(summary),
        "AGGREGATE_STATISTICS.csv": statistics_csv,
        **aggregate_common,
        **completion_members,
    }
    _write_combined_zip(
        batch_dir / "DiR_SUMMARY_REPORT.zip",
        source_archives={
            "A": reports["A"]["summary"],
            "B": reports["B"]["summary"],
        },
        generated_members=summary_generated,
    )

    raw_generated = {
        "RESULTS_OVERVIEW.md": overview,
        "AGGREGATE_SUMMARY.json": _json_text(summary),
        "AGGREGATE_STATISTICS.csv": statistics_csv,
        **aggregate_common,
        **completion_members,
        "aggregate/replicates/run_A_summary.json": _json_text(summaries["A"]),
        "aggregate/replicates/run_B_summary.json": _json_text(summaries["B"]),
        "aggregate/replicates/run_A_training_metrics.csv": _zip_text(
            reports["A"]["raw"], "training/training_metrics.csv"
        ),
        "aggregate/replicates/run_B_training_metrics.csv": _zip_text(
            reports["B"]["raw"], "training/training_metrics.csv"
        ),
    }
    _write_combined_zip(
        batch_dir / "DiR_RAW_REPORT.zip",
        source_archives={
            "A": reports["A"]["raw"],
            "B": reports["B"]["raw"],
        },
        generated_members=raw_generated,
    )

    # Per-run ZIPs are staging artifacts only. Successful finalization leaves two files.
    for name in (*REPLICATE_LABELS, "aggregate"):
        path = batch_dir / name
        if path.exists():
            shutil.rmtree(path)

