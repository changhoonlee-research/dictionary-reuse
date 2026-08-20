"""Experiment contracts, artifact constants, and shared I/O helpers."""

from __future__ import annotations

from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "dir_experiment_v5"

IMPLEMENTATION_REVISION = "dir_v1_2_4_two_seed_final_validity_fixed"

PROGRESS_FILE = "progress.json"

TRAINING_FILE = "training_metrics.csv"

CHECKPOINT_TRAINING_FILE = "training_metrics.csv"

CHECKPOINT_PROVENANCE_FILE = "checkpoint_provenance.json"

ARTIFACT_RUN_FILE = "run_identity.json"

ARTIFACT_IDENTITY_SCHEMA_VERSION = "dir_artifact_identity_v2"

ARTIFACT_COMPLETION_FILE = "completion_receipt.json"

ZIP_MANIFEST_FILE = "archive_manifest.json"

CHECKPOINT_SCHEMA_VERSION = "dir_model_only_checkpoint_v5"

EXPECTED_CHECKPOINT_COUNT = 12

OWNERSHIP_FILE = "parameter_ownership.json"

SAMPLE_MANIFEST_FILE = "sample_manifest.json"

CORE_MEASUREMENTS_FILE = "core_measurements.json"

MODULE_MANIFEST_FILE = "measurement_module_manifest.json"

MEASUREMENT_SHARD_DIRECTORY = "measurement_shards"

SUMMARY_FILE = "summary.json"

RESULTS_OVERVIEW_FILE = "RESULTS_OVERVIEW.md"

MANIFEST_FILE = "manifest.json"


def _write_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    temporary.replace(path)

def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)

def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _safe_module_file_name(name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    readable = "".join(character if character.isalnum() else "_" for character in name)
    return f"{readable[:120]}__{digest}.json.gz"

def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

def _completed_status(status: str) -> bool:
    return str(status) in {"completed", "reused_completed"}

def _reusable_shard_status(status: str) -> bool:
    """Return True when the expensive module computation finished deterministically.

    A primary view may be scientifically inconclusive while the underlying
    capture/JVP/Jacobian work is complete. Re-running that same shard cannot
    recover signal and only wastes the long experiment, so these view-validity
    outcomes remain reusable. Runtime exceptions and non-finite failures are
    deliberately excluded and are retried.
    """

    return str(status) in {
        "completed",
        "reused_completed",
        "partial_primary_views",
        "inconclusive_no_valid_primary_cka",
        "inconclusive_no_primary_view_common_subset",
        "inconclusive_no_valid_primary_jacobian_view",
        "inconclusive_no_valid_internal_vjp",
        "partial_comparable_blocks",
        "inconclusive_no_comparable_atoms",
        "inconclusive_no_transformer_dictionary_layers",
    }

def _reportable_status(status: str) -> bool:
    """Return True when any completed primary-view result may be summarized."""

    return str(status) in {
        "completed",
        "reused_completed",
        "partial_primary_views",
        "inconclusive_no_valid_primary_cka",
        "inconclusive_no_valid_primary_jacobian_view",
        "inconclusive_no_valid_internal_vjp",
        "partial_comparable_blocks",
        "inconclusive_no_comparable_atoms",
        "inconclusive_no_transformer_dictionary_layers",
    }

def _superseded_status(status: str) -> bool:
    return str(status).startswith("superseded_")

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_name(path.name + ".temporary")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _best_effort_unlink(path: Path) -> None:
    """Remove a post-training artifact without allowing cleanup failure to mask results."""

    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_artifact_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"dir-{timestamp}-{uuid.uuid4().hex[:12]}"

def _invalidate_configured_final_artifacts(config: Mapping[str, Any]) -> list[str]:
    """Remove replaceable report ZIPs inside the selected run directory."""

    paths = dict(config.get("paths", {}) or {})
    removed: list[str] = []
    for key in ("raw_report_zip", "summary_report_zip"):
        value = str(paths.get(key, "") or "").strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        for candidate in (path, path.with_name(path.name + ".temporary")):
            if candidate.is_file() or candidate.is_symlink():
                candidate.unlink(missing_ok=True)
                removed.append(str(candidate))
    return sorted(set(removed))

def _clear_previous_report_files(output_dir: Path) -> None:
    """Clear only replaceable report files; preserve checkpoints and shards."""

    for name in (
        PROGRESS_FILE,
        TRAINING_FILE,
        OWNERSHIP_FILE,
        SAMPLE_MANIFEST_FILE,
        CORE_MEASUREMENTS_FILE,
        MODULE_MANIFEST_FILE,
        SUMMARY_FILE,
        RESULTS_OVERVIEW_FILE,
        MANIFEST_FILE,
        ARTIFACT_RUN_FILE,
        ARTIFACT_COMPLETION_FILE,
        "experiment_error.json",
    ):
        (output_dir / name).unlink(missing_ok=True)

def _measurement_contract_sha256(raw_config: Mapping[str, Any]) -> str:
    """Hash only configuration capable of changing raw measurement shards."""

    plan = _functional_correspondence_config(raw_config)
    samples = dict(plan.get("samples", {}) or {})
    raw_sample_keys = (
        "representation",
        "probe_train",
        "probe_validation",
        "probe_test",
        "response",
        "direct_wide_windows",
        "attention_spectral",
        "gradient",
        "jacobian",
    )
    raw_plan = {
        "condition_order": list(plan.get("condition_order", [])),
        "require_cuda": bool(plan.get("require_cuda", True)),
        "samples": {key: samples.get(key) for key in raw_sample_keys},
        "patching": deepcopy(plan.get("patching", {})),
        "jacobian": deepcopy(plan.get("jacobian", {})),
        "measurement_quality": deepcopy(plan.get("measurement_quality", {})),
        "atom_group_ablation": deepcopy(plan.get("atom_group_ablation", {})),
    }
    role = deepcopy(_dictionary_reuse_config(raw_config))
    for key in (
        "description",
        "latest_summary",
        "metadata",
    ):
        role.pop(key, None)
    runtime_contract = role.get("runtime")
    if isinstance(runtime_contract, Mapping):
        role["runtime"] = {
            key: deepcopy(value)
            for key, value in runtime_contract.items()
            if key not in _MEASUREMENT_CONTRACT_IGNORED_RUNTIME_KEYS
        }
    return _canonical_json_sha256(
        {
            "schema_version": "dir_raw_measurement_contract_v3",
            "plan": raw_plan,
            "role": role,
        }
    )

def _statistics_contract_sha256(raw_config: Mapping[str, Any]) -> str:
    """Hash only configuration capable of changing derived statistics."""

    plan = _functional_correspondence_config(raw_config)
    samples = dict(plan.get("samples", {}) or {})
    return _canonical_json_sha256(
        {
            "schema_version": "dir_statistics_contract_v1",
            "statistics_seed": int(plan.get("statistics_seed", 0)),
            "bootstrap_iterations": int(samples.get("bootstrap_iterations", 0)),
            "global_permutations": int(samples.get("global_permutations", 0)),
            "depth_band_permutations": int(
                samples.get("depth_band_permutations", 0)
            ),
            "primary_metrics": list(plan.get("primary_metrics", [])),
        }
    )

_TRAINING_CONTRACT_IGNORED_RUNTIME_KEYS = {
    "device",
    "torch_num_threads",
    "console_logging_enabled",
    "default_preset",
    "fail_fast_if_cuda_unavailable",
}

_MEASUREMENT_CONTRACT_IGNORED_RUNTIME_KEYS = _TRAINING_CONTRACT_IGNORED_RUNTIME_KEYS | {
    "cache_dataset_object",
    "cache_task_subsets",
    "num_workers",
}

def _training_contract_sha256(raw_config: Mapping[str, Any]) -> str:
    """Hash only configuration capable of changing trained DiR endpoints.

    Measurement sample counts, Jacobian settings, report paths, and output
    packaging are deliberately excluded so a measurement-only change does not
    force 80/52-epoch retraining.
    """

    role = _dictionary_reuse_config(raw_config)
    plan = _functional_correspondence_config(raw_config)
    excluded_role_keys = {
        "description",
        "latest_summary",
        "metadata",
    }
    role_training_contract = {
        key: deepcopy(value)
        for key, value in role.items()
        if key not in excluded_role_keys
    }
    runtime_contract = role_training_contract.get("runtime")
    if isinstance(runtime_contract, Mapping):
        role_training_contract["runtime"] = {
            key: deepcopy(value)
            for key, value in runtime_contract.items()
            if key not in _TRAINING_CONTRACT_IGNORED_RUNTIME_KEYS
        }
    return _canonical_json_sha256(
        {
            "schema_version": "dir_training_contract_v1",
            "dir_training": plan["training"],
            "role_training_contract": role_training_contract,
        }
    )

def _canonical_module_name(
    module_name: str,
    module_results: Mapping[str, Any],
    module_status: Mapping[str, Any],
) -> str | None:
    """Return the fixed-release result when it is reportable.

    The DiR release uses one immutable 40-probe/rank-32 Jacobian contract with
    holdout validity gating. Measurement quality is recorded in the result and never changes the probe count or the
    canonical module name after execution.
    """

    name = str(module_name)
    status = str(module_status.get(name, {}).get("status", "missing"))
    if _reportable_status(status) and name in module_results:
        return name
    return None

def _functional_correspondence_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    value = raw_config.get("functional_correspondence", {})
    if not isinstance(value, Mapping):
        raise ValueError("functional_correspondence must be an object")
    return dict(value)

def _dictionary_reuse_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    value = raw_config.get("dictionary_reuse", {})
    if not isinstance(value, Mapping):
        raise ValueError("dictionary_reuse must be an object")
    return dict(value)
