"""Final report payloads, human-readable overview, and fail-soft artifact packaging."""

from __future__ import annotations


# Human-readable results overview
from pathlib import Path
import math
import shutil
from typing import Any, Callable, Mapping, Sequence

from .common import _read_csv_rows, _reusable_shard_status, _superseded_status
from .matrix import CONDITION_ORDER, PAIR_BY_CONDITION

def _format_overview_number(value: Any, *, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _final_training_accuracy_rows(training_csv_path: Path) -> list[dict[str, str]]:
    rows = _read_csv_rows(training_csv_path)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", "") or "").strip()
        if not run_id:
            continue
        try:
            epoch = int(float(str(row.get("epoch", "0") or "0")))
        except (TypeError, ValueError):
            epoch = -1
        previous = latest.get(run_id)
        if previous is None or epoch >= int(previous["epoch"]):
            latest[run_id] = {"epoch": epoch, "row": row}
    output: list[dict[str, str]] = []
    run_order = (
        "dir_source_a",
        "dir_same_task",
        "dir_dictionary_fixed",
        "dir_dictionary_trainable",
        "dense_source_a",
        "dense_same_task",
        "dense_different_task",
    )
    ordered_run_ids = [run_id for run_id in run_order if run_id in latest]
    ordered_run_ids.extend(sorted(set(latest) - set(run_order)))
    for run_id in ordered_run_ids:
        item = latest[run_id]
        row = item["row"]
        output.append(
            {
                "run_id": run_id,
                "epoch": str(item["epoch"]),
                "task_id": str(row.get("task_id", "") or ""),
                "eval_accuracy": _format_overview_number(row.get("eval_accuracy")),
                "eval_loss": _format_overview_number(row.get("eval_loss")),
            }
        )
    return output


def _human_training_run_name(run_id: str) -> str:
    return {
        "dir_source_a": "DiR Source · Task 1",
        "dir_same_task": "DiR Same-task · Task 1",
        "dir_dictionary_fixed": "DiR Dictionary-Fixed · Task 2",
        "dir_dictionary_trainable": "DiR Dictionary-Trainable · Task 2",
        "dense_source_a": "Dense Source · Task 1",
        "dense_same_task": "Dense Same-task · Task 1",
        "dense_different_task": "Dense Full-Transfer · Task 2",
    }.get(run_id, run_id)


def _human_condition_name(condition: str) -> str:
    return {
        "dir_same_task": "DiR · Same-task",
        "dir_dictionary_fixed": "DiR · Dictionary-Fixed",
        "dir_dictionary_trainable": "DiR · Dictionary-Trainable",
        "dense_same_task": "Dense · Same-task",
        "dense_different_task": "Dense · Full-Transfer",
    }.get(condition, condition)


def _human_task_name(task: str) -> str:
    return {"task1": "Task 1", "task2": "Task 2"}.get(task, task)


def _human_metric_name(metric_name: str) -> str:
    labels = {
        "direct_function.single_bidirectional_mean_cls_debiased_cka_12x12": "Direct · CLS",
        "direct_function.single_bidirectional_mean_patch_debiased_cka_12x12": "Direct · Patch",
        "ablation.block_update.post_layernorm_cls_delta_debiased_cka_12x12": "Ablation · CLS",
        "ablation.block_update.post_layernorm_patch_delta_debiased_cka_12x12": "Ablation · Patch",
        "patching.block_update.common_valid_post_layernorm_cls_recovery_debiased_cka_12x12": "Patching · CLS (mask/blur/noise)",
        "patching.block_update.common_valid_post_layernorm_patch_recovery_debiased_cka_12x12": "Patching · Patch (mask/blur/noise)",
        "jacobian.input_jvp.input_to_block_update_cls_debiased_cka_12x12": "Input JVP · CLS",
        "jacobian.input_jvp.input_to_block_update_patch_debiased_cka_12x12": "Input JVP · Patch",
    }
    return labels.get(metric_name, metric_name)


def _format_primary_record_field(record: Mapping[str, Any], field: str) -> str:
    members = record.get("family_members")
    if isinstance(members, Mapping):
        labels = {"mask": "M", "blur": "B", "noise": "N"}
        parts: list[str] = []
        for variant in record.get("variant_order", ("mask", "blur", "noise")):
            member = members.get(str(variant), {})
            value = _format_overview_number(
                member.get(field) if isinstance(member, Mapping) else None
            )
            parts.append(f"{labels.get(str(variant), str(variant))}:{value}")
        return " / ".join(parts)
    return _format_overview_number(record.get(field))


def _format_primary_record_status(record: Mapping[str, Any]) -> str:
    status = str(
        record.get(
            "statistics_status",
            record.get("status", record.get("module_status", "unknown")),
        )
    )
    total = record.get("family_total_variants")
    if total is None:
        return status
    usable = int(record.get("family_usable_variants", 0))
    return f"{status} ({usable}/{int(total)} variants usable)"


def _primary_overview_rows(primary_results: Mapping[str, Any]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for condition in CONDITION_ORDER:
        condition_payload = primary_results.get(condition, {})
        if not isinstance(condition_payload, Mapping):
            continue
        for task in (PAIR_BY_CONDITION[condition].task_key,):
            task_payload = condition_payload.get(task, {})
            if not isinstance(task_payload, Mapping):
                continue
            for metric_name, record in task_payload.items():
                if not isinstance(record, Mapping):
                    continue
                output.append(
                    {
                        "condition": condition,
                        "condition_label": _human_condition_name(condition),
                        "task": task,
                        "task_label": _human_task_name(task),
                        "metric": str(metric_name),
                        "metric_label": _human_metric_name(str(metric_name)),
                        "status": _format_primary_record_status(record),
                        "diagonal_mean": _format_primary_record_field(record, "diagonal_mean"),
                        "depth_band_margin": _format_primary_record_field(
                            record, "symmetric_depth_band_matched_same_index_margin"
                        ),
                        "same_index_rank_mean": _format_primary_record_field(
                            record, "symmetric_same_index_rank_mean"
                        ),
                        "rank1_fraction": _format_primary_record_field(record, "rank1_fraction"),
                    }
                )
    return output


def _write_results_overview(
    path: Path,
    *,
    summary: Mapping[str, Any],
    training_csv_path: Path,
) -> None:
    """Write a concise human-readable index without changing machine-readable results."""

    environment = summary.get("runtime_environment", {})
    if not isinstance(environment, Mapping):
        environment = {}

    lines = [
        "# DiR Results",
        "",
        "## Status",
        "",
        "| item | status |",
        "|---|---|",
        f"| Run | {summary.get('run_status', 'unknown')} |",
        f"| Training | {summary.get('training_status', 'unknown')} |",
        f"| Measurements | {summary.get('measurement_status', 'unknown')} |",
        f"| Core measurements | {summary.get('core_measurement_status', 'unknown')} |",
        f"| Supplementary measurements | {summary.get('supplementary_measurement_status', 'unknown')} |",
        f"| Post-training warnings | {summary.get('post_training_warning_count', 0)} |",
        "",
        "## Environment",
        "",
        "| item | value |",
        "|---|---|",
        f"| GPU | {environment.get('gpu_device_name', 'unknown')} |",
        f"| GPU compute capability | {environment.get('gpu_compute_capability', 'unknown')} |",
        f"| GPU memory (bytes) | {environment.get('gpu_total_memory_bytes', 'unknown')} |",
        f"| Python | {environment.get('python_version', 'unknown')} |",
        f"| PyTorch | {environment.get('torch_version', 'unknown')} |",
        f"| torchvision | {environment.get('torchvision_version', 'unknown')} |",
        f"| NumPy | {environment.get('numpy_version', 'unknown')} |",
        f"| CUDA runtime | {environment.get('cuda_runtime_version', 'unknown')} |",
        f"| NVIDIA driver | {environment.get('nvidia_driver_version', 'unknown')} |",
        f"| cuDNN | {environment.get('cudnn_version', 'unknown')} |",
        f"| Platform | {environment.get('platform', 'unknown')} |",
        f"| Execution environment | {environment.get('execution_environment', 'unknown')} |",
    ]
    colab_release = str(environment.get("colab_release_tag", "unknown"))
    colab_backend = str(environment.get("colab_backend_version", "unknown"))
    if colab_release not in {"", "unknown", "None"}:
        lines.append(f"| Colab release | {colab_release} |")
    if colab_backend not in {"", "unknown", "None"}:
        lines.append(f"| Colab backend | {colab_backend} |")

    lines.extend(
        [
            "",
            "## Training",
            "",
            "| run | task | epoch | accuracy | loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    training_rows = _final_training_accuracy_rows(training_csv_path)
    if training_rows:
        for row in training_rows:
            lines.append(
                f"| {_human_training_run_name(row['run_id'])} | {_human_task_name(row['task_id'])} | "
                f"{row['epoch']} | {row['eval_accuracy']} | {row['eval_loss']} |"
            )
    else:
        lines.append("| — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Primary measurements",
            "",
            "`Diagonal` is matched-block similarity. Positive `depth margin`, lower `same-index rank`, "
            "higher `rank-1` favor stronger same-index correspondence. Same-task DiR-vs-Dense and different-task Dictionary-Fixed-vs-Dictionary-Trainable/Dense comparisons are reported separately in `summary.json`.",
        ]
    )
    primary_rows = _primary_overview_rows(summary.get("primary_results", {}))
    if primary_rows:
        for condition in CONDITION_ORDER:
            condition_rows = [row for row in primary_rows if row["condition"] == condition]
            if not condition_rows:
                continue
            lines.extend(["", f"### {_human_condition_name(condition)}"] )
            for task in (PAIR_BY_CONDITION[condition].task_key,):
                task_rows = [row for row in condition_rows if row["task"] == task]
                if not task_rows:
                    continue
                lines.extend(
                    [
                        "",
                        f"**{_human_task_name(task)}**",
                        "",
                        "| measurement | status | diagonal | depth margin | same-index rank | rank-1 |",
                        "|---|---|---:|---:|---:|---:|",
                    ]
                )
                for row in task_rows:
                    metric = row["metric_label"].replace("|", "\\|")
                    lines.append(
                        f"| {metric} | {row['status']} | {row['diagonal_mean']} | "
                        f"{row['depth_band_margin']} | {row['same_index_rank_mean']} | "
                        f"{row['rank1_fraction']} |"
                    )
    else:
        lines.extend(["", "Primary measurements unavailable."])

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.json`: complete statistics and matrices.",
            "- `DiR_RAW_REPORT.zip`: raw measurement shards, manifests, training metrics, and provenance.",
            "",
        ]
    )
    temporary = path.with_name(path.name + ".temporary")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


# Report payload builders
from .common import ARTIFACT_RUN_FILE, MANIFEST_FILE, _canonical_json_sha256, _utc_now_iso
from .runtime import _archive_member_name
from .training_stage import TrainingStageResult
from .measurement_stage import MeasurementStageResult


def build_summary_payload(
    *,
    plan: Mapping[str, Any],
    training: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    dataset_sample_reference: Mapping[str, Any],
    post_training_warnings: Sequence[Mapping[str, Any]],
    training_stage: TrainingStageResult,
    measurement_stage: MeasurementStageResult,
    artifact_run_id: str,
    measurement_contract_sha256: str,
    statistics_contract_sha256: str,
) -> dict[str, Any]:
    checkpoint_inventory_status = training_stage.checkpoint_inventory_status
    checkpoint_provenance = training_stage.checkpoint_provenance
    checkpoint_provenance_rebuilt = training_stage.checkpoint_provenance_rebuilt
    checkpoint_provenance_sha256 = training_stage.checkpoint_provenance_sha256
    ownership = training_stage.ownership
    resume_from_checkpoints = training_stage.resume_from_checkpoints
    support_commit_output_parity = training_stage.support_commit_output_parity

    completed = measurement_stage.completed
    warning_count = measurement_stage.warning_count
    inconclusive_count = measurement_stage.inconclusive_count
    superseded_count = measurement_stage.superseded_count
    core_names = measurement_stage.core_names
    supplementary_names = measurement_stage.supplementary_names
    core_measurement_status = measurement_stage.core_measurement_status
    supplementary_measurement_status = measurement_stage.supplementary_measurement_status
    measurement_status = measurement_stage.measurement_status
    training_valid = measurement_stage.training_valid
    condition_comparisons = measurement_stage.condition_comparisons
    primary_results = measurement_stage.primary_results
    supporting_results = measurement_stage.supporting_results
    jacobian_rank32_results = measurement_stage.jacobian_rank32_results
    jacobian_stability_contract = measurement_stage.jacobian_stability_contract
    module_status = measurement_stage.module_status
    run_fingerprint = measurement_stage.run_fingerprint
    statistics_fingerprint = measurement_stage.statistics_fingerprint
    stale_measurement_shards_removed = measurement_stage.stale_measurement_shards_removed
    current_measurement_shard_count = measurement_stage.current_measurement_shard_count

    module_execution_complete = all(
        _superseded_status(str(payload.get("status", "")))
        or _reusable_shard_status(str(payload.get("status", "")))
        for payload in module_status.values()
    ) and bool(module_status)
    measurement_fully_valid = bool(
        module_execution_complete
        and measurement_status == "complete"
        and not post_training_warnings
    )
    training_status = (
        "valid"
        if training_valid and support_commit_output_parity["passed"]
        else "warning"
    )
    overall_valid = bool(training_status == "valid" and measurement_fully_valid)
    if overall_valid:
        run_status = "completed"
    elif (
        training_status == "valid"
        and module_execution_complete
        and measurement_status in {"partial", "inconclusive"}
        and not post_training_warnings
    ):
        run_status = "completed_with_measurement_limitations"
    else:
        run_status = "completed_with_warnings"

    return {
            "schema_version": "functional_correspondence_summary_v13",
            "scientific_execution": True,
            "run_status": run_status,
            "runtime_environment": dict(runtime_environment),
            "training_status": training_status,
            "measurement_status": measurement_status,
            "core_measurement_status": core_measurement_status,
            "supplementary_measurement_status": supplementary_measurement_status,
            "measurement_execution_complete": bool(module_execution_complete),
            "measurement_fully_valid": bool(measurement_fully_valid),
            "overall_valid": overall_valid,
            "overall_valid_scope": "training_validity_plus_support_commit_output_parity_plus_complete_warning_free_measurements",
            "support_commit_output_parity": support_commit_output_parity,
            "module_count": len(module_status),
            "core_module_count": len(core_names),
            "supplementary_module_count": len(supplementary_names),
            "completed_module_count": completed,
            "warning_module_count": warning_count,
            "inconclusive_module_count": inconclusive_count,
            "superseded_module_count": superseded_count,
            "jacobian_stability_contract": jacobian_stability_contract,
            "jacobian_probe_fallback_enabled": False,
            "jacobian_split_half_role": "advisory_only",
            "module_status": module_status,
            "parameter_ownership": ownership,
            "measurement_state_guard": "per_module_runtime_signature_with_checkpoint_reload_on_detected_mutation",
            "checkpoint_resume_used": bool(resume_from_checkpoints),
            "checkpoint_provenance_rebuilt": bool(checkpoint_provenance_rebuilt),
            "checkpoint_provenance_sha256": checkpoint_provenance_sha256,
            "checkpoint_training_run_id": checkpoint_provenance.get("training_run_id"),
            "checkpoint_provenance_origin": checkpoint_provenance.get("provenance_origin"),
            "artifact_run_id": artifact_run_id,
            "measurement_device_residency_policy": "only_current_module_pair_on_cuda_all_other_endpoints_on_cpu",
            "condition_initialization": {
                "dir_same_task": {
                    "initialization_source": "fresh_DiR_Target_with_Source-active_D_and_D-owned_scales_copied_and_fixed;_C_route_support_fresh",
                    "model_seed": int(training["dir_same_task_seed"]),
                    "data_order_seed": int(training["dir_same_task_data_order_seed"]),
                    "source_active_D_and_D_owned_scales_fixed": True,
                    "inactive_D_phase_allowed_slices_trainable": True,
                    "inactive_D_training_scope": "phase_allowed_internal_facing_block_D_plus_included_head_D",
                    "transferred_coefficients": False,
                },
                "dir_dictionary_fixed": {
                    "initialization_source": "Source_DiR_full_backbone_exact_copy_then_fresh_Target_head;_Source_dynamic_route_history_cleared_and_fixed_support_reopened_before_Target_training",
                    "head_reset_seed": int(training["dir_different_task_head_seed"]),
                    "data_order_seed": int(training["different_task_data_order_seed"]),
                    "source_active_D_and_D_owned_scales_fixed": True,
                    "inactive_D_phase_allowed_slices_trainable": True,
                    "inactive_D_training_scope": "phase_allowed_internal_facing_block_D_plus_included_head_D",
                    "transferred_coefficients": True,
                },
                "dir_dictionary_trainable": {
                    "initialization_source": "same_exact_Source_DiR_full_backbone_and_same_fresh_Target_head_as_Dictionary-Fixed",
                    "head_reset_seed": int(training["dir_different_task_head_seed"]),
                    "data_order_seed": int(training["different_task_data_order_seed"]),
                    "source_active_D_and_D_owned_scales_fixed": False,
                    "transferred_coefficients": True,
                },
                "dense_same_task": {
                    "initialization_source": "independent_Dense_scratch_Target",
                    "model_seed": int(training["dense_same_task_seed"]),
                    "data_order_seed": int(training["dense_same_task_data_order_seed"]),
                },
                "dense_different_task": {
                    "initialization_source": "Dense_Source_endpoint_full_weight_copy_then_fresh_Target_head",
                    "head_reset_seed": int(training["dense_different_task_head_seed"]),
                    "data_order_seed": int(training["different_task_data_order_seed"]),
                },
            },
            "initialization_audit": training_stage.initialization_audit,
            "measurement_shard_resume_enabled": True,
            "stale_measurement_shards_removed": stale_measurement_shards_removed,
            "current_measurement_shard_count": current_measurement_shard_count,
            "measurement_run_fingerprint": run_fingerprint,
            "measurement_contract_sha256": measurement_contract_sha256,
            "statistics_contract_sha256": statistics_contract_sha256,
            "statistics_fingerprint": statistics_fingerprint,
            "post_training_failure_policy": "warnings_preserve_individual_checkpoints_and_continue_all_remaining_measurements_when_scientifically_interpretable",
            "post_training_warning_count": len(post_training_warnings),
            "post_training_warnings": post_training_warnings,
            "checkpoint_inventory_status": checkpoint_inventory_status,
            "interpretation_boundary": (
                "Same-task tests correspondence from dictionary reuse across seeds against "
                "Dense scratch. Different-task Dictionary-Fixed-vs-Dictionary-Trainable "
                "isolates Source-active dictionary anchoring from an identical transferred "
                "DiR state; Source-inactive D remains Target-trainable only on phase-allowed "
                "dictionary coordinates (internal-facing block D plus included head D) in "
                "Dictionary-Fixed. "
                "Dense Full-Transfer provides the general weight-transfer baseline. "
                "Low-signal blocks remain inconclusive rather than aligned."
            ),
            "primary_metric_contract": list(plan["primary_metrics"]),
            "condition_comparisons": condition_comparisons,
            "primary_results": primary_results,
            "supporting_results": supporting_results,
            "jacobian_rank32_results": jacobian_rank32_results,
            "dataset_sample_reference_sha256": str(dataset_sample_reference.get("sample_manifest_sha256", "")),
            "supporting_measurement_contract": {
                "native_block_update": "reported_in_raw_core_with_full_matrix_validity_summary_and_permutation",
                "internal_vjp": "projected_VJP_probe_sketch_reported_in_raw_core_with_pathwise_matrix_validity_summary_and_permutation_not_explicit_internal_Jacobian_SVD",
                "post_o_and_post_w2": "supporting_causal_sites_not_primary",
            },
            "augmentation": "none",
        }


def build_manifest_payload(
    *,
    summary: Mapping[str, Any],
    resume_from_checkpoints: bool,
    checkpoint_provenance: Mapping[str, Any],
    artifact_run_id: str,
    artifact_identity_sha256: str,
    effective_config_path: Path,
    effective_config_available: bool,
    effective_config_sha256: str | None,
    artifact_identity_prepackaging_status: str,
    missing_raw_files: Sequence[str],
    missing_summary_files: Sequence[str],
    run_fingerprint: str,
    measurement_contract_sha256: str,
    statistics_contract_sha256: str,
    statistics_fingerprint: str,
    raw_files: Sequence[Path],
    summary_files: Sequence[Path],
    output_dir: Path,
    raw_zip_manifest: Mapping[str, Any] | None,
    raw_zip_status: str,
    checkpoint_inventory_status: str,
    post_training_warnings: Sequence[Mapping[str, Any]],
    raw_zip: Path,
    checkpoint_dir: Path,
    checkpoint_files: Sequence[Path],
    summary_zip: Path,
) -> dict[str, Any]:
    return {
            "schema_version": "dir_manifest_v4",
            "status": summary["run_status"],
            "training_status": summary["training_status"],
            "measurement_status": summary["measurement_status"],
            "measurement_execution_complete": summary["measurement_execution_complete"],
            "measurement_fully_valid": summary["measurement_fully_valid"],
            "overall_valid": summary["overall_valid"],
            "core_measurement_status": summary["core_measurement_status"],
            "supplementary_measurement_status": summary["supplementary_measurement_status"],
            "warning_module_count": summary["warning_module_count"],
            "inconclusive_module_count": summary["inconclusive_module_count"],
            "checkpoint_resume_used": bool(resume_from_checkpoints),
            "checkpoint_training_run_id": checkpoint_provenance.get("training_run_id"),
            "checkpoint_provenance_origin": checkpoint_provenance.get("provenance_origin"),
            "artifact_run_id": artifact_run_id,
            "artifact_identity_file": ARTIFACT_RUN_FILE,
            "artifact_identity_sha256": artifact_identity_sha256,
            "effective_run_config_file": effective_config_path.name if effective_config_available else None,
            "effective_run_config_sha256": effective_config_sha256,
            "effective_run_config_status": "completed" if effective_config_available else "warning_unavailable",
            "artifact_identity_prepackaging_status": artifact_identity_prepackaging_status,
            "missing_expected_raw_files": missing_raw_files,
            "missing_expected_summary_files": missing_summary_files,
            "measurement_run_fingerprint": run_fingerprint,
            "measurement_contract_sha256": measurement_contract_sha256,
            "statistics_contract_sha256": statistics_contract_sha256,
            "statistics_fingerprint": statistics_fingerprint,
            "report_files": sorted(
                {
                    *[_archive_member_name(path, output_dir) for path in raw_files + summary_files],
                    MANIFEST_FILE,
                }
            ),
            "report_file_path_contract": (
                "public_archive_paths_only; runtime .work paths and hashed shard filenames are not exposed"
            ),
            "raw_zip_manifest": raw_zip_manifest,
            "raw_report_zip_status": raw_zip_status,
            "checkpoint_inventory_status": checkpoint_inventory_status,
            "post_training_warning_count": len(post_training_warnings),
            "post_training_warnings": post_training_warnings,
            "raw_report_zip": str(raw_zip),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_count": sum(path.is_file() for path in checkpoint_files),
            "summary_report_zip": str(summary_zip),
            "summary_archive_contract": {
                "contains_final_manifest": True,
                "final_manifest_path": MANIFEST_FILE,
                "verification": "atomic_publish_after_exact_member_set_and_CRC_check",
            },
            "artifact_identity_contract": "same_immutable_identity_file_in_raw_and_summary_archives_with_checkpoint_provenance_preserved_separately",
        }


def build_completion_payload(
    *,
    completion_status: str,
    artifact_run_id: str,
    artifact_identity_sha256: str,
    post_training_warnings: Sequence[Mapping[str, Any]],
    final_warning_module_names: Sequence[str],
    final_archive_statuses: Mapping[str, str],
    manifest_path: Path,
    scientific_manifest_sha256: str | None,
    checkpoint_inventory_status: str,
    checkpoint_dir: Path,
    checkpoint_files: Sequence[Path],
    checkpoint_training_csv_path: Path,
    checkpoint_provenance_path: Path,
    raw_zip_status: str,
    raw_zip: Path,
    raw_zip_manifest: Mapping[str, Any] | None,
    summary_zip_status: str,
    summary_zip: Path,
    summary_zip_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
            "schema_version": "dir_artifact_completion_v1",
            "status": completion_status,
            "artifact_run_id": artifact_run_id,
            "artifact_identity_sha256": artifact_identity_sha256,
            "completed_at": _utc_now_iso(),
            "final_warning_ledger_role": (
                "authoritative_post_training_warning_ledger_after_all_archive_and_identity_checks"
            ),
            "post_training_warning_count": len(post_training_warnings),
            "post_training_warnings": list(post_training_warnings),
            "warning_module_count": len(final_warning_module_names),
            "warning_modules": final_warning_module_names,
            "archive_statuses": final_archive_statuses,
            "scientific_manifest_path": MANIFEST_FILE if manifest_path.is_file() else None,
            "scientific_manifest_sha256": scientific_manifest_sha256,
            "checkpoint_inventory": {
                "status": checkpoint_inventory_status,
                "directory": str(checkpoint_dir),
                "individual_checkpoint_count": sum(path.is_file() for path in checkpoint_files),
                "training_csv": str(checkpoint_training_csv_path),
                "provenance": str(checkpoint_provenance_path),
            },
            "archives": {
                "raw_report": {
                    "status": raw_zip_status,
                    "path": str(raw_zip),
                    "zip_manifest_sha256": (
                        _canonical_json_sha256(raw_zip_manifest)
                        if raw_zip_manifest is not None
                        else None
                    ),
                },
                "summary_report": {
                    "status": summary_zip_status,
                    "path": str(summary_zip),
                    "zip_manifest_sha256": (
                        _canonical_json_sha256(summary_zip_manifest)
                        if summary_zip_manifest is not None
                        else None
                    ),
                },
            },
        }


# Artifact packaging lifecycle
from ..artifacts import write_json_file as _write_json
from .common import (
    ARTIFACT_COMPLETION_FILE,
    CORE_MEASUREMENTS_FILE,
    MANIFEST_FILE,
    MODULE_MANIFEST_FILE,
    OWNERSHIP_FILE,
    RESULTS_OVERVIEW_FILE,
    SAMPLE_MANIFEST_FILE,
    SUMMARY_FILE,
    TRAINING_FILE,
    _sha256_file,
)
from .runtime import (
    _current_measurement_shards,
    _finalize_report_archive_fail_soft,
)


def _write_summary_and_overview(
    *,
    plan: Mapping[str, Any],
    training: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    dataset_sample_reference: Mapping[str, Any],
    post_training_warnings: list[dict[str, Any]],
    training_stage: TrainingStageResult,
    measurement_stage: MeasurementStageResult,
    artifact_run_id: str,
    measurement_contract_sha256: str,
    statistics_contract_sha256: str,
    output_dir: Path,
    record_post_training_warning: Callable[..., None],
) -> tuple[dict[str, Any], Path, Path]:
    summary = build_summary_payload(
        plan=plan,
        training=training,
        runtime_environment=runtime_environment,
        dataset_sample_reference=dataset_sample_reference,
        post_training_warnings=post_training_warnings,
        training_stage=training_stage,
        measurement_stage=measurement_stage,
        artifact_run_id=artifact_run_id,
        measurement_contract_sha256=measurement_contract_sha256,
        statistics_contract_sha256=statistics_contract_sha256,
    )
    summary_path = output_dir / SUMMARY_FILE
    try:
        _write_json(summary_path, summary)
    except Exception as exc:
        record_post_training_warning(
            stage="summary_file_write",
            message=(
                "summary JSON could not be written after the scientific measurements; "
                "continue raw/manifest/archive finalization with all available files"
            ),
            error=exc,
        )

    overview_path = output_dir / RESULTS_OVERVIEW_FILE
    try:
        _write_results_overview(
            overview_path,
            summary=summary,
            training_csv_path=output_dir / TRAINING_FILE,
        )
    except Exception as exc:
        record_post_training_warning(
            stage="results_overview_write",
            message=(
                "human-readable results overview could not be written after the "
                "scientific measurements; machine-readable summary/raw results remain "
                "authoritative and packaging continues"
            ),
            error=exc,
        )
    return summary, summary_path, overview_path


def _effective_config_provenance(
    config_path: Path,
    output_dir: Path,
    *,
    record_post_training_warning: Callable[..., None],
) -> tuple[Path, bool, str | None]:
    effective_config_path = config_path
    effective_config_sha256: str | None = None
    effective_config_available = bool(
        effective_config_path.is_file() and effective_config_path.parent == output_dir
    )
    if not effective_config_available:
        record_post_training_warning(
            stage="effective_run_config_pre_packaging",
            message=(
                "effective_run_config.json is missing or outside paths.output_dir after "
                "the scientific run; continue packaging available artifacts and mark "
                "provenance incomplete"
            ),
        )
        return effective_config_path, False, None

    try:
        effective_config_sha256 = _sha256_file(effective_config_path)
    except Exception as exc:
        effective_config_available = False
        record_post_training_warning(
            stage="effective_run_config_hash",
            message=(
                "effective_run_config.json could not be hashed after the scientific run; "
                "continue packaging available artifacts"
            ),
            error=exc,
        )
    return effective_config_path, effective_config_available, effective_config_sha256


def _collect_report_files(
    *,
    output_dir: Path,
    overview_path: Path,
    summary_path: Path,
    artifact_run_path: Path,
    effective_config_path: Path,
    effective_config_available: bool,
    measurement_shard_dir: Path,
    run_fingerprint: str,
    module_status: Mapping[str, Mapping[str, Any]],
    record_post_training_warning: Callable[..., None],
) -> tuple[list[Path], list[Path], list[str], list[str]]:
    # Public archive order is deliberate: readable entry points first, then data,
    # then metadata/provenance. Runtime storage order remains independent.
    optional_effective_config = (
        [effective_config_path] if effective_config_available else []
    )
    raw_candidates = [
        overview_path,
        output_dir / TRAINING_FILE,
        output_dir / CORE_MEASUREMENTS_FILE,
        output_dir / OWNERSHIP_FILE,
        output_dir / SAMPLE_MANIFEST_FILE,
        output_dir / MODULE_MANIFEST_FILE,
        artifact_run_path,
        *optional_effective_config,
    ]
    summary_candidates = [
        overview_path,
        summary_path,
        output_dir / OWNERSHIP_FILE,
        artifact_run_path,
        *optional_effective_config,
    ]
    raw_files = [path for path in raw_candidates if path.is_file()]
    summary_files = [path for path in summary_candidates if path.is_file()]
    missing_raw_files = [
        str(path.name) for path in raw_candidates if not path.is_file()
    ]
    missing_summary_files = [
        str(path.name) for path in summary_candidates if not path.is_file()
    ]
    if missing_raw_files:
        record_post_training_warning(
            stage="raw_file_inventory_pre_packaging",
            message=(
                "some expected raw files are unavailable after the scientific run; "
                f"package the remaining files: {missing_raw_files}"
            ),
        )
    if missing_summary_files:
        record_post_training_warning(
            stage="summary_file_inventory_pre_packaging",
            message=(
                "some expected summary files are unavailable after the scientific run; "
                f"package the remaining files: {missing_summary_files}"
            ),
        )
    try:
        current_measurement_shards = _current_measurement_shards(
            measurement_shard_dir,
            run_fingerprint=run_fingerprint,
            module_status=module_status,
        )
    except Exception as exc:
        current_measurement_shards = []
        record_post_training_warning(
            stage="measurement_shard_inventory_pre_packaging",
            message=(
                "measurement shard inventory could not be enumerated; continue packaging "
                "the consolidated in-memory/report results"
            ),
            error=exc,
        )
    raw_files.extend(current_measurement_shards)
    return raw_files, summary_files, missing_raw_files, missing_summary_files


def _verify_artifact_identity_before_packaging(
    artifact_run_path: Path,
    artifact_identity_sha256: str,
    *,
    record_post_training_warning: Callable[..., None],
) -> str:
    status = "completed"
    try:
        if _sha256_file(artifact_run_path) != artifact_identity_sha256:
            status = "warning_changed"
            record_post_training_warning(
                stage="artifact_identity_pre_packaging",
                message=(
                    "immutable artifact identity changed before packaging; continue "
                    "preserving scientific files but mark artifact identity provenance "
                    "as warning"
                ),
            )
    except Exception as exc:
        status = "warning_unavailable"
        record_post_training_warning(
            stage="artifact_identity_pre_packaging",
            message=(
                "immutable artifact identity could not be verified before packaging; "
                "continue with available scientific files"
            ),
            error=exc,
        )
    return status


def _mark_packaging_started(
    write_progress: Callable[[Mapping[str, Any]], None],
    run_fingerprint: str,
    *,
    record_post_training_warning: Callable[..., None],
) -> None:
    try:
        write_progress(
            {
                "stage": "artifact_packaging",
                "status": "scientific_run_complete_artifacts_pending",
                "measurement_run_fingerprint": run_fingerprint,
            }
        )
    except Exception as exc:
        record_post_training_warning(
            stage="artifact_packaging_progress",
            message="artifact-packaging progress marker failed; packaging continues",
            error=exc,
        )


def _finalize_completion_metadata(
    *,
    output_dir: Path,
    work_dir: Path,
    artifact_run_path: Path,
    artifact_run_id: str,
    artifact_identity_sha256: str,
    post_training_warnings: list[dict[str, Any]],
    module_status: Mapping[str, Mapping[str, Any]],
    checkpoint_inventory_status: Any,
    checkpoint_dir: Path,
    checkpoint_files: Any,
    checkpoint_training_csv_path: Any,
    checkpoint_provenance_path: Any,
    raw_zip_status: str,
    raw_zip: Path,
    raw_zip_manifest: Any,
    summary_zip_status: str,
    summary_zip: Path,
    summary_zip_manifest: Any,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    write_progress: Callable[[Mapping[str, Any]], None],
    record_post_training_warning: Callable[..., None],
) -> None:
    try:
        if _sha256_file(artifact_run_path) != artifact_identity_sha256:
            record_post_training_warning(
                stage="artifact_identity_post_packaging",
                message=(
                    "immutable artifact identity changed during packaging; scientific "
                    "files are preserved and completion is marked with a post-training "
                    "warning"
                ),
            )
    except Exception as exc:
        record_post_training_warning(
            stage="artifact_identity_post_packaging",
            message=(
                "immutable artifact identity could not be verified after packaging; "
                "scientific files and any completed archives are preserved"
            ),
            error=exc,
        )

    scientific_manifest_sha256: str | None = None
    if manifest_path.is_file():
        try:
            scientific_manifest_sha256 = _sha256_file(manifest_path)
        except Exception as exc:
            record_post_training_warning(
                stage="scientific_manifest_hash",
                message=(
                    "final scientific manifest exists but could not be hashed; completion "
                    "metadata continues with a null manifest hash"
                ),
                error=exc,
            )

    final_archive_statuses = {
        "raw_report": str(raw_zip_status),
        "summary_report": str(summary_zip_status),
    }
    final_warning_module_names = sorted(
        name
        for name, payload in module_status.items()
        if str(payload.get("status", "")).startswith("warning_")
    )
    completion_status = (
        "completed"
        if all(value == "completed" for value in final_archive_statuses.values())
        and not post_training_warnings
        and not final_warning_module_names
        else "completed_with_warnings"
    )
    completion = build_completion_payload(
        completion_status=completion_status,
        artifact_run_id=artifact_run_id,
        artifact_identity_sha256=artifact_identity_sha256,
        post_training_warnings=post_training_warnings,
        final_warning_module_names=final_warning_module_names,
        final_archive_statuses=final_archive_statuses,
        manifest_path=manifest_path,
        scientific_manifest_sha256=scientific_manifest_sha256,
        checkpoint_inventory_status=checkpoint_inventory_status,
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=checkpoint_files,
        checkpoint_training_csv_path=checkpoint_training_csv_path,
        checkpoint_provenance_path=checkpoint_provenance_path,
        raw_zip_status=raw_zip_status,
        raw_zip=raw_zip,
        raw_zip_manifest=raw_zip_manifest,
        summary_zip_status=summary_zip_status,
        summary_zip=summary_zip,
        summary_zip_manifest=summary_zip_manifest,
    )
    completion_path = output_dir / ARTIFACT_COMPLETION_FILE
    try:
        _write_json(completion_path, completion)
    except Exception as exc:
        record_post_training_warning(
            stage="artifact_completion_write",
            message=(
                "artifact completion marker could not be written; all previously completed "
                "scientific files and archives remain preserved"
            ),
            error=exc,
        )
    try:
        write_progress(
            {
                "stage": "complete",
                "status": completion["status"],
                "measurement_status": manifest["measurement_status"],
                "warning_module_count": manifest["warning_module_count"],
                "inconclusive_module_count": manifest["inconclusive_module_count"],
                "post_training_warning_count": completion[
                    "post_training_warning_count"
                ],
                "artifact_completion_file": (
                    ARTIFACT_COMPLETION_FILE if completion_path.is_file() else None
                ),
            }
        )
    except Exception:
        pass
    if completion_status == "completed":
        try:
            if work_dir.is_dir():
                shutil.rmtree(work_dir)
        except Exception:
            pass


def finalize_experiment_artifacts(
    *,
    raw_config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    work_dir: Path,
    checkpoint_dir: Path,
    artifact_run_id: str,
    artifact_run_path: Path,
    artifact_identity_sha256: str,
    plan: Mapping[str, Any],
    training: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    dataset_sample_reference: Mapping[str, Any],
    post_training_warnings: list[dict[str, Any]],
    training_stage: TrainingStageResult,
    measurement_stage: MeasurementStageResult,
    measurement_contract_sha256: str,
    statistics_contract_sha256: str,
    write_progress: Callable[[Mapping[str, Any]], None],
    write_post_training_json: Callable[..., bool],
    record_post_training_warning: Callable[..., None],
) -> str:
    checkpoint_files = training_stage.checkpoint_files
    checkpoint_inventory_status = training_stage.checkpoint_inventory_status
    checkpoint_provenance = training_stage.checkpoint_provenance
    checkpoint_provenance_path = training_stage.checkpoint_provenance_path
    checkpoint_provenance_rebuilt = training_stage.checkpoint_provenance_rebuilt
    checkpoint_provenance_sha256 = training_stage.checkpoint_provenance_sha256
    checkpoint_training_csv_path = training_stage.checkpoint_training_csv_path
    ownership = training_stage.ownership
    resume_from_checkpoints = training_stage.resume_from_checkpoints
    support_commit_output_parity = training_stage.support_commit_output_parity

    completed = measurement_stage.completed
    warning_count = measurement_stage.warning_count
    inconclusive_count = measurement_stage.inconclusive_count
    superseded_count = measurement_stage.superseded_count
    core_names = measurement_stage.core_names
    supplementary_names = measurement_stage.supplementary_names
    core_measurement_status = measurement_stage.core_measurement_status
    supplementary_measurement_status = measurement_stage.supplementary_measurement_status
    measurement_status = measurement_stage.measurement_status
    training_valid = measurement_stage.training_valid
    condition_comparisons = measurement_stage.condition_comparisons
    primary_results = measurement_stage.primary_results
    supporting_results = measurement_stage.supporting_results
    jacobian_rank32_results = measurement_stage.jacobian_rank32_results
    jacobian_stability_contract = measurement_stage.jacobian_stability_contract
    measurement_shard_dir = measurement_stage.measurement_shard_dir
    module_status = measurement_stage.module_status
    run_fingerprint = measurement_stage.run_fingerprint
    statistics_fingerprint = measurement_stage.statistics_fingerprint
    stale_measurement_shards_removed = measurement_stage.stale_measurement_shards_removed
    current_measurement_shard_count = measurement_stage.current_measurement_shard_count

    summary, summary_path, overview_path = _write_summary_and_overview(
        plan=plan,
        training=training,
        runtime_environment=runtime_environment,
        dataset_sample_reference=dataset_sample_reference,
        post_training_warnings=post_training_warnings,
        training_stage=training_stage,
        measurement_stage=measurement_stage,
        artifact_run_id=artifact_run_id,
        measurement_contract_sha256=measurement_contract_sha256,
        statistics_contract_sha256=statistics_contract_sha256,
        output_dir=output_dir,
        record_post_training_warning=record_post_training_warning,
    )

    raw_zip = Path(str(raw_config["paths"]["raw_report_zip"])).expanduser().resolve()
    summary_zip = Path(
        str(raw_config["paths"]["summary_report_zip"])
    ).expanduser().resolve()
    (
        effective_config_path,
        effective_config_available,
        effective_config_sha256,
    ) = _effective_config_provenance(
        config_path,
        output_dir,
        record_post_training_warning=record_post_training_warning,
    )
    (
        raw_files,
        summary_files,
        missing_raw_files,
        missing_summary_files,
    ) = _collect_report_files(
        output_dir=output_dir,
        overview_path=overview_path,
        summary_path=summary_path,
        artifact_run_path=artifact_run_path,
        effective_config_path=effective_config_path,
        effective_config_available=effective_config_available,
        measurement_shard_dir=measurement_shard_dir,
        run_fingerprint=run_fingerprint,
        module_status=module_status,
        record_post_training_warning=record_post_training_warning,
    )
    artifact_identity_prepackaging_status = (
        _verify_artifact_identity_before_packaging(
            artifact_run_path,
            artifact_identity_sha256,
            record_post_training_warning=record_post_training_warning,
        )
    )
    _mark_packaging_started(
        write_progress,
        run_fingerprint,
        record_post_training_warning=record_post_training_warning,
    )
    raw_zip_manifest, raw_zip_status = _finalize_report_archive_fail_soft(
        zip_path=raw_zip,
        root=output_dir,
        files=raw_files,
        artifact_run_id=artifact_run_id,
        artifact_kind="raw_report",
        run_marker_path=artifact_run_path,
        warning_stage="raw_report_archive",
        warning_message=(
            "raw report ZIP finalization failed after the scientific run; "
            "unpacked raw files are preserved and summary packaging continues"
        ),
        warning_recorder=record_post_training_warning,
    )
    manifest = build_manifest_payload(
        summary=summary,
        resume_from_checkpoints=resume_from_checkpoints,
        checkpoint_provenance=checkpoint_provenance,
        artifact_run_id=artifact_run_id,
        artifact_identity_sha256=artifact_identity_sha256,
        effective_config_path=effective_config_path,
        effective_config_available=effective_config_available,
        effective_config_sha256=effective_config_sha256,
        artifact_identity_prepackaging_status=artifact_identity_prepackaging_status,
        missing_raw_files=missing_raw_files,
        missing_summary_files=missing_summary_files,
        run_fingerprint=run_fingerprint,
        measurement_contract_sha256=measurement_contract_sha256,
        statistics_contract_sha256=statistics_contract_sha256,
        statistics_fingerprint=statistics_fingerprint,
        raw_files=raw_files,
        summary_files=summary_files,
        output_dir=output_dir,
        raw_zip_manifest=raw_zip_manifest,
        raw_zip_status=raw_zip_status,
        checkpoint_inventory_status=checkpoint_inventory_status,
        post_training_warnings=post_training_warnings,
        raw_zip=raw_zip,
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=checkpoint_files,
        summary_zip=summary_zip,
    )
    manifest_path = output_dir / MANIFEST_FILE
    try:
        _write_json(manifest_path, manifest)
    except Exception as exc:
        record_post_training_warning(
            stage="final_manifest_write",
            message=(
                "final manifest write failed after scientific measurements; continue summary archive "
                "finalization with all files that were successfully written"
            ),
            error=exc,
        )
    if manifest_path.is_file():
        summary_files.append(manifest_path)
    summary_zip_manifest, summary_zip_status = _finalize_report_archive_fail_soft(
        zip_path=summary_zip,
        root=output_dir,
        files=summary_files,
        artifact_run_id=artifact_run_id,
        artifact_kind="summary_report",
        run_marker_path=artifact_run_path,
        warning_stage="summary_report_archive",
        warning_message=(
            "summary report ZIP finalization failed after the scientific run; "
            "unpacked summary and manifest files are preserved"
        ),
        warning_recorder=record_post_training_warning,
    )
    _finalize_completion_metadata(
        output_dir=output_dir,
        work_dir=work_dir,
        artifact_run_path=artifact_run_path,
        artifact_run_id=artifact_run_id,
        artifact_identity_sha256=artifact_identity_sha256,
        post_training_warnings=post_training_warnings,
        module_status=module_status,
        checkpoint_inventory_status=checkpoint_inventory_status,
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=checkpoint_files,
        checkpoint_training_csv_path=checkpoint_training_csv_path,
        checkpoint_provenance_path=checkpoint_provenance_path,
        raw_zip_status=raw_zip_status,
        raw_zip=raw_zip,
        raw_zip_manifest=raw_zip_manifest,
        summary_zip_status=summary_zip_status,
        summary_zip=summary_zip,
        summary_zip_manifest=summary_zip_manifest,
        manifest=manifest,
        manifest_path=manifest_path,
        write_progress=write_progress,
        record_post_training_warning=record_post_training_warning,
    )
    return str(output_dir)
