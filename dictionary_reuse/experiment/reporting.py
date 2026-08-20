"""Reporting, matrix statistics, summaries, and fail-soft measurement execution."""

from __future__ import annotations


# Measurement module execution and status
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import capture_model_runtime_signature
from .common import (
    _read_gzip_json, _reusable_shard_status, _safe_module_file_name, _write_gzip_json,
)
from .runtime import _restore_models_from_checkpoints, _write_module_manifest

def _run_measurement_module(
    *,
    name: str,
    phase: str,
    function: Any,
    models: Mapping[str, nn.Module],
    module_results: dict[str, Any],
    module_status: dict[str, Any],
    shard_dir: Path,
    module_manifest_path: Path,
    run_fingerprint: str,
    model_checkpoint_paths: Mapping[int, Path],
    warning_recorder: Any | None = None,
) -> None:
    shard_path = shard_dir / _safe_module_file_name(name)
    if shard_path.is_file():
        try:
            cached = _read_gzip_json(shard_path)
            if (
                str(cached.get("run_fingerprint", "")) == str(run_fingerprint)
                and str(cached.get("name", "")) == str(name)
            ):
                cached_status = dict(cached.get("status", {}) or {})
                original_status = str(cached_status.get("status", "completed"))
                if _reusable_shard_status(original_status):
                    module_results[name] = cached.get("result", {})
                    if original_status in {"completed", "reused_completed"}:
                        cached_status["status"] = "reused_completed"
                    else:
                        # Preserve the scientific view-validity category while
                        # recording reuse separately. Prefixing it with
                        # ``reused_`` would hide partial/inconclusive phase state.
                        cached_status["status"] = original_status
                    cached_status["reused_from_shard"] = True
                    module_status[name] = cached_status
                    try:
                        _write_module_manifest(
                            module_manifest_path,
                            run_fingerprint=run_fingerprint,
                            module_status=module_status,
                        )
                    except Exception as exc:
                        cached_status.setdefault("artifact_warnings", []).append(
                            f"module_manifest_write_failed: {type(exc).__name__}: {exc}"
                        )
                        if warning_recorder is not None:
                            warning_recorder(
                                stage=f"measurement_manifest:{name}",
                                message=(
                                    "measurement module was recovered from a valid shard, but the "
                                    "module manifest refresh failed; continue with remaining measurements"
                                ),
                                error=exc,
                            )
                    return
        except Exception:
            # Corrupt or stale shards are ignored and replaced by a fresh result.
            pass

    before = {
        key: capture_model_runtime_signature(model) for key, model in models.items()
    }
    error = ""
    trace = ""
    result: Any = {}
    try:
        result = function()
        if isinstance(result, Mapping):
            status = _inferred_result_measurement_status(result)
        else:
            status = "completed"
    except Exception as exc:
        status = "warning_module_exception"
        error = f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc()
        result = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = {
        key: capture_model_runtime_signature(model) for key, model in models.items()
    }
    state_unchanged = before == after
    restored_models: dict[str, str] = {}
    if not state_unchanged:
        status = "warning_model_state_changed"
        restored_models = _restore_models_from_checkpoints(
            models, model_checkpoint_paths
        )
        if error:
            error += "; "
        error += "measurement mutated model runtime state; affected models reloaded when checkpoints were available"
    auxiliary_measurement_status: dict[str, str] = {}
    if isinstance(result, Mapping):
        cosine_status = result.get("cosine_measurement_status")
        if cosine_status is not None:
            auxiliary_measurement_status["paired_cosine"] = str(cosine_status)
    primary_measurement_status = (
        _primary_metric_view_statuses(result)
        if isinstance(result, Mapping)
        else {}
    )
    status_payload = {
        "phase": str(phase),
        "status": status,
        "primary_measurement_status": primary_measurement_status,
        "error": error,
        "traceback": trace,
        "model_state_unchanged": state_unchanged,
        "restored_models": restored_models,
        "reused_from_shard": False,
        "auxiliary_measurement_status": auxiliary_measurement_status,
    }
    module_results[name] = result
    module_status[name] = status_payload

    # Scientific/module warnings must participate in the authoritative final
    # post-training ledger, not only in the per-module status table. This is
    # intentionally recorded before artifact persistence so a later manifest
    # or ZIP failure cannot hide the original measurement warning.
    if warning_recorder is not None and str(status).startswith("warning_"):
        warning_recorder(
            stage=f"measurement_module:{name}",
            message=(
                f"measurement module finished with {status}; preserve any completed "
                "sibling/partial results and mark final completion with warnings"
            ),
        )
    if isinstance(result, Mapping):
        cleanup_payload = result.get("cache_cleanup")
        if isinstance(cleanup_payload, Mapping) and str(
            cleanup_payload.get("status", "")
        ).startswith("warning_"):
            status_payload.setdefault("auxiliary_warnings", []).append(
                {
                    "kind": "causal_cache_cleanup",
                    "status": str(cleanup_payload.get("status")),
                    "warning": str(cleanup_payload.get("warning", "")),
                }
            )
            if warning_recorder is not None:
                warning_recorder(
                    stage=f"measurement_cache_cleanup:{name}",
                    message=(
                        "causal measurement results were computed successfully but temporary "
                        "cache cleanup failed; preserve the scientific result and mark final "
                        "completion with warnings"
                    ),
                )
    artifact_warnings: list[str] = []
    try:
        _write_gzip_json(
            shard_path,
            {
                "schema_version": "dir_measurement_shard_v2",
                "name": name,
                "phase": str(phase),
                "run_fingerprint": str(run_fingerprint),
                "result": result,
                "status": status_payload,
            },
        )
    except Exception as exc:
        artifact_warnings.append(
            f"measurement_shard_write_failed: {type(exc).__name__}: {exc}"
        )
        if warning_recorder is not None:
            warning_recorder(
                stage=f"measurement_shard:{name}",
                message=(
                    "measurement completed in memory but its reusable shard could not be written; "
                    "continue with remaining measurements and keep the in-memory result"
                ),
                error=exc,
            )
    if artifact_warnings:
        status_payload["artifact_warnings"] = list(artifact_warnings)
        status_payload["artifact_persistence_status"] = "warning"
    else:
        status_payload["artifact_persistence_status"] = "completed"
    try:
        _write_module_manifest(
            module_manifest_path,
            run_fingerprint=run_fingerprint,
            module_status=module_status,
        )
    except Exception as exc:
        artifact_warnings.append(
            f"module_manifest_write_failed: {type(exc).__name__}: {exc}"
        )
        status_payload["artifact_warnings"] = list(artifact_warnings)
        status_payload["artifact_persistence_status"] = "warning"
        if warning_recorder is not None:
            warning_recorder(
                stage=f"measurement_manifest:{name}",
                message=(
                    "measurement module manifest write failed after the module result was computed; "
                    "continue with remaining measurements"
                ),
                error=exc,
            )

def _is_square_numeric_matrix(payload: Any) -> bool:
    if not isinstance(payload, list) or len(payload) < 2:
        return False
    size = len(payload)
    return all(
        isinstance(row, list)
        and len(row) == size
        and all(isinstance(value, (int, float)) for value in row)
        for row in payload
    )

def _primary_metric_view_statuses(result: Mapping[str, Any]) -> dict[str, str]:
    """Classify every declared primary view, including nested Jacobian views.

    Matrix-backed primary CKA views are validated here from their actual matrix
    and mask. Measurements with nested primary objects (for example rank-32
    Jacobian descriptors and internal-VJP paths) declare their own statuses in
    ``primary_measurement_status``; those declarations are preserved alongside
    the independently revalidated matrix-backed views.
    """

    output: dict[str, str] = {}
    declared = result.get("primary_measurement_status", {})
    if isinstance(declared, Mapping):
        output.update({str(key): str(value) for key, value in declared.items()})

    primary_metrics = result.get("primary_metrics", [])
    masks = result.get("validity_masks", {})
    if not isinstance(primary_metrics, Sequence) or isinstance(
        primary_metrics, (str, bytes)
    ):
        return output
    for raw_key in primary_metrics:
        key = str(raw_key)
        matrix = result.get(key)
        if not _is_square_numeric_matrix(matrix):
            output[key] = "inconclusive_invalid_primary_matrix"
            continue
        matrix_array = np.asarray(matrix, dtype=np.float64)
        mask = masks.get(key) if isinstance(masks, Mapping) else None
        if mask is None:
            valid_mask = np.isfinite(matrix_array)
        else:
            valid_mask = np.asarray(mask, dtype=bool)
            if valid_mask.shape != matrix_array.shape:
                output[key] = "inconclusive_invalid_primary_mask"
                continue
        diagonal_valid = np.diag(valid_mask) & np.isfinite(np.diag(matrix_array))
        if not bool(diagonal_valid.any()):
            output[key] = "inconclusive_no_valid_diagonal"
            continue
        if not bool(np.isfinite(matrix_array[valid_mask]).all()):
            output[key] = "inconclusive_nonfinite_valid_primary_cells"
            continue
        output[key] = "valid"
    return output

def _inferred_result_measurement_status(result: Mapping[str, Any]) -> str:
    explicit = result.get("measurement_status")

    def nested_nonfinite_status(value: Any) -> str | None:
        if isinstance(value, Mapping):
            status = value.get("measurement_status")
            if str(status).startswith("inconclusive_nonfinite"):
                return str(status)
            for nested in value.values():
                found = nested_nonfinite_status(nested)
                if found is not None:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                found = nested_nonfinite_status(nested)
                if found is not None:
                    return found
        return None

    nonfinite_status = nested_nonfinite_status(result)
    if nonfinite_status is not None:
        return nonfinite_status

    view_statuses = _primary_metric_view_statuses(result)
    explicit_status = str(explicit) if explicit is not None else ""
    if explicit_status and explicit_status not in {
        "completed",
        "pending_primary_view_validation",
    }:
        # The measurement function may aggregate more primary views than the
        # top-level CKA matrices (notably rank-32 Jacobian descriptors and
        # internal-VJP paths). Never upgrade that broader partial or
        # inconclusive decision merely because the top-level CKA views passed.
        return explicit_status
    if not view_statuses:
        return explicit_status or "completed"
    valid_count = sum(value == "valid" for value in view_statuses.values())
    if valid_count == len(view_statuses):
        return "completed"
    if valid_count > 0:
        return "partial_primary_views"
    if explicit_status.startswith("inconclusive_no_valid_"):
        return explicit_status
    return "inconclusive_no_valid_primary_cka"


# Matrix statistics and paired comparisons
from ..statistics import (
    block_axis_permutation_test, matrix_summary, paired_bootstrap_difference,
    same_index_advantage_vector,
)

def _matrix_objective(path: Sequence[str]) -> str:
    key = str(path[-1]).lower() if path else ""
    if "rms_effect" in key or "disruption" in key or "normalized_l2" in key or "distance" in key:
        return "minimize"
    return "maximize"

def _is_scientific_alignment_matrix(
    key: str,
    *,
    declared_primary_metrics: Sequence[Any] | None = None,
) -> bool:
    """Return whether a 12x12 numeric matrix is an alignment statistic target.

    Quality-control matrices such as sample counts, validity masks, and
    categorical/rank classifications are intentionally excluded from
    same-index rank and permutation inference. They remain in the report as
    diagnostics but are not treated as evidence of functional alignment.
    """

    name = str(key)
    lower = name.lower()
    declared = {
        str(value)
        for value in (declared_primary_metrics or ())
        if isinstance(value, (str, bytes))
    }
    if name in declared:
        return True
    if any(
        token in lower
        for token in (
            "sample_count",
            "valid_count",
            "validity_mask",
            "finite_sample",
            "nonfinite",
            "both_zero",
            "one_zero",
            "classification",
            "prediction_retention",
        )
    ):
        return False
    scientific_tokens = (
        "cka",
        "cosine",
        "subspace_overlap",
        "normalized_l2",
        "norm_ratio",
        "rms_effect",
        "disruption",
        "transport",
    )
    if any(token in lower for token in scientific_tokens):
        return True
    # Jacobian response matrices use these names without an explicit metric
    # suffix; they are genuine block-to-block functional comparisons.
    if lower.startswith("input_to_"):
        return True
    return False

def _matrix_reports(
    payload: Any,
    *,
    samples: dict[str, Any],
    seed: int,
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(payload, dict):
        masks = payload.get("validity_masks", {})
        declared_primary_metrics = payload.get("primary_metrics", [])
        output: dict[str, Any] = {}
        for index, (key, value) in enumerate(payload.items()):
            if key == "validity_masks":
                output[key] = value
                continue
            if _is_square_numeric_matrix(value):
                if not _is_scientific_alignment_matrix(
                    str(key), declared_primary_metrics=declared_primary_metrics
                ):
                    output[key] = {
                        "matrix": value,
                        "statistics_status": "not_applicable_diagnostic_matrix",
                        "statistics_reason": (
                            "quality_control_or_non_alignment_matrix_excluded_from_rank_and_permutation"
                        ),
                    }
                    continue
                objective = _matrix_objective((*path, str(key)))
                valid_mask = masks.get(key) if isinstance(masks, dict) else None
                try:
                    output[key] = {
                        "matrix": value,
                        "validity_mask": valid_mask,
                        "objective": objective,
                        "summary": matrix_summary(
                            value, objective=objective, valid_mask=valid_mask
                        ),
                        "permutation": block_axis_permutation_test(
                            value,
                            global_permutations=int(samples["global_permutations"]),
                            depth_band_permutations=int(samples["depth_band_permutations"]),
                            seed=seed + index,
                            objective=objective,
                            valid_mask=valid_mask,
                        ),
                    }
                except Exception as error:
                    output[key] = {
                        "matrix": value,
                        "validity_mask": valid_mask,
                        "statistics_status": "unavailable",
                        "statistics_error": f"{type(error).__name__}: {error}",
                    }
            else:
                output[key] = _matrix_reports(
                    value,
                    samples=samples,
                    seed=seed + index,
                    path=(*path, str(key)),
                )
        return output
    if _is_square_numeric_matrix(payload):
        key = str(path[-1]) if path else ""
        if not _is_scientific_alignment_matrix(key):
            return {
                "matrix": payload,
                "statistics_status": "not_applicable_diagnostic_matrix",
                "statistics_reason": (
                    "quality_control_or_non_alignment_matrix_excluded_from_rank_and_permutation"
                ),
            }
        objective = _matrix_objective(path)
        return {
            "matrix": payload,
            "objective": objective,
            "summary": matrix_summary(payload, objective=objective),
            "permutation": block_axis_permutation_test(
                payload,
                global_permutations=int(samples["global_permutations"]),
                depth_band_permutations=int(samples["depth_band_permutations"]),
                seed=seed,
                objective=objective,
            ),
        }
    return payload

def _paired_matrix_difference_reports(
    left_payload: Any,
    right_payload: Any,
    *,
    bootstrap_iterations: int,
    seed: int,
    path: tuple[str, ...] = (),
    left_mask: Any = None,
    right_mask: Any = None,
) -> Any:
    if isinstance(left_payload, dict) and isinstance(right_payload, dict):
        if path and path[-1] in {"paired_output_metrics", "paired_shared_projection_metrics"}:
            return {
                "status": "reported_in_primary_summary_with_primary_CKA_validity_mask",
                "reason": "Avoid unmasked duplicate condition comparisons for paired output diagnostics.",
            }
        common = sorted((set(left_payload) & set(right_payload)) - {"validity_masks"})
        left_masks = left_payload.get("validity_masks", {})
        right_masks = right_payload.get("validity_masks", {})
        return {
            key: _paired_matrix_difference_reports(
                left_payload[key],
                right_payload[key],
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + index,
                path=(*path, str(key)),
                left_mask=(left_masks.get(key) if isinstance(left_masks, dict) else None),
                right_mask=(right_masks.get(key) if isinstance(right_masks, dict) else None),
            )
            for index, key in enumerate(common)
        }
    if _is_square_numeric_matrix(left_payload) and _is_square_numeric_matrix(right_payload):
        left = np.asarray(left_payload, dtype=np.float64)
        right = np.asarray(right_payload, dtype=np.float64)
        if left.shape != right.shape:
            return None
        if left_mask is None:
            left_valid = np.ones(left.shape, dtype=bool)
        else:
            left_valid = np.asarray(left_mask, dtype=bool)
        if right_mask is None:
            right_valid = np.ones(right.shape, dtype=bool)
        else:
            right_valid = np.asarray(right_mask, dtype=bool)
        if left_valid.shape != left.shape or right_valid.shape != right.shape:
            return None
        common_valid = (
            left_valid
            & right_valid
            & np.isfinite(left)
            & np.isfinite(right)
        )
        objective = _matrix_objective(path)
        multiplier = 1.0 if objective == "maximize" else -1.0
        alignment_difference = multiplier * left - multiplier * right
        left_advantage = same_index_advantage_vector(
            left.tolist(), objective=objective, valid_mask=common_valid.tolist(), symmetric=True
        )
        right_advantage = same_index_advantage_vector(
            right.tolist(), objective=objective, valid_mask=common_valid.tolist(), symmetric=True
        )
        alignment_difference_summary = matrix_summary(
            alignment_difference.tolist(), objective="maximize", valid_mask=common_valid.tolist()
        )
        report: dict[str, Any] = {
            "objective": objective,
            "common_validity_mask": common_valid.tolist(),
            "raw_difference_matrix_left_minus_right": (left - right).tolist(),
            "alignment_score_difference_matrix": alignment_difference.tolist(),
            "alignment_score_difference_summary": alignment_difference_summary,
            "symmetric_direction_coverage": {
                "direction_count_by_block": alignment_difference_summary.get(
                    "symmetric_direction_count_by_block", []
                ),
                "two_direction_block_count": alignment_difference_summary.get(
                    "two_direction_block_count", 0
                ),
                "one_direction_block_count": alignment_difference_summary.get(
                    "one_direction_block_count", 0
                ),
                "zero_direction_block_count": alignment_difference_summary.get(
                    "zero_direction_block_count", 0
                ),
                "contract": (
                    "one_available_direction_is_retained_not_excluded_and_is_explicitly_recorded"
                ),
            },
            "bootstrap_unit": f"matched_valid_block_indices_not_individual_samples",
            "primary_pairing_contract": (
                "left_minus_right_of_row_column_same_index_advantage_relative_to_valid_other_blocks_in_the_same_depth_band_using_both_directions_when_available_or_the_single_available_direction_with_coverage_recorded"
            ),
            "raw_diagonal_role": "auxiliary_not_primary",
        }
        try:
            report["depth_band_matched_same_index_advantage_bootstrap"] = paired_bootstrap_difference(
                left_advantage,
                right_advantage,
                iterations=int(bootstrap_iterations),
                seed=int(seed),
            )
        except ValueError as error:
            report["depth_band_matched_same_index_advantage_bootstrap"] = {
                "status": "inconclusive_no_common_valid_blocks",
                "error": str(error),
            }
        diagonal_indices = [i for i in range(left.shape[0]) if common_valid[i, i]]
        if diagonal_indices:
            report["raw_same_index_diagonal_bootstrap_auxiliary"] = paired_bootstrap_difference(
                [float(left[i, i]) for i in diagonal_indices],
                [float(right[i, i]) for i in diagonal_indices],
                iterations=int(bootstrap_iterations),
                seed=int(seed) + 1,
            )
        else:
            report["raw_same_index_diagonal_bootstrap_auxiliary"] = {
                "status": "inconclusive_no_common_valid_diagonal"
            }
        return report
    return None

def _paired_corresponding_output_diagnostics(
    payload: Mapping[str, Any],
    *,
    module_suffix: str,
    matrix_key: str,
    valid_mask: Any,
) -> dict[str, Any] | None:
    """Summarize actual same-index output agreement alongside structure-only CKA."""

    source: Mapping[str, Any] | None = None
    vector_mode = False
    if module_suffix == "direct_function":
        mode = {
            "single_bidirectional_mean_full_token_debiased_cka_12x12": "full_token",
            "single_bidirectional_mean_cls_debiased_cka_12x12": "cls",
            "single_bidirectional_mean_patch_debiased_cka_12x12": "patch",
        }.get(matrix_key)
        if mode is not None:
            source = payload.get("same_index_paired_output_metrics", {}).get(mode, {})
            vector_mode = True
    elif module_suffix.startswith("ablation."):
        output_name = matrix_key.removesuffix("_debiased_cka_12x12")
        source = payload.get("paired_output_metrics", {}).get(output_name, {})
    elif module_suffix.startswith("patching."):
        output_name = matrix_key.removeprefix("common_valid_").removesuffix(
            "_debiased_cka_12x12"
        )
        source = payload.get("paired_output_metrics", {}).get(output_name, {})
    elif module_suffix == "jacobian.input_jvp":
        output_name = {
            "input_to_block_update_full_debiased_cka_12x12": "block_update_full",
            "input_to_block_update_cls_debiased_cka_12x12": "block_update_cls",
            "input_to_block_update_patch_debiased_cka_12x12": "block_update_patch",
            "input_to_class_token_debiased_cka_12x12": "class_token",
        }.get(matrix_key)
        if output_name is not None:
            source = payload.get("paired_shared_projection_metrics", {}).get(output_name, {})
    if not isinstance(source, Mapping):
        return None

    metric_keys = {
        "signed_cosine": ("signed_cosine_mean" if vector_mode else "signed_cosine_12x12"),
        "normalized_l2": ("normalized_l2_mean" if vector_mode else "normalized_l2_12x12"),
        "symmetric_norm_ratio": (
            "symmetric_norm_ratio_mean" if vector_mode else "symmetric_norm_ratio_12x12"
        ),
    }
    mask = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    output: dict[str, Any] = {}
    for label, source_key in metric_keys.items():
        value = source.get(source_key)
        if vector_mode:
            if not isinstance(value, Mapping):
                continue
            vector = value.get("1")
            if not isinstance(vector, list):
                continue
            selected = [
                float(item)
                for index, item in enumerate(vector)
                if np.isfinite(float(item))
                and (mask is None or (index < mask.shape[0] and bool(mask[index, index])))
            ]
            by_block = [
                (
                    float(item)
                    if np.isfinite(float(item))
                    and (mask is None or (index < mask.shape[0] and bool(mask[index, index])))
                    else float("nan")
                )
                for index, item in enumerate(vector)
            ]
        else:
            if not _is_square_numeric_matrix(value):
                continue
            array = np.asarray(value, dtype=np.float64)
            selected = [
                float(array[index, index])
                for index in range(array.shape[0])
                if np.isfinite(array[index, index])
                and (mask is None or bool(mask[index, index]))
            ]
            by_block = [
                (
                    float(array[index, index])
                    if np.isfinite(array[index, index])
                    and (mask is None or bool(mask[index, index]))
                    else float("nan")
                )
                for index in range(array.shape[0])
            ]
        output[label] = {
            "same_index_mean": float(np.mean(selected)) if selected else float("nan"),
            "same_index_values": selected,
            "same_index_by_block": by_block,
            "valid_block_count": len(selected),
            "objective": "minimize" if label == "normalized_l2" else "maximize",
        }
    return output or None


# Paper-facing summaries
from ..statistics import block_axis_permutation_test, matrix_summary
from .common import _canonical_module_name, _reportable_status
from .matrix import CONDITION_ORDER, PAIR_BY_CONDITION, SUMMARY_COMPARISONS

def _primary_matrix_record(
    module_results: Mapping[str, Any],
    module_status: Mapping[str, Any],
    *,
    condition: str,
    task: str,
    module_suffix: str,
    matrix_key: str,
    global_permutations: int,
    depth_band_permutations: int,
    seed: int,
    reportable_modules: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one primary matrix record without changing the underlying metric."""

    module_name = f"{condition}.{task}.core.{module_suffix}"
    canonical_name = _canonical_module_name(module_name, module_results, module_status)
    status = str(module_status.get(canonical_name or module_name, {}).get("status", "missing"))
    payload = module_results.get(canonical_name, {}) if canonical_name is not None else {}
    matrix = payload.get(matrix_key) if isinstance(payload, dict) else None
    valid_mask = (
        payload.get("validity_masks", {}).get(matrix_key)
        if isinstance(payload, dict) and isinstance(payload.get("validity_masks", {}), dict)
        else None
    )
    if canonical_name is None or not _is_square_numeric_matrix(matrix):
        return {
            "module_status": status,
            "canonical_module": canonical_name,
            "status": "unavailable_or_inconclusive",
            "statistics_status": "unavailable_or_inconclusive",
        }

    objective = _matrix_objective((matrix_key,))
    prepared_record = None
    if isinstance(reportable_modules, Mapping):
        prepared_module = reportable_modules.get(canonical_name, {})
        if isinstance(prepared_module, Mapping):
            candidate = prepared_module.get(matrix_key)
            if isinstance(candidate, Mapping) and _is_square_numeric_matrix(candidate.get("matrix")):
                prepared_record = candidate
    if prepared_record is not None:
        summary = dict(prepared_record.get("summary", {}) or {})
        permutation = dict(prepared_record.get("permutation", {}) or {})
        valid_mask = prepared_record.get("validity_mask", valid_mask)
    else:
        has_valid_diagonal = True
        if valid_mask is not None:
            valid_array = np.asarray(valid_mask, dtype=bool)
            has_valid_diagonal = bool(
                valid_array.ndim == 2
                and valid_array.shape[0] == valid_array.shape[1]
                and np.diag(valid_array).any()
            )
        if not has_valid_diagonal:
            summary = {"status": "inconclusive_no_valid_diagonal", "valid_diagonal_count": 0}
            permutation = {"status": "inconclusive_no_valid_diagonal"}
        else:
            try:
                summary = matrix_summary(matrix, objective=objective, valid_mask=valid_mask)
            except ValueError as error:
                summary = {"status": "inconclusive_no_valid_diagonal", "error": str(error)}
            try:
                permutation = block_axis_permutation_test(
                    matrix,
                    global_permutations=int(global_permutations),
                    depth_band_permutations=int(depth_band_permutations),
                    seed=int(seed),
                    objective=objective,
                    valid_mask=valid_mask,
                )
            except ValueError as error:
                permutation = {"status": "inconclusive_no_valid_diagonal", "error": str(error)}

    diagonal_available = bool(
        int(summary.get("valid_diagonal_count", 0)) > 0
        and np.isfinite(float(summary.get("diagonal_mean", float("nan"))))
    )
    rank_available = bool(
        str(summary.get("rank_statistics_status", "")) == "available"
        and np.isfinite(float(summary.get("same_index_rank_mean", float("nan"))))
    )
    advantage_available = bool(
        str(summary.get("advantage_statistics_status", "")) == "available"
        and np.isfinite(float(summary.get("depth_band_matched_same_index_margin", float("nan"))))
    )
    permutation_status = str(permutation.get("status", "unavailable"))
    if not diagonal_available:
        statistics_status = "inconclusive_no_valid_diagonal"
    elif rank_available and advantage_available and permutation_status == "available":
        statistics_status = "available"
    else:
        statistics_status = "partial"

    record: dict[str, Any] = {
        "module_status": status,
        "canonical_module": canonical_name,
        "objective": objective,
        "matrix": matrix,
        "validity_mask": valid_mask,
        "matrix_summary": summary,
        "permutation": permutation,
        "statistics_status": statistics_status,
        "statistic_status": {
            "diagonal": "available" if diagonal_available else "inconclusive_no_valid_diagonal",
            "depth_band_advantage": (
                "available"
                if advantage_available
                else str(summary.get("advantage_statistics_status", "inconclusive_no_valid_depth_band_competitor"))
            ),
            "rank": (
                "available"
                if rank_available
                else str(summary.get("rank_statistics_status", "inconclusive_no_valid_off_diagonal_competitor"))
            ),
            "permutation": permutation_status,
        },
    }
    if diagonal_available:
        record.update({"valid_diagonal_count": summary["valid_diagonal_count"], "diagonal_mean": summary["diagonal_mean"]})
    if advantage_available:
        record["symmetric_depth_band_matched_same_index_margin"] = summary[
            "depth_band_matched_same_index_margin"
        ]
    if rank_available:
        record.update(
            {
                "symmetric_same_index_rank_mean": summary["same_index_rank_mean"],
                "rank1_fraction": summary["rank1_fraction"],
                "top3_fraction": summary["top3_fraction"],
                "row_rank1_fraction": summary["row_rank1_fraction"],
                "column_rank1_fraction": summary["column_rank1_fraction"],
                "symmetric_rank_defined_count": summary["symmetric_rank_defined_count"],
            }
        )
    paired_diagnostics = _paired_corresponding_output_diagnostics(
        payload,
        module_suffix=module_suffix,
        matrix_key=matrix_key,
        valid_mask=valid_mask,
    )
    if paired_diagnostics is not None:
        record["paired_corresponding_output_diagnostics"] = paired_diagnostics
    return record


def _primary_metric_summary(
    module_results: Mapping[str, Any],
    module_status: Mapping[str, Any],
    *,
    bootstrap_iterations: int,
    global_permutations: int = 5000,
    depth_band_permutations: int = 13824,
    seed: int,
    reportable_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize exactly eight primary metric families.

    Mask, blur, and noise patching remain separate measured variants inside the
    two patching families. They are never numerically averaged across corruption
    types, so the family view adds no new scientific statistic.
    """

    del bootstrap_iterations  # Paired bootstrap lives in control-comparison reports.
    specs: list[tuple[str, tuple[tuple[str, str, str, int], ...]]] = [
        (
            "direct_function.single_bidirectional_mean_cls_debiased_cka_12x12",
            (("direct", "direct_function", "single_bidirectional_mean_cls_debiased_cka_12x12", 0),),
        ),
        (
            "direct_function.single_bidirectional_mean_patch_debiased_cka_12x12",
            (("direct", "direct_function", "single_bidirectional_mean_patch_debiased_cka_12x12", 1),),
        ),
        (
            "ablation.block_update.post_layernorm_cls_delta_debiased_cka_12x12",
            (("ablation", "ablation.block_update", "post_layernorm_cls_delta_debiased_cka_12x12", 2),),
        ),
        (
            "ablation.block_update.post_layernorm_patch_delta_debiased_cka_12x12",
            (("ablation", "ablation.block_update", "post_layernorm_patch_delta_debiased_cka_12x12", 3),),
        ),
        (
            "patching.block_update.common_valid_post_layernorm_cls_recovery_debiased_cka_12x12",
            tuple(
                (
                    corruption,
                    f"patching.{corruption}.block_update",
                    "common_valid_post_layernorm_cls_recovery_debiased_cka_12x12",
                    4 + corruption_index,
                )
                for corruption_index, corruption in enumerate(("mask", "blur", "noise"))
            ),
        ),
        (
            "patching.block_update.common_valid_post_layernorm_patch_recovery_debiased_cka_12x12",
            tuple(
                (
                    corruption,
                    f"patching.{corruption}.block_update",
                    "common_valid_post_layernorm_patch_recovery_debiased_cka_12x12",
                    7 + corruption_index,
                )
                for corruption_index, corruption in enumerate(("mask", "blur", "noise"))
            ),
        ),
        (
            "jacobian.input_jvp.input_to_block_update_cls_debiased_cka_12x12",
            (("input_jvp", "jacobian.input_jvp", "input_to_block_update_cls_debiased_cka_12x12", 10),),
        ),
        (
            "jacobian.input_jvp.input_to_block_update_patch_debiased_cka_12x12",
            (("input_jvp", "jacobian.input_jvp", "input_to_block_update_patch_debiased_cka_12x12", 11),),
        ),
    ]

    output: dict[str, Any] = {}
    for condition_index, condition in enumerate(CONDITION_ORDER):
        output[condition] = {}
        task = PAIR_BY_CONDITION[condition].task_key
        task_output: dict[str, Any] = {}
        for metric_name, variants in specs:
            variant_records: dict[str, Any] = {}
            for variant_name, module_suffix, matrix_key, seed_offset in variants:
                variant_records[variant_name] = _primary_matrix_record(
                    module_results,
                    module_status,
                    condition=condition,
                    task=task,
                    module_suffix=module_suffix,
                    matrix_key=matrix_key,
                    global_permutations=int(global_permutations),
                    depth_band_permutations=int(depth_band_permutations),
                    seed=(
                        int(seed)
                        + 30000 * condition_index
                        + int(seed_offset)
                    ),
                    reportable_modules=reportable_modules,
                )
            if len(variants) == 1:
                task_output[metric_name] = next(iter(variant_records.values()))
                continue

            usable_count = sum(
                str(record.get("statistics_status", "")) in {"available", "partial"}
                for record in variant_records.values()
            )
            available_count = sum(
                str(record.get("statistics_status", "")) == "available"
                for record in variant_records.values()
            )
            if available_count == len(variant_records):
                family_status = "available"
            elif usable_count > 0:
                family_status = "partial"
            else:
                family_status = "unavailable_or_inconclusive"
            task_output[metric_name] = {
                "module_status": "family",
                "statistics_status": family_status,
                "family_available_variants": int(available_count),
                "family_usable_variants": int(usable_count),
                "family_total_variants": int(len(variant_records)),
                "variant_order": [name for name, _module, _matrix, _seed_offset in variants],
                "family_members": variant_records,
                "family_contract": (
                    "mask_blur_noise_are_reported_as_separate_variants_without_cross_corruption_numeric_aggregation"
                ),
            }
        output[condition][task] = task_output
    return output

def _supporting_metric_summary(
    module_results: Mapping[str, Any],
    module_status: Mapping[str, Any],
    *,
    global_permutations: int = 5000,
    depth_band_permutations: int = 13824,
    seed: int,
    reportable_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose supporting native-update and internal-VJP evidence in Summary.

    These metrics do not become primary merely by being visible. The compact
    records include the raw 12x12 matrix, validity mask, same-index summary,
    and exact depth-band permutation result so the Summary can be reviewed
    without opening the raw shard archive.
    """

    conditions = CONDITION_ORDER

    def record_matrix(
        matrix: Any,
        mask: Any,
        *,
        local_seed: int,
        objective: str = "maximize",
        prepared: Any = None,
    ) -> dict[str, Any]:
        if not _is_square_numeric_matrix(matrix):
            return {"status": "unavailable_or_inconclusive"}
        if isinstance(prepared, Mapping) and _is_square_numeric_matrix(
            prepared.get("matrix")
        ):
            return {
                "status": "completed",
                "objective": str(prepared.get("objective", objective)),
                "matrix": prepared["matrix"],
                "validity_mask": prepared.get("validity_mask", mask),
                "matrix_summary": dict(prepared.get("summary", {}) or {}),
                "permutation": dict(prepared.get("permutation", {}) or {}),
                "statistics_reused_from_raw_core_report": True,
            }
        summary = matrix_summary(matrix, objective=objective, valid_mask=mask)
        return {
            "status": "completed",
            "objective": objective,
            "matrix": matrix,
            "validity_mask": mask,
            "matrix_summary": summary,
            "permutation": block_axis_permutation_test(
                matrix,
                global_permutations=int(global_permutations),
                depth_band_permutations=int(depth_band_permutations),
                seed=int(local_seed),
                objective=objective,
                valid_mask=mask,
            ),
            "statistics_reused_from_raw_core_report": False,
        }

    output: dict[str, Any] = {}
    for condition_index, condition in enumerate(conditions):
        output[condition] = {}
        for task_index, task in enumerate((PAIR_BY_CONDITION[condition].task_key,)):
            item: dict[str, Any] = {}
            block_name = f"{condition}.{task}.core.block_update"
            block_canonical = _canonical_module_name(
                block_name, module_results, module_status
            )
            block_payload = (
                module_results.get(block_canonical, {}) if block_canonical else {}
            )
            block_record: dict[str, Any] = {
                "module_status": str(
                    module_status.get(block_canonical or block_name, {}).get(
                        "status", "missing"
                    )
                ),
                "canonical_module": block_canonical,
            }
            prepared_block = (
                reportable_modules.get(block_canonical, {})
                if isinstance(reportable_modules, Mapping) and block_canonical
                else {}
            )
            if isinstance(block_payload, Mapping):
                masks = block_payload.get("validity_masks", {})
                for metric_index, key in enumerate(
                    (
                        "block_update_cls_debiased_cka_12x12",
                        "block_update_patch_debiased_cka_12x12",
                    )
                ):
                    block_record[key] = record_matrix(
                        block_payload.get(key),
                        masks.get(key) if isinstance(masks, Mapping) else None,
                        local_seed=int(seed)
                        + 10000 * condition_index
                        + 1000 * task_index
                        + metric_index,
                        prepared=(
                            prepared_block.get(key)
                            if isinstance(prepared_block, Mapping)
                            else None
                        ),
                    )
            item["native_block_update"] = block_record

            vjp_name = f"{condition}.{task}.core.jacobian.internal_vjp"
            vjp_canonical = _canonical_module_name(
                vjp_name, module_results, module_status
            )
            vjp_payload = module_results.get(vjp_canonical, {}) if vjp_canonical else {}
            vjp_status_payload = module_status.get(vjp_canonical or vjp_name, {})
            vjp_record: dict[str, Any] = {
                "module_status": str(vjp_status_payload.get("status", "missing")),
                "primary_measurement_status": dict(
                    vjp_status_payload.get("primary_measurement_status", {}) or {}
                ),
                "canonical_module": vjp_canonical,
                "paths": {},
            }
            paths = vjp_payload.get("paths", {}) if isinstance(vjp_payload, Mapping) else {}
            prepared_vjp = (
                reportable_modules.get(vjp_canonical, {})
                if isinstance(reportable_modules, Mapping) and vjp_canonical
                else {}
            )
            prepared_paths = (
                prepared_vjp.get("paths", {})
                if isinstance(prepared_vjp, Mapping)
                else {}
            )
            if isinstance(paths, Mapping):
                for path_index, (path_name, path_payload) in enumerate(sorted(paths.items())):
                    if not isinstance(path_payload, Mapping):
                        continue
                    key = "sample_gram_debiased_cka_12x12"
                    masks = path_payload.get("validity_masks", {})
                    vjp_record["paths"][path_name] = record_matrix(
                        path_payload.get(key),
                        masks.get(key) if isinstance(masks, Mapping) else None,
                        local_seed=int(seed)
                        + 50000
                        + 10000 * condition_index
                        + 1000 * task_index
                        + path_index,
                        prepared=(
                            prepared_paths.get(path_name, {}).get(key)
                            if isinstance(prepared_paths, Mapping)
                            and isinstance(prepared_paths.get(path_name, {}), Mapping)
                            else None
                        ),
                    )
            item["internal_vjp"] = vjp_record
            output[condition][task] = item
    return output

def _jacobian_rank32_summary(
    module_results: Mapping[str, Any],
    module_status: Mapping[str, Any],
    *,
    bootstrap_iterations: int,
    global_permutations: int = 5000,
    depth_band_permutations: int = 13824,
    seed: int,
    reportable_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose the actual rank-32 randomized Jacobian results in Summary."""

    conditions = CONDITION_ORDER
    views = ("block_update_cls", "block_update_patch")
    matrix_keys = (
        "singular_spectrum_cosine_12x12",
        "input_singular_subspace_overlap_12x12",
        "output_singular_subspace_overlap_12x12",
        "low_rank_operator_cosine_12x12",
    )

    def payload_for(condition: str, task: str) -> tuple[str | None, Mapping[str, Any], str]:
        name = f"{condition}.{task}.core.jacobian.input_jvp"
        canonical = _canonical_module_name(name, module_results, module_status)
        if canonical is None and name in module_results:
            canonical = name
        payload = module_results.get(canonical, {}) if canonical else {}
        status = str(module_status.get(canonical or name, {}).get("status", "missing"))
        return canonical, payload if isinstance(payload, Mapping) else {}, status

    output: dict[str, Any] = {"conditions": {}, "condition_comparisons": {}}
    for condition_index, condition in enumerate(conditions):
        output["conditions"][condition] = {}
        for task_index, task in enumerate((PAIR_BY_CONDITION[condition].task_key,)):
            canonical, payload, status = payload_for(condition, task)
            rank_payload = payload.get("rank32_sample_mean_jacobian", {})
            prepared = (
                reportable_modules.get(canonical, {})
                if isinstance(reportable_modules, Mapping) and canonical
                else {}
            )
            prepared_rank = (
                prepared.get("rank32_sample_mean_jacobian", {})
                if isinstance(prepared, Mapping)
                else {}
            )
            status_payload = module_status.get(canonical or f"{condition}.{task}.core.jacobian.input_jvp", {})
            task_record: dict[str, Any] = {
                "module_status": status,
                "primary_measurement_status": dict(
                    status_payload.get("primary_measurement_status", {}) or {}
                ),
                "canonical_module": canonical,
                "rank": rank_payload.get("rank", 32) if isinstance(rank_payload, Mapping) else 32,
                "range_probe_count": rank_payload.get("range_probe_count", 40) if isinstance(rank_payload, Mapping) else 40,
                "views": {},
            }
            for view_index, view in enumerate(views):
                view_payload = rank_payload.get(view, {}) if isinstance(rank_payload, Mapping) else {}
                prepared_view = prepared_rank.get(view, {}) if isinstance(prepared_rank, Mapping) else {}
                masks = view_payload.get("validity_masks", {}) if isinstance(view_payload, Mapping) else {}
                view_record: dict[str, Any] = {
                    "measurement_status": view_payload.get("measurement_status", "unavailable"),
                }
                for metric_index, key in enumerate(matrix_keys):
                    prepared_record = prepared_view.get(key) if isinstance(prepared_view, Mapping) else None
                    if isinstance(prepared_record, Mapping) and _is_square_numeric_matrix(prepared_record.get("matrix")):
                        view_record[key] = {
                            "status": "completed",
                            "matrix": prepared_record.get("matrix"),
                            "validity_mask": prepared_record.get("validity_mask"),
                            "matrix_summary": dict(prepared_record.get("summary", {}) or {}),
                            "permutation": dict(prepared_record.get("permutation", {}) or {}),
                            "statistics_reused_from_raw_core_report": True,
                        }
                    else:
                        matrix = view_payload.get(key) if isinstance(view_payload, Mapping) else None
                        mask = masks.get(key) if isinstance(masks, Mapping) else None
                        if _is_square_numeric_matrix(matrix):
                            try:
                                view_record[key] = {
                                    "status": "completed",
                                    "matrix": matrix,
                                    "validity_mask": mask,
                                    "matrix_summary": matrix_summary(matrix, valid_mask=mask),
                                    "permutation": block_axis_permutation_test(
                                        matrix,
                                        global_permutations=int(global_permutations),
                                        depth_band_permutations=int(depth_band_permutations),
                                        seed=int(seed) + 10000 * condition_index + 1000 * task_index + 100 * view_index + metric_index,
                                        valid_mask=mask,
                                    ),
                                    "statistics_reused_from_raw_core_report": False,
                                }
                            except Exception as error:
                                view_record[key] = {
                                    "status": "inconclusive",
                                    "matrix": matrix,
                                    "validity_mask": mask,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                        else:
                            view_record[key] = {"status": "unavailable_or_inconclusive"}
                for key in (
                    "left_leading_singular_values_by_block",
                    "right_leading_singular_values_by_block",
                    "left_rank_used_by_block",
                    "right_rank_used_by_block",
                    "left_descriptor_status_by_block",
                    "right_descriptor_status_by_block",
                    "left_holdout_relative_residual_by_block",
                    "right_holdout_relative_residual_by_block",
                    "left_valid_by_block",
                    "right_valid_by_block",
                ):
                    if isinstance(view_payload, Mapping) and key in view_payload:
                        view_record[key] = view_payload[key]
                task_record["views"][view] = view_record
            output["conditions"][condition][task] = task_record

    for comparison_index, (comparison, left_condition, right_condition) in enumerate(SUMMARY_COMPARISONS):
        output["condition_comparisons"][comparison] = {}
        task = PAIR_BY_CONDITION[left_condition].task_key
        if PAIR_BY_CONDITION[right_condition].task_key != task:
            raise ValueError(f"Comparison task mismatch: {comparison}")
        for task_index, task in enumerate((task,)):
            _l_name, left_payload, left_status = payload_for(left_condition, task)
            _r_name, right_payload, right_status = payload_for(right_condition, task)
            task_comparison: dict[str, Any] = {}
            if not (_reportable_status(left_status) and _reportable_status(right_status)):
                task_comparison["status"] = "excluded_due_to_noncompleted_module"
                task_comparison["left_status"] = left_status
                task_comparison["right_status"] = right_status
                output["condition_comparisons"][comparison][task] = task_comparison
                continue
            left_rank = left_payload.get("rank32_sample_mean_jacobian", {})
            right_rank = right_payload.get("rank32_sample_mean_jacobian", {})
            for view_index, view in enumerate(views):
                left_view = left_rank.get(view, {}) if isinstance(left_rank, Mapping) else {}
                right_view = right_rank.get(view, {}) if isinstance(right_rank, Mapping) else {}
                left_masks = left_view.get("validity_masks", {}) if isinstance(left_view, Mapping) else {}
                right_masks = right_view.get("validity_masks", {}) if isinstance(right_view, Mapping) else {}
                view_comparison: dict[str, Any] = {}
                for metric_index, key in enumerate(matrix_keys):
                    view_comparison[key] = _paired_matrix_difference_reports(
                        left_view.get(key) if isinstance(left_view, Mapping) else None,
                        right_view.get(key) if isinstance(right_view, Mapping) else None,
                        bootstrap_iterations=int(bootstrap_iterations),
                        seed=int(seed) + 100000 * comparison_index + 10000 * task_index + 1000 * view_index + metric_index,
                        path=("rank32_sample_mean_jacobian", view, key),
                        left_mask=(left_masks.get(key) if isinstance(left_masks, Mapping) else None),
                        right_mask=(right_masks.get(key) if isinstance(right_masks, Mapping) else None),
                    )
                task_comparison[view] = view_comparison
            output["condition_comparisons"][comparison][task] = task_comparison
    output["contract"] = {
        "operator": "fixed_sample_mean_CLS_and_patch_block_update_Jacobian",
        "algorithm": "40_probe_rank32_randomized_SVD_without_materializing_full_Jacobian",
        "numerical_rank": (
            "below_tolerance_components_are_excluded_from_subspaces_rank_zero_is_recorded_as_inconclusive_for_similarity_and_positive_low_rank_uses_its_actual_numerical_rank"
        ),
        "range_audit": (
            "first_32_discovery_last_8_holdout_residual_is_an_advisory_approximation_quality_record_final_fit_uses_all_40"
        ),
        "degenerate_similarity_contract": (
            "rank_zero_pairs_have_no_numeric_alignment_score_and_are_excluded_by_explicit_validity_masks_positive_rank_uses_standard_similarity"
        ),
    }
    return output

def _control_comparisons(
    module_results: Mapping[str, Any],
    module_status: Mapping[str, Any],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Paired final-paper comparisons within a shared task family.

    Same-task compares DiR dictionary reuse with an independently trained Dense
    target. Different-task compares Dictionary-Fixed against both Dictionary-Trainable (the
    direct preservation control) and Dense Full-Transfer.
    """

    output: dict[str, Any] = {}
    base_module_names = sorted(module_results)
    for comparison_index, (label, left_prefix, right_prefix) in enumerate(SUMMARY_COMPARISONS):
        output[label] = {}
        left_task = PAIR_BY_CONDITION[left_prefix].task_key
        right_task = PAIR_BY_CONDITION[right_prefix].task_key
        if left_task != right_task:
            raise ValueError(f"Comparison task mismatch: {label}")
        task_token = f".{left_task}."
        local_index = 0
        for left_name in base_module_names:
            if not left_name.startswith(left_prefix + task_token):
                continue
            right_name = right_prefix + left_name[len(left_prefix):]
            if right_name not in module_results:
                continue
            canonical_left = _canonical_module_name(left_name, module_results, module_status)
            canonical_right = _canonical_module_name(right_name, module_results, module_status)
            output_name = left_name + "__minus__" + right_name
            if canonical_left is None or canonical_right is None:
                output[label][output_name] = {
                    "status": "excluded_due_to_noncompleted_module",
                    "left_status": module_status.get(left_name, {}),
                    "right_status": module_status.get(right_name, {}),
                    "left_canonical_module": canonical_left,
                    "right_canonical_module": canonical_right,
                }
                continue
            report = _paired_matrix_difference_reports(
                module_results[canonical_left],
                module_results[canonical_right],
                bootstrap_iterations=int(bootstrap_iterations),
                seed=int(seed) + 100000 * comparison_index + local_index,
            )
            output[label][output_name] = {
                "status": "completed",
                "left_canonical_module": canonical_left,
                "right_canonical_module": canonical_right,
                "difference_orientation": f"positive_favors_{left_prefix}",
                "report": report,
            }
            local_index += 1
    return output

def _support_commit_output_parity_summary(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    commit_epoch: int = 52,
    expected_sample_count: int = 128,
) -> dict[str, Any]:
    required_run_ids = {"dir_same_task", "dir_dictionary_fixed", "dir_dictionary_trainable"}
    candidates: dict[str, list[Mapping[str, Any]]] = {
        run_id: [] for run_id in required_run_ids
    }
    for row in training_rows:
        run_id = str(row.get("run_id", ""))
        if run_id not in required_run_ids:
            continue
        status = str(row.get("forward_support_commit_output_parity_status", ""))
        if status in {"passed", "failed"}:
            candidates[run_id].append(row)

    checked: dict[str, dict[str, Any]] = {}
    invalid_reasons: list[str] = []
    failed: list[str] = []
    finite_fields = (
        "forward_support_commit_output_parity_max_abs_logit_difference",
        "forward_support_commit_output_parity_relative_l2_difference",
        "forward_support_commit_output_parity_accuracy_abs_difference",
    )
    for run_id in sorted(required_run_ids):
        rows = candidates[run_id]
        if len(rows) != 1:
            invalid_reasons.append(
                f"{run_id}:expected_exactly_one_commit_parity_row_found_{len(rows)}"
            )
            continue
        row = rows[0]
        try:
            status = str(row.get("forward_support_commit_output_parity_status", ""))
            passed_value = row.get("forward_support_commit_output_parity_passed", False)
            passed_flag = passed_value is True or str(passed_value).strip().lower() == "true"
            epoch = int(float(row.get("epoch", 0) or 0))
            sample_count = int(float(row.get("forward_support_commit_output_parity_sample_count", 0) or 0))
            mismatch_count = int(float(row.get("forward_support_commit_output_parity_prediction_mismatch_count", 0) or 0))
            numeric = {field: float(row.get(field, float("nan"))) for field in finite_fields}
        except (TypeError, ValueError, OverflowError) as exc:
            invalid_reasons.append(f"{run_id}:unparseable_parity_row:{type(exc).__name__}")
            continue
        if epoch != int(commit_epoch):
            invalid_reasons.append(f"{run_id}:epoch_{epoch}_expected_{commit_epoch}")
        if sample_count != int(expected_sample_count):
            invalid_reasons.append(
                f"{run_id}:sample_count_{sample_count}_expected_{expected_sample_count}"
            )
        if mismatch_count < 0 or mismatch_count > sample_count:
            invalid_reasons.append(f"{run_id}:invalid_prediction_mismatch_count")
        if not all(np.isfinite(value) for value in numeric.values()):
            invalid_reasons.append(f"{run_id}:nonfinite_parity_metric")
        elif any(value < 0.0 for value in numeric.values()):
            invalid_reasons.append(f"{run_id}:negative_parity_metric")
        if (status == "passed") != bool(passed_flag):
            invalid_reasons.append(f"{run_id}:status_passed_flag_mismatch")
        checked[run_id] = {
            "status": status,
            "passed": bool(passed_flag),
            "epoch": epoch,
            "sample_count": sample_count,
            "max_abs_logit_difference": numeric[finite_fields[0]],
            "relative_l2_difference": numeric[finite_fields[1]],
            "prediction_mismatch_count": mismatch_count,
            "accuracy_abs_difference": numeric[finite_fields[2]],
        }
        if status != "passed" or not passed_flag:
            failed.append(run_id)

    missing = sorted(required_run_ids - set(checked))
    passed = not missing and not failed and not invalid_reasons
    return {
        "status": "passed" if passed else "review_required",
        "passed": bool(passed),
        "commit_epoch": int(commit_epoch),
        "expected_sample_count": int(expected_sample_count),
        "required_run_ids": sorted(required_run_ids),
        "missing_run_ids": missing,
        "failed_run_ids": sorted(failed),
        "invalid_reasons": invalid_reasons,
        "checked_runs": checked,
        "contract": "target_epoch52_hard_support_commit_must_have_one_finite_exact_sample_count_row_and_preserve_logits_predictions_and_accuracy_within_predeclared_tolerances",
    }
