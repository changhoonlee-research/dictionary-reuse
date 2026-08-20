"""Block-ablation functional-correspondence measurements."""

from __future__ import annotations

from contextlib import contextmanager

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import Intervention, forward_from_block_input_with_interventions, forward_with_capture_and_interventions
from .causal_common import _causal_point_exception_result
from .causal_common import _normalize_causal_intervention_points, _response_signature, _structural_response_profiles
from .representation_cache import (
    _estimate_exact_causal_raw_cache_bytes,
    _failsoft_temporary_directory,
    _feature_rms_and_variation_from_gram,
    _gram_from_feature_chunks,
    _pairwise_biased_gram_cka_matrix,
    _pairwise_gram_cka_matrix,
    _save_feature_chunks,
    _select_exact_causal_cache_backend,
    _streaming_cached_paired_output_metric_matrices,
    combined_signal_variation_validity,
    outer_validity_mask,
)
from .representation_similarity import _feature_view, pairwise_biased_cka_matrix, pairwise_cka_matrix

def _finalize_ablation_response_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    intervention_point: str,
    depth: int,
    cache_group_size: int,
    same_head: bool,
    label_semantics_valid: bool,
    minimum_signal_rms_absolute: float,
    minimum_signal_rms_relative_to_median: float,
    cleanup_status: dict[str, Any],
) -> dict[str, Any]:
    feature_specs = {
        "post_layernorm_full_delta": "post_layernorm_full",
        "post_layernorm_cls_delta": "post_layernorm_cls",
        "post_layernorm_patch_delta": "post_layernorm_patch",
        "pre_layernorm_full_delta": "pre_layernorm_full",
        "pre_layernorm_cls_delta": "pre_layernorm_cls",
        "pre_layernorm_patch_delta": "pre_layernorm_patch",
    }
    if same_head:
        feature_specs["logit_delta"] = "logits"
    result: dict[str, Any] = {
        "measurement_status": "completed",
        "intervention_point": intervention_point,
        "cka_contract": "U_centered_debiased_primary_biased_auxiliary",
        "paired_output_contract": "signed_cosine_normalized_l2_and_symmetric_norm_ratio_on_corresponding_samples",
        "delta_contract": "intervened_final_state_minus_clean_final_state_fixed_dimension_per_sample",
        "token_contract": "post_layernorm_CLS_and_patch_are_primary_pre_layernorm_and_full_token_are_auxiliary",
        "token_stage_contract": "final_pre_classifier_layernorm_output",
        "memory_policy": "bounded_block_group_exact_gram_once_adaptive_exact_RAM_or_workdir_chunk_cache_single_pass_all_block_pairs_full_token_derived_from_cls_plus_patch",
        "causal_cache_backend": str(cleanup_status.get("cache_backend", "disk")),
        "causal_cache_plan": dict(cleanup_status.get("cache_plan", {}) or {}),
        "cache_block_group_size": int(cache_group_size),
        "capture_execution": {
            "left_baseline_forward_count": int(left["capture_forward_count"]),
            "right_baseline_forward_count": int(right["capture_forward_count"]),
            "left_intervention_forward_count": int(left["intervention_forward_count"]),
            "right_intervention_forward_count": int(right["intervention_forward_count"]),
            "baseline_shared_across_intervention_points": bool(
                left.get("baseline_shared_across_intervention_points", False)
                and right.get("baseline_shared_across_intervention_points", False)
            ),
        },
        "task_semantics_contract": {
            "shared_head_logit_comparison": bool(same_head),
            "native_label_margin_comparison": bool(label_semantics_valid),
            "rule": "logits_need_shared_head_true_class_margin_needs_current_dataset_native_to_that_head",
        },
    }
    auxiliary_biased: dict[str, Any] = {}
    paired_metrics = _streaming_cached_paired_output_metric_matrices(
        left["references"],
        right["references"],
        output_components={
            "post_layernorm_full_delta": ("post_layernorm_cls", "post_layernorm_patch"),
            "post_layernorm_cls_delta": ("post_layernorm_cls",),
            "post_layernorm_patch_delta": ("post_layernorm_patch",),
            "pre_layernorm_full_delta": ("pre_layernorm_cls", "pre_layernorm_patch"),
            "pre_layernorm_cls_delta": ("pre_layernorm_cls",),
            "pre_layernorm_patch_delta": ("pre_layernorm_patch",),
            **({"logit_delta": ("logits",)} if same_head else {}),
        },
    )
    validity_masks: dict[str, Any] = {}
    low_signal: dict[str, Any] = {}
    for output_name, source_key in feature_specs.items():
        matrix_key = f"{output_name}_debiased_cka_12x12"
        matrix = _pairwise_gram_cka_matrix(
            left["grams"][source_key],
            right["grams"][source_key],
            invalid_as_nan=True,
        )
        result[matrix_key] = matrix
        auxiliary_biased[f"{output_name}_biased_cka_12x12"] = (
            _pairwise_biased_gram_cka_matrix(
                left["grams"][source_key], right["grams"][source_key]
            )
        )
        left_validity = combined_signal_variation_validity(
            left["signal"][source_key],
            left["variation"][source_key],
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        right_validity = combined_signal_variation_validity(
            right["signal"][source_key],
            right["variation"][source_key],
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        signal_mask = np.asarray(
            outer_validity_mask(
                left_validity["valid_by_block"], right_validity["valid_by_block"]
            ),
            dtype=bool,
        )
        finite_mask = np.isfinite(np.asarray(matrix, dtype=np.float64))
        validity_masks[matrix_key] = (signal_mask & finite_mask).tolist()
        low_signal[output_name] = {"left": left_validity, "right": right_validity}

    appended_task_columns = int(label_semantics_valid) + int(same_head)
    left_structural_profiles = _structural_response_profiles(
        left["scalar_profiles"], appended_task_columns=appended_task_columns
    )
    right_structural_profiles = _structural_response_profiles(
        right["scalar_profiles"], appended_task_columns=appended_task_columns
    )
    result["downstream_rms_profile_debiased_cka_12x12"] = pairwise_cka_matrix(
        left_structural_profiles, right_structural_profiles
    )
    if appended_task_columns:
        result["task_metric_augmented_response_debiased_cka_12x12"] = pairwise_cka_matrix(
            left["scalar_profiles"], right["scalar_profiles"]
        )
    else:
        result["task_metric_augmented_response_status"] = (
            "not_applicable_no_shared_head_or_native_label_semantics"
        )
    auxiliary_biased["downstream_rms_profile_biased_cka_12x12"] = pairwise_biased_cka_matrix(
        left_structural_profiles, right_structural_profiles
    )
    result["auxiliary_biased_cka"] = auxiliary_biased
    result["paired_output_metrics"] = paired_metrics
    result["validity_masks"] = validity_masks
    result["low_signal"] = low_signal
    result["primary_metrics"] = [
        "post_layernorm_cls_delta_debiased_cka_12x12",
        "post_layernorm_patch_delta_debiased_cka_12x12",
    ]
    result["full_token_role"] = "auxiliary_only"
    result["scalar_profile_role"] = "secondary_depth_length_dependent_diagnostic_only"
    result["structural_scalar_profile_column_count_by_block"] = [
        int(value.shape[1]) for value in left_structural_profiles
    ]
    signature_columns = [
        "downstream_block_update_rms",
        "post_layernorm_full_rms",
        "post_layernorm_cls_rms",
        "post_layernorm_patch_rms",
    ]
    if label_semantics_valid:
        signature_columns.append("native_true_class_margin_change")
    if same_head:
        signature_columns.append("shared_head_prediction_index_flip")
    result["signature_columns"] = signature_columns
    preview_count = min(8, int(left["scalar_profiles"][0].shape[0])) if depth else 0
    result["signature_sample_count"] = int(left["scalar_profiles"][0].shape[0]) if depth else 0
    result["signature_storage_policy"] = (
        "primary_gram_matrices_plus_temporary_disk_vectors_and_first_8_scalar_profile_audit_only"
    )
    result["left_signature_preview"] = [
        value[:preview_count].tolist() for value in left["scalar_profiles"]
    ]
    result["right_signature_preview"] = [
        value[:preview_count].tolist() for value in right["scalar_profiles"]
    ]
    result["cache_cleanup"] = cleanup_status
    return result



class _AblationResponseRuntime:
    """State shared by one exact block-ablation measurement pass."""

    def __init__(
        self,
        *,
        device,
        points,
        base_feature_keys,
        same_head,
        depth,
        final_output_key,
        cache_group_size,
        batches,
        label_semantics_valid,
        temporary_root,
        cache_backend,
        cache_tracker,
    ) -> None:
        self.device = device
        self.points = points
        self.base_feature_keys = base_feature_keys
        self.same_head = same_head
        self.depth = depth
        self.final_output_key = final_output_key
        self.cache_group_size = cache_group_size
        self.batches = batches
        self.label_semantics_valid = label_semantics_valid
        self.temporary_root = temporary_root
        self.cache_backend = cache_backend
        self.cache_tracker = cache_tracker

    def responses(self, model: nn.Module, *, side: str) -> dict[str, dict[str, Any]]:
        model.eval().to(self.device)
        payloads: dict[str, dict[str, Any]] = {}
        point_errors: dict[str, list[dict[str, Any]]] = {point: [] for point in self.points}
        for point in self.points:
            keys = list(self.base_feature_keys) + (["logits"] if self.same_head else [])
            payloads[point] = {
                "grams": {key: [None] * self.depth for key in keys},
                "signal": {key: [None] * self.depth for key in keys},
                "variation": {key: [None] * self.depth for key in keys},
                "references": {key: [None] * self.depth for key in keys},
                "feature_dimensions": {key: [None] * self.depth for key in keys},
                "scalar_profiles": [None] * self.depth,
                "intervention_forward_count": 0,
            }
        response_capture_points = [
            "pre_classifier",
            self.final_output_key,
            *[f"block_{i:02d}_update" for i in range(self.depth)],
        ]
        capture_forward_count = 0
        with torch.no_grad():
            for group_start in range(0, self.depth, self.cache_group_size):
                block_indices = list(
                    range(group_start, min(self.depth, group_start + self.cache_group_size))
                )
                feature_chunks = {
                    point: {
                        key: {index: [] for index in block_indices}
                        for key in payloads[point]["references"]
                    }
                    for point in self.points
                }
                scalar_chunks = {
                    point: {index: [] for index in block_indices} for point in self.points
                }
                baseline_capture_points = [
                    *response_capture_points,
                    *[f"block_{index:02d}_input" for index in block_indices],
                ]
                for images_cpu, labels_cpu, _ids in self.batches:
                    images = images_cpu.to(self.device)
                    labels = labels_cpu.to(self.device)
                    baseline_logits, baseline = forward_with_capture_and_interventions(
                        model, images, capture_points=baseline_capture_points
                    )
                    capture_forward_count += 1
                    for block_index in block_indices:
                        block_input = baseline[f"block_{block_index:02d}_input"]
                        for point in self.points:
                            if point_errors[point]:
                                continue
                            try:
                                changed_logits, changed = forward_from_block_input_with_interventions(
                                    model,
                                    block_input,
                                    start_block_index=block_index,
                                    capture_points=response_capture_points,
                                    interventions=[
                                        Intervention(block_index, point, zero=True)
                                    ],
                                )
                                payloads[point]["intervention_forward_count"] += 1
                                pre_delta = (
                                    changed[self.final_output_key].float()
                                    - baseline[self.final_output_key].float()
                                )
                                post_delta = (
                                    changed["pre_classifier"].float()
                                    - baseline["pre_classifier"].float()
                                )
                                feature_chunks[point]["post_layernorm_cls"][block_index].append(
                                    _feature_view(post_delta, "cls").cpu()
                                )
                                feature_chunks[point]["post_layernorm_patch"][block_index].append(
                                    _feature_view(post_delta, "patch").cpu()
                                )
                                feature_chunks[point]["pre_layernorm_cls"][block_index].append(
                                    _feature_view(pre_delta, "cls").cpu()
                                )
                                feature_chunks[point]["pre_layernorm_patch"][block_index].append(
                                    _feature_view(pre_delta, "patch").cpu()
                                )
                                if self.same_head:
                                    feature_chunks[point]["logits"][block_index].append(
                                        (changed_logits.float() - baseline_logits.float())
                                        .reshape(images.shape[0], -1)
                                        .cpu()
                                    )
                                scalar_chunks[point][block_index].append(
                                    _response_signature(
                                        baseline_logits,
                                        baseline,
                                        changed_logits,
                                        changed,
                                        labels,
                                        intervention_block=block_index,
                                        include_logit_comparison=bool(self.same_head),
                                        include_label_metrics=bool(self.label_semantics_valid),
                                    )
                                )
                                del changed, changed_logits, pre_delta, post_delta
                            except Exception as error:
                                point_errors[point].append(
                                    {
                                        "side": str(side),
                                        "stage": "intervention_capture",
                                        "block_index": int(block_index),
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                    }
                                )
                    del baseline, baseline_logits, images, labels

                for point in self.points:
                    if point_errors[point]:
                        continue
                    payload = payloads[point]
                    for block_index in block_indices:
                        payload["scalar_profiles"][block_index] = torch.cat(
                            scalar_chunks[point][block_index], dim=0
                        )
                        for key in payload["references"]:
                            chunks = feature_chunks[point][key][block_index]
                            gram = _gram_from_feature_chunks(chunks, device=self.device)
                            feature_dimension = int(
                                chunks[0].reshape(chunks[0].shape[0], -1).shape[1]
                            )
                            block_signal, block_variation = _feature_rms_and_variation_from_gram(
                                gram, feature_dimension=feature_dimension
                            )
                            payload["references"][key][block_index] = _save_feature_chunks(
                                chunks,
                                root=self.temporary_root,
                                stem=f"{side}_ablation_{point}_{key}_{block_index:02d}",
                                backend=self.cache_backend,
                                cache_tracker=self.cache_tracker,
                            )
                            payload["grams"][key][block_index] = gram
                            payload["signal"][key][block_index] = block_signal
                            payload["variation"][key][block_index] = block_variation
                            payload["feature_dimensions"][key][block_index] = feature_dimension
                    del feature_chunks[point], scalar_chunks[point]

        def require(values: Sequence[Any], *, label: str) -> list[Any]:
            if any(value is None for value in values):
                raise RuntimeError(f"DiR ablation cache incomplete: {label}")
            return [value for value in values if value is not None]

        for point in self.points:
            if point_errors[point]:
                payloads[point] = {
                    "point_errors": point_errors[point],
                    "capture_forward_count": int(capture_forward_count),
                    "intervention_forward_count": int(
                        payloads[point]["intervention_forward_count"]
                    ),
                    "baseline_shared_across_intervention_points": len(self.points) > 1,
                }
                continue
            payload = payloads[point]
            for key in list(payload["references"]):
                payload["references"][key] = require(
                    payload["references"][key], label=f"{point}/references/{key}"
                )
                payload["grams"][key] = require(
                    payload["grams"][key], label=f"{point}/grams/{key}"
                )
                payload["signal"][key] = require(
                    payload["signal"][key], label=f"{point}/signal/{key}"
                )
                payload["variation"][key] = require(
                    payload["variation"][key], label=f"{point}/variation/{key}"
                )
                payload["feature_dimensions"][key] = require(
                    payload["feature_dimensions"][key],
                    label=f"{point}/feature_dimensions/{key}",
                )
            payload["scalar_profiles"] = require(
                payload["scalar_profiles"], label=f"{point}/scalar_profiles"
            )
            for stage in ("post_layernorm", "pre_layernorm"):
                cls_key = f"{stage}_cls"
                patch_key = f"{stage}_patch"
                full_key = f"{stage}_full"
                payload["grams"][full_key] = [
                    cls_gram + patch_gram
                    for cls_gram, patch_gram in zip(
                        payload["grams"][cls_key], payload["grams"][patch_key]
                    )
                ]
                payload["feature_dimensions"][full_key] = [
                    int(cls_dimension) + int(patch_dimension)
                    for cls_dimension, patch_dimension in zip(
                        payload["feature_dimensions"][cls_key],
                        payload["feature_dimensions"][patch_key],
                    )
                ]
                full_moments = [
                    _feature_rms_and_variation_from_gram(
                        gram, feature_dimension=dimension
                    )
                    for gram, dimension in zip(
                        payload["grams"][full_key],
                        payload["feature_dimensions"][full_key],
                    )
                ]
                payload["signal"][full_key] = [value[0] for value in full_moments]
                payload["variation"][full_key] = [value[1] for value in full_moments]
            payload["capture_forward_count"] = int(capture_forward_count)
            payload["baseline_shared_across_intervention_points"] = len(self.points) > 1
        return payloads


def ablation_response_alignment_suite(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    intervention_points: Sequence[str],
    same_head: bool,
    label_semantics_valid: bool,
    minimum_signal_rms_absolute: float = 1e-8,
    minimum_signal_rms_relative_to_median: float = 0.05,
    cache_directory: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate several ablation points while sharing every clean baseline forward."""

    points = _normalize_causal_intervention_points(intervention_points)
    if label_semantics_valid and not same_head:
        raise ValueError("DiR label semantics cannot be valid without a shared head")
    depth = len(left_model.transformer_blocks)
    if depth != len(right_model.transformer_blocks):
        raise ValueError("DiR ablation comparison requires equal depth")
    final_output_key = f"block_{depth - 1:02d}_output"
    cache_group_size = 2
    base_feature_keys = (
        "post_layernorm_cls",
        "post_layernorm_patch",
        "pre_layernorm_cls",
        "pre_layernorm_patch",
    )
    estimated_cache_bytes = _estimate_exact_causal_raw_cache_bytes(
        left_model,
        batches,
        depth=depth,
        point_count=len(points),
        same_head=bool(same_head),
    )
    cache_plan = _select_exact_causal_cache_backend(
        estimated_bytes=estimated_cache_bytes
    )
    cache_backend = str(cache_plan["backend"])
    cache_tracker: dict[str, Any] = {
        "backend": cache_backend,
        "ram_budget_bytes": int(cache_plan.get("ram_budget_bytes", 0)),
        "ram_bytes_used": 0,
        "disk_bytes_written": 0,
        "ram_chunk_count": 0,
        "disk_chunk_count": 0,
    }
    cache_parent = Path(cache_directory).expanduser().resolve() if cache_directory is not None else None
    cleanup_status: dict[str, Any] = {}
    with _failsoft_temporary_directory(
        prefix="dir_ablation_suite_",
        parent=cache_parent,
        cleanup_status=cleanup_status,
    ) as temporary_root:
        cleanup_status["cache_backend"] = cache_backend
        cleanup_status["cache_plan"] = dict(cache_plan)
        cleanup_status["cache_runtime"] = cache_tracker


        runtime = _AblationResponseRuntime(
            device=device,
            points=points,
            base_feature_keys=base_feature_keys,
            same_head=same_head,
            depth=depth,
            final_output_key=final_output_key,
            cache_group_size=cache_group_size,
            batches=batches,
            label_semantics_valid=label_semantics_valid,
            temporary_root=temporary_root,
            cache_backend=cache_backend,
            cache_tracker=cache_tracker,
        )
        left_by_point = runtime.responses(left_model, side="left")
        right_by_point = runtime.responses(right_model, side="right")
        results: dict[str, dict[str, Any]] = {}
        for point in points:
            point_errors = [
                *left_by_point[point].get("point_errors", []),
                *right_by_point[point].get("point_errors", []),
            ]
            if point_errors:
                results[point] = _causal_point_exception_result(
                    intervention_point=point,
                    stage="intervention_capture",
                    errors=point_errors,
                    cleanup_status=cleanup_status,
                )
                continue
            try:
                results[point] = _finalize_ablation_response_pair(
                    left_by_point[point],
                    right_by_point[point],
                    intervention_point=point,
                    depth=depth,
                    cache_group_size=cache_group_size,
                    same_head=bool(same_head),
                    label_semantics_valid=bool(label_semantics_valid),
                    minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
                    minimum_signal_rms_relative_to_median=float(
                        minimum_signal_rms_relative_to_median
                    ),
                    cleanup_status=cleanup_status,
                )
            except Exception as error:
                results[point] = _causal_point_exception_result(
                    intervention_point=point,
                    stage="pair_finalization",
                    errors=[
                        {
                            "side": "paired",
                            "stage": "pair_finalization",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    ],
                    cleanup_status=cleanup_status,
                )
        return results


@contextmanager
def atom_output_mask(model: nn.Module, masks: Mapping[str, torch.Tensor]):
    saved: list[tuple[nn.Module, Any]] = []
    from ..model.dictionary_operator import iter_dictionary_layers

    for layer_name, layer in iter_dictionary_layers(model):
        previous = getattr(layer, "_intervention_output_atom_mask", None)
        saved.append((layer, previous))
        mask = masks[layer_name].to(device=layer.coefficient_magnitude.device, dtype=layer.coefficient_magnitude.dtype)
        layer._intervention_output_atom_mask = mask
    try:
        yield
    finally:
        for layer, previous in saved:
            if previous is None:
                delattr(layer, "_intervention_output_atom_mask")
            else:
                layer._intervention_output_atom_mask = previous
