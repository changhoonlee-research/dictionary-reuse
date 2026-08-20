"""Activation-patching and recovery correspondence measurements."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import Intervention, forward_from_block_input_with_interventions, forward_with_capture_and_interventions
from .causal_common import (
    _causal_point_exception_result,
    _normalize_causal_intervention_points,
    _true_class_margin,
)
from .corruption import _patching_corruption_validity_audit, apply_weak_corruption
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
    recovery_validity,
)
from .representation_similarity import _feature_view

def _finalize_patching_response_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    corruption: str,
    intervention_point: str,
    depth: int,
    cache_group_size: int,
    same_head: bool,
    label_semantics_valid: bool,
    minimum_common_valid_samples: int,
    minimum_block_recovery_fraction: float,
    minimum_median_recovery_fraction: float,
    minimum_positive_recovery_sample_fraction: float,
    minimum_signal_rms_absolute: float,
    minimum_signal_rms_relative_to_median: float,
    cleanup_status: dict[str, Any],
    external_common_valid_masks: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    left_audit = left["audits"]
    right_audit = right["audits"]
    left_valid = left["valid_masks"]
    right_valid = right["valid_masks"]
    feature_specs = {
        "post_layernorm_full_recovery": "post_layernorm_full",
        "post_layernorm_cls_recovery": "post_layernorm_cls",
        "post_layernorm_patch_recovery": "post_layernorm_patch",
        "pre_layernorm_full_recovery": "pre_layernorm_full",
        "pre_layernorm_cls_recovery": "pre_layernorm_cls",
        "pre_layernorm_patch_recovery": "pre_layernorm_patch",
    }
    if same_head:
        feature_specs["logit_recovery"] = "logits"
    common_valid_by_view: dict[str, torch.Tensor] = {}
    for key in feature_specs.values():
        mask = left_valid[key] & right_valid[key]
        if external_common_valid_masks is not None:
            if key not in external_common_valid_masks:
                raise ValueError(f"DiR family common-valid mask missing view: {key}")
            external_mask = external_common_valid_masks[key].detach().bool().cpu()
            if int(external_mask.numel()) != int(mask.numel()):
                raise ValueError(f"DiR family common-valid mask length mismatch: {key}")
            mask = mask & external_mask
        common_valid_by_view[key] = mask
    common_count_by_view = {
        key: int(mask.sum()) for key, mask in common_valid_by_view.items()
    }
    view_status: dict[str, str] = {}
    required_common_valid_samples_for_cka = max(4, int(minimum_common_valid_samples))
    for key in feature_specs.values():
        if not (
            bool(left_audit[key]["validity_passed"])
            and bool(right_audit[key]["validity_passed"])
        ):
            view_status[key] = "warning_invalid_corruption"
        elif common_count_by_view[key] < required_common_valid_samples_for_cka:
            view_status[key] = "inconclusive_common_valid_subset"
        else:
            view_status[key] = "completed"
    primary_view_keys = ("post_layernorm_cls", "post_layernorm_patch")
    primary_completed = [view_status[key] == "completed" for key in primary_view_keys]
    measurement_status = (
        "completed"
        if all(primary_completed)
        else "partial_primary_views"
        if any(primary_completed)
        else "inconclusive_no_primary_view_common_subset"
    )
    result: dict[str, Any] = {
        "measurement_status": measurement_status,
        "corruption": corruption,
        "intervention_point": intervention_point,
        "common_valid_sample_count": min(
            common_count_by_view[key] for key in primary_view_keys
        ),
        "common_valid_sample_count_by_view": common_count_by_view,
        "primary_view_status": {key: view_status[key] for key in primary_view_keys},
        "minimum_common_valid_samples": int(minimum_common_valid_samples),
        "minimum_common_valid_samples_for_debiased_cka": int(
            required_common_valid_samples_for_cka
        ),
        "minimum_block_recovery_fraction": float(minimum_block_recovery_fraction),
        "minimum_median_recovery_fraction": float(minimum_median_recovery_fraction),
        "minimum_positive_recovery_sample_fraction": float(
            minimum_positive_recovery_sample_fraction
        ),
        "left_validity": left_audit,
        "right_validity": right_audit,
        "recovery_vector_contract": "patched_minus_corrupted_compared_only_after_clean_target_recovery_audit",
        "clean_target_contract": "target_vector_is_clean_minus_corrupted_and_positive_distance_reduction_is_required",
        "sample_contract": (
            "CLS_patch_full_pre_and_logit_each_use_their_own_left_right_common_valid_mask"
            if external_common_valid_masks is None
            else "DiR_and_Dense_family_share_the_same_four_model_intersection_sample_mask_per_view"
        ),
        "family_common_valid_mask_applied": bool(external_common_valid_masks is not None),
        "family_common_valid_mask_sha256_by_view": (
            {
                key: hashlib.sha256(mask.detach().bool().cpu().numpy().tobytes()).hexdigest()
                for key, mask in sorted(common_valid_by_view.items())
            }
            if external_common_valid_masks is not None
            else {}
        ),
        "cka_contract": "U_centered_debiased_primary_biased_auxiliary",
        "token_contract": "post_layernorm_CLS_and_patch_are_primary_pre_layernorm_and_full_token_are_auxiliary",
        "token_stage_contract": "final_pre_classifier_layernorm_output",
        "memory_policy": "bounded_block_group_exact_component_gram_once_adaptive_exact_RAM_or_workdir_chunk_cache_single_read_all_block_pairs_common_subset",
        "causal_cache_backend": str(cleanup_status.get("cache_backend", "disk")),
        "causal_cache_plan": dict(cleanup_status.get("cache_plan", {}) or {}),
        "cache_block_group_size": int(cache_group_size),
        "capture_execution": {
            "left_corruption_audit_forward_count": int(left["audit_forward_count"]),
            "right_corruption_audit_forward_count": int(right["audit_forward_count"]),
            "left_corruption_audit_reused": bool(left.get("corruption_audit_reused", False)),
            "right_corruption_audit_reused": bool(right.get("corruption_audit_reused", False)),
            "left_grouped_baseline_forward_count": int(left["grouped_baseline_forward_count"]),
            "right_grouped_baseline_forward_count": int(right["grouped_baseline_forward_count"]),
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
    validity_masks: dict[str, Any] = {}
    low_signal: dict[str, Any] = {}
    actual_recovery: dict[str, Any] = {}
    recovery_offsets = {
        "post_layernorm_full": 0,
        "post_layernorm_cls": 4,
        "post_layernorm_patch": 8,
    }

    def gram_signal_variation_by_block(
        side_payload: Mapping[str, Any],
        component_keys: Sequence[str],
        sample_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
        grams: list[torch.Tensor] = []
        rms_values: list[float] = []
        variation_values: list[float] = []
        mask = sample_mask.bool().cpu()
        for block_index in range(depth):
            combined_gram: torch.Tensor | None = None
            total_feature_dimension = 0
            for key in component_keys:
                full_gram = side_payload["grams"][key][block_index]
                if int(mask.numel()) != int(full_gram.shape[0]):
                    raise ValueError("DiR patching common-valid mask length mismatch")
                component_gram = full_gram[mask][:, mask]
                combined_gram = (
                    component_gram if combined_gram is None else combined_gram + component_gram
                )
                total_feature_dimension += int(
                    side_payload["feature_dimensions"][key][block_index]
                )
            if combined_gram is None:
                raise RuntimeError("DiR patching Gram cache has no components")
            block_rms, block_variation = _feature_rms_and_variation_from_gram(
                combined_gram,
                feature_dimension=total_feature_dimension,
            )
            grams.append(combined_gram)
            rms_values.append(block_rms)
            variation_values.append(block_variation)
        return grams, rms_values, variation_values

    paired_output_components = {
        "post_layernorm_full_recovery": ("post_layernorm_cls", "post_layernorm_patch"),
        "post_layernorm_cls_recovery": ("post_layernorm_cls",),
        "post_layernorm_patch_recovery": ("post_layernorm_patch",),
        "pre_layernorm_full_recovery": ("pre_layernorm_cls", "pre_layernorm_patch"),
        "pre_layernorm_cls_recovery": ("pre_layernorm_cls",),
        "pre_layernorm_patch_recovery": ("pre_layernorm_patch",),
        **({"logit_recovery": ("logits",)} if same_head else {}),
    }
    completed_paired_components = {
        output_name: paired_output_components[output_name]
        for output_name, source_key in feature_specs.items()
        if view_status[source_key] == "completed"
    }
    paired_metrics = _streaming_cached_paired_output_metric_matrices(
        left["references"],
        right["references"],
        output_components=completed_paired_components,
        sample_masks={
            output_name: common_valid_by_view[source_key]
            for output_name, source_key in feature_specs.items()
            if output_name in completed_paired_components
        },
    )

    for output_name, source_key in feature_specs.items():
        matrix_key = f"common_valid_{output_name}_debiased_cka_12x12"
        common_valid = common_valid_by_view[source_key]
        if view_status[source_key] != "completed":
            result[matrix_key] = None
            validity_masks[matrix_key] = [[False] * depth for _ in range(depth)]
            low_signal[output_name] = {
                "status": view_status[source_key],
                "common_valid_sample_count": common_count_by_view[source_key],
            }
            continue
        component_keys = paired_output_components[output_name]
        left_grams, left_rms, left_variation = gram_signal_variation_by_block(
            left, component_keys, common_valid
        )
        right_grams, right_rms, right_variation = gram_signal_variation_by_block(
            right, component_keys, common_valid
        )
        matrix = _pairwise_gram_cka_matrix(left_grams, right_grams, invalid_as_nan=True)
        result[matrix_key] = matrix
        auxiliary_biased[f"common_valid_{output_name}_biased_cka_12x12"] = (
            _pairwise_biased_gram_cka_matrix(left_grams, right_grams)
        )
        left_signal = combined_signal_variation_validity(
            left_rms,
            left_variation,
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        right_signal = combined_signal_variation_validity(
            right_rms,
            right_variation,
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        if source_key in recovery_offsets:
            offset = recovery_offsets[source_key]
            left_fraction = [
                float(value[common_valid, offset + 1].mean()) for value in left["scalar"]
            ]
            right_fraction = [
                float(value[common_valid, offset + 1].mean()) for value in right["scalar"]
            ]
            left_median_fraction = [
                float(value[common_valid, offset + 1].median()) for value in left["scalar"]
            ]
            right_median_fraction = [
                float(value[common_valid, offset + 1].median()) for value in right["scalar"]
            ]
            left_positive_fraction = [
                float((value[common_valid, offset + 1] > 0.0).float().mean())
                for value in left["scalar"]
            ]
            right_positive_fraction = [
                float((value[common_valid, offset + 1] > 0.0).float().mean())
                for value in right["scalar"]
            ]
            left_projection = [
                float(value[common_valid, offset + 2].mean()) for value in left["scalar"]
            ]
            right_projection = [
                float(value[common_valid, offset + 2].mean()) for value in right["scalar"]
            ]
            left_direction = [
                float(value[common_valid, offset + 3].mean()) for value in left["scalar"]
            ]
            right_direction = [
                float(value[common_valid, offset + 3].mean()) for value in right["scalar"]
            ]
            left_recovery_valid = recovery_validity(
                left_signal["valid_by_block"],
                left_fraction,
                left_median_fraction,
                left_positive_fraction,
                left_projection,
                minimum_block_recovery_fraction=float(minimum_block_recovery_fraction),
                minimum_median_recovery_fraction=float(minimum_median_recovery_fraction),
                minimum_positive_recovery_sample_fraction=float(
                    minimum_positive_recovery_sample_fraction
                ),
            )
            right_recovery_valid = recovery_validity(
                right_signal["valid_by_block"],
                right_fraction,
                right_median_fraction,
                right_positive_fraction,
                right_projection,
                minimum_block_recovery_fraction=float(minimum_block_recovery_fraction),
                minimum_median_recovery_fraction=float(minimum_median_recovery_fraction),
                minimum_positive_recovery_sample_fraction=float(
                    minimum_positive_recovery_sample_fraction
                ),
            )
            actual_recovery[output_name] = {
                "left_mean_clean_distance_reduction_fraction": left_fraction,
                "right_mean_clean_distance_reduction_fraction": right_fraction,
                "left_median_clean_distance_reduction_fraction": left_median_fraction,
                "right_median_clean_distance_reduction_fraction": right_median_fraction,
                "left_positive_recovery_sample_fraction": left_positive_fraction,
                "right_positive_recovery_sample_fraction": right_positive_fraction,
                "left_mean_clean_target_projection_fraction": left_projection,
                "right_mean_clean_target_projection_fraction": right_projection,
                "left_mean_clean_target_direction_cosine": left_direction,
                "right_mean_clean_target_direction_cosine": right_direction,
                "left_valid_by_block": left_recovery_valid,
                "right_valid_by_block": right_recovery_valid,
                "common_valid_sample_count": common_count_by_view[source_key],
            }
            signal_mask = np.asarray(
                outer_validity_mask(left_recovery_valid, right_recovery_valid),
                dtype=bool,
            )
        else:
            signal_mask = np.asarray(
                outer_validity_mask(
                    left_signal["valid_by_block"], right_signal["valid_by_block"]
                ),
                dtype=bool,
            )
        finite_mask = np.isfinite(np.asarray(matrix, dtype=np.float64))
        validity_masks[matrix_key] = (signal_mask & finite_mask).tolist()
        low_signal[output_name] = {"left": left_signal, "right": right_signal}

    result["auxiliary_biased_cka"] = auxiliary_biased
    result["paired_output_metrics"] = paired_metrics
    result["validity_masks"] = validity_masks
    result["low_signal"] = low_signal
    result["actual_clean_recovery"] = actual_recovery
    result["primary_metrics"] = [
        "common_valid_post_layernorm_cls_recovery_debiased_cka_12x12",
        "common_valid_post_layernorm_patch_recovery_debiased_cka_12x12",
    ]
    result["full_token_role"] = "auxiliary_only"
    signature_columns = [
        "post_layernorm_full_absolute_clean_distance_reduction",
        "post_layernorm_full_fractional_clean_distance_reduction",
        "post_layernorm_full_clean_target_projection_fraction",
        "post_layernorm_full_clean_target_direction_cosine",
        "post_layernorm_cls_absolute_clean_distance_reduction",
        "post_layernorm_cls_fractional_clean_distance_reduction",
        "post_layernorm_cls_clean_target_projection_fraction",
        "post_layernorm_cls_clean_target_direction_cosine",
        "post_layernorm_patch_absolute_clean_distance_reduction",
        "post_layernorm_patch_fractional_clean_distance_reduction",
        "post_layernorm_patch_clean_target_projection_fraction",
        "post_layernorm_patch_clean_target_direction_cosine",
    ]
    if same_head:
        signature_columns.extend(
            [
                "logit_absolute_clean_distance_reduction",
                "logit_fractional_clean_distance_reduction",
                "logit_clean_target_projection_fraction",
                "logit_clean_target_direction_cosine",
            ]
        )
    if label_semantics_valid:
        signature_columns.append("native_true_class_margin_recovery")
    if same_head:
        signature_columns.append("shared_head_prediction_flip_restored")
    signature_columns.extend(
        [
            "post_layernorm_full_model_specific_valid_flag",
            "post_layernorm_cls_model_specific_valid_flag",
            "post_layernorm_patch_model_specific_valid_flag",
        ]
    )
    result["signature_columns"] = signature_columns
    sample_count = int(next(iter(common_valid_by_view.values())).numel())
    preview_count = min(8, sample_count)
    result["signature_sample_count"] = sample_count
    result["signature_storage_policy"] = (
        "view_specific_common_valid_gram_matrices_workdir_chunk_cache_single_pass_paired_metrics_and_first_8_scalar_audit_only"
    )
    result["left_signature_preview"] = [
        value[:preview_count].tolist() for value in left["scalar"]
    ]
    result["right_signature_preview"] = [
        value[:preview_count].tolist() for value in right["scalar"]
    ]
    result["cache_cleanup"] = cleanup_status
    return result



class _PatchingRecoveryRuntime:
    """State shared by one corruption-specific activation-patching pass."""

    def __init__(
        self,
        *,
        device,
        batches,
        corruption,
        mean,
        std,
        same_head,
        minimum_relative_effect,
        minimum_prediction_retention,
        noise_sigma,
        noise_seed,
        blur_sigma,
        blur_kernel_size,
        blur_padding,
        mask_size,
        mask_positions,
        mask_fill,
        points,
        base_feature_keys,
        depth,
        cache_group_size,
        final_output_key,
        point_capture_name,
        label_semantics_valid,
        temporary_root,
        cache_backend,
        cache_tracker,
    ) -> None:
        self.device = device
        self.batches = batches
        self.corruption = corruption
        self.mean = mean
        self.std = std
        self.same_head = same_head
        self.minimum_relative_effect = minimum_relative_effect
        self.minimum_prediction_retention = minimum_prediction_retention
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size
        self.blur_padding = blur_padding
        self.mask_size = mask_size
        self.mask_positions = mask_positions
        self.mask_fill = mask_fill
        self.points = points
        self.base_feature_keys = base_feature_keys
        self.depth = depth
        self.cache_group_size = cache_group_size
        self.final_output_key = final_output_key
        self.point_capture_name = point_capture_name
        self.label_semantics_valid = label_semantics_valid
        self.temporary_root = temporary_root
        self.cache_backend = cache_backend
        self.cache_tracker = cache_tracker

    def vector_recovery_metrics(
        self,
        patch_vector: torch.Tensor,
        target_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_flat = patch_vector.reshape(patch_vector.shape[0], -1).float()
        target_flat = target_vector.reshape(target_vector.shape[0], -1).float()
        target_square = target_flat.square().sum(dim=1)
        patch_norm = patch_flat.norm(dim=1)
        target_norm = target_flat.norm(dim=1)
        dot = (patch_flat * target_flat).sum(dim=1)
        projection_fraction = dot / target_square.clamp_min(1e-12)
        target_cosine = dot / (patch_norm * target_norm).clamp_min(1e-12)
        return projection_fraction, target_cosine

    def recovery_responses(
        self,
        model: nn.Module,
        *,
        side: str,
        precomputed_corruption_audit: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        model.eval().to(self.device)
        if precomputed_corruption_audit is None:
            corruption_audit = _patching_corruption_validity_audit(
                model,
                self.batches,
                device=self.device,
                corruption=self.corruption,
                mean=self.mean,
                std=self.std,
                same_head=bool(self.same_head),
                minimum_relative_effect=float(self.minimum_relative_effect),
                minimum_prediction_retention=float(self.minimum_prediction_retention),
                noise_sigma=float(self.noise_sigma),
                noise_seed=int(self.noise_seed),
                blur_sigma=float(self.blur_sigma),
                blur_kernel_size=int(self.blur_kernel_size),
                blur_padding=str(self.blur_padding),
                mask_size=int(self.mask_size),
                mask_positions=self.mask_positions,
                mask_fill=str(self.mask_fill),
            )
            audit_reused = False
        else:
            corruption_audit = dict(precomputed_corruption_audit)
            audit_reused = True
            if str(corruption_audit.get("corruption", "")) != str(self.corruption):
                raise ValueError("DiR precomputed corruption audit type mismatch")
            if bool(corruption_audit.get("same_head", False)) != bool(self.same_head):
                raise ValueError("DiR precomputed corruption audit head contract mismatch")
        audits = dict(corruption_audit["audits"])
        valid_masks = {
            str(key): value.detach().bool().cpu()
            for key, value in dict(corruption_audit["valid_masks"]).items()
        }
        audit_forward_count = int(corruption_audit.get("audit_forward_count", 0))
        payloads: dict[str, dict[str, Any]] = {}
        point_errors: dict[str, list[dict[str, Any]]] = {point: [] for point in self.points}
        for point in self.points:
            payloads[point] = {
                "references": {key: [None] * self.depth for key in self.base_feature_keys},
                "grams": {key: [None] * self.depth for key in self.base_feature_keys},
                "feature_dimensions": {key: [None] * self.depth for key in self.base_feature_keys},
                "scalar": [None] * self.depth,
                "intervention_forward_count": 0,
            }
        grouped_baseline_forward_count = 0
        with torch.no_grad():
            for group_start in range(0, self.depth, self.cache_group_size):
                block_indices = list(
                    range(group_start, min(self.depth, group_start + self.cache_group_size))
                )
                chunks = {
                    point: {
                        key: {index: [] for index in block_indices}
                        for key in self.base_feature_keys
                    }
                    for point in self.points
                }
                scalar_chunks = {
                    point: {index: [] for index in block_indices} for point in self.points
                }
                group_capture_points = [
                    "pre_classifier",
                    self.final_output_key,
                    *[f"block_{index:02d}_input" for index in block_indices],
                    *[
                        f"block_{index:02d}_{self.point_capture_name[point]}"
                        for index in block_indices
                        for point in self.points
                    ],
                ]
                sample_offset = 0
                for images_cpu, labels_cpu, ids_cpu in self.batches:
                    images = images_cpu.to(self.device)
                    labels = labels_cpu.to(self.device)
                    ids = ids_cpu.to(self.device)
                    batch_size = int(images.shape[0])
                    corrupted = apply_weak_corruption(
                        images,
                        corruption=self.corruption,
                        sample_ids=ids,
                        mean=self.mean,
                        std=self.std,
                        noise_sigma=float(self.noise_sigma),
                        noise_seed=int(self.noise_seed),
                        blur_sigma=float(self.blur_sigma),
                        blur_kernel_size=int(self.blur_kernel_size),
                        blur_padding=str(self.blur_padding),
                        mask_size=int(self.mask_size),
                        mask_positions=self.mask_positions,
                        mask_fill=str(self.mask_fill),
                    )
                    clean_logits, clean = forward_with_capture_and_interventions(
                        model, images, capture_points=group_capture_points
                    )
                    corrupted_logits, corrupted_taps = forward_with_capture_and_interventions(
                        model, corrupted, capture_points=group_capture_points
                    )
                    grouped_baseline_forward_count += 2
                    clean_post = clean["pre_classifier"].float()
                    corrupted_post = corrupted_taps["pre_classifier"].float()
                    clean_pre = clean[self.final_output_key].float()
                    corrupted_pre = corrupted_taps[self.final_output_key].float()
                    post_target_tensor = clean_post - corrupted_post
                    post_targets = {
                        "full": _feature_view(post_target_tensor, "full_token"),
                        "cls": _feature_view(post_target_tensor, "cls"),
                        "patch": _feature_view(post_target_tensor, "patch"),
                    }
                    targets = {
                        "post_layernorm_full": post_targets["full"],
                        "post_layernorm_cls": post_targets["cls"],
                        "post_layernorm_patch": post_targets["patch"],
                    }
                    if self.same_head:
                        targets["logits"] = clean_logits.float() - corrupted_logits.float()
                    post_corruption_distances = {
                        key: value.norm(dim=1) for key, value in post_targets.items()
                    }
                    valid_slice = slice(sample_offset, sample_offset + batch_size)
                    local_valid = {
                        key: valid_masks[key][valid_slice].to(self.device)
                        for key in (
                            "post_layernorm_full",
                            "post_layernorm_cls",
                            "post_layernorm_patch",
                        )
                    }
                    corrupted_block_inputs = {
                        block_index: corrupted_taps[f"block_{block_index:02d}_input"]
                        for block_index in block_indices
                    }
                    for block_index in block_indices:
                        for point in self.points:
                            if point_errors[point]:
                                continue
                            try:
                                replacement = clean[
                                    f"block_{block_index:02d}_{self.point_capture_name[point]}"
                                ]
                                patched_logits, patched = forward_from_block_input_with_interventions(
                                    model,
                                    corrupted_block_inputs[block_index],
                                    start_block_index=block_index,
                                    capture_points=["pre_classifier", "final_cls", self.final_output_key],
                                    interventions=[
                                        Intervention(
                                            block_index,
                                            point,
                                            replacement=replacement,
                                        )
                                    ],
                                )
                                payloads[point]["intervention_forward_count"] += 1
                                patched_post = patched["pre_classifier"].float()
                                patched_pre = patched[self.final_output_key].float()
                                post_patch_tensor = patched_post - corrupted_post
                                pre_patch_tensor = patched_pre - corrupted_pre
                                patches = {
                                    "post_layernorm_cls": _feature_view(post_patch_tensor, "cls"),
                                    "post_layernorm_patch": _feature_view(post_patch_tensor, "patch"),
                                    "pre_layernorm_cls": _feature_view(pre_patch_tensor, "cls"),
                                    "pre_layernorm_patch": _feature_view(pre_patch_tensor, "patch"),
                                }
                                if self.same_head:
                                    patches["logits"] = patched_logits.float() - corrupted_logits.float()
                                for key in self.base_feature_keys:
                                    chunks[point][key][block_index].append(patches[key].cpu())

                                post_patch_views = {
                                    "full": _feature_view(post_patch_tensor, "full_token"),
                                    "cls": patches["post_layernorm_cls"],
                                    "patch": patches["post_layernorm_patch"],
                                }
                                scalar_columns: list[torch.Tensor] = []
                                for mode in ("full", "cls", "patch"):
                                    patched_distance = _feature_view(
                                        patched_post - clean_post,
                                        "full_token" if mode == "full" else mode,
                                    ).norm(dim=1)
                                    corruption_distance = post_corruption_distances[mode]
                                    absolute = corruption_distance - patched_distance
                                    fractional = absolute / corruption_distance.clamp_min(1e-12)
                                    projection, direction = self.vector_recovery_metrics(
                                        post_patch_views[mode], post_targets[mode]
                                    )
                                    scalar_columns.extend([absolute, fractional, projection, direction])
                                if self.same_head:
                                    logit_target = targets["logits"]
                                    logit_patch = patches["logits"]
                                    logit_corruption_distance = logit_target.reshape(
                                        images.shape[0], -1
                                    ).norm(dim=1)
                                    logit_patched_distance = (
                                        patched_logits.float() - clean_logits.float()
                                    ).reshape(images.shape[0], -1).norm(dim=1)
                                    logit_absolute = logit_corruption_distance - logit_patched_distance
                                    logit_fractional = logit_absolute / logit_corruption_distance.clamp_min(1e-12)
                                    logit_projection, logit_direction = self.vector_recovery_metrics(
                                        logit_patch, logit_target
                                    )
                                    scalar_columns.extend(
                                        [
                                            logit_absolute,
                                            logit_fractional,
                                            logit_projection,
                                            logit_direction,
                                        ]
                                    )
                                if self.label_semantics_valid:
                                    scalar_columns.append(
                                        _true_class_margin(patched_logits, labels)
                                        - _true_class_margin(corrupted_logits, labels)
                                    )
                                if self.same_head:
                                    scalar_columns.append(
                                        (
                                            (patched_logits.argmax(dim=1) == clean_logits.argmax(dim=1))
                                            & (
                                                corrupted_logits.argmax(dim=1)
                                                != clean_logits.argmax(dim=1)
                                            )
                                        ).float()
                                    )
                                scalar_columns.extend(
                                    [
                                        local_valid["post_layernorm_full"].float(),
                                        local_valid["post_layernorm_cls"].float(),
                                        local_valid["post_layernorm_patch"].float(),
                                    ]
                                )
                                scalar_chunks[point][block_index].append(
                                    torch.stack(scalar_columns, dim=1).cpu()
                                )
                                del patched, patched_logits, patched_post, patched_pre
                                del post_patch_tensor, pre_patch_tensor, patches, post_patch_views
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
                    sample_offset += batch_size
                    del clean, corrupted_taps, clean_logits, corrupted_logits, targets

                for point in self.points:
                    if point_errors[point]:
                        continue
                    payload = payloads[point]
                    for block_index in block_indices:
                        payload["scalar"][block_index] = torch.cat(
                            scalar_chunks[point][block_index], dim=0
                        )
                        for key in self.base_feature_keys:
                            block_chunks = chunks[point][key][block_index]
                            gram = _gram_from_feature_chunks(block_chunks, device=self.device)
                            feature_dimension = int(
                                block_chunks[0].reshape(block_chunks[0].shape[0], -1).shape[1]
                            )
                            payload["references"][key][block_index] = _save_feature_chunks(
                                block_chunks,
                                root=self.temporary_root,
                                stem=f"{side}_patching_{self.corruption}_{point}_{key}_{block_index:02d}",
                                backend=self.cache_backend,
                                cache_tracker=self.cache_tracker,
                            )
                            payload["grams"][key][block_index] = gram
                            payload["feature_dimensions"][key][block_index] = feature_dimension
                    del chunks[point], scalar_chunks[point]

        def require(values: Sequence[Any], *, label: str) -> list[Any]:
            if any(value is None for value in values):
                raise RuntimeError(f"DiR patching cache incomplete: {label}")
            return [value for value in values if value is not None]

        for point in self.points:
            if point_errors[point]:
                payloads[point] = {
                    "point_errors": point_errors[point],
                    "audit_forward_count": int(audit_forward_count),
                    "grouped_baseline_forward_count": int(grouped_baseline_forward_count),
                    "intervention_forward_count": int(
                        payloads[point]["intervention_forward_count"]
                    ),
                    "baseline_shared_across_intervention_points": len(self.points) > 1,
                }
                continue
            payload = payloads[point]
            for key in self.base_feature_keys:
                payload["references"][key] = require(
                    payload["references"][key], label=f"{point}/references/{key}"
                )
                payload["grams"][key] = require(
                    payload["grams"][key], label=f"{point}/grams/{key}"
                )
                payload["feature_dimensions"][key] = require(
                    payload["feature_dimensions"][key],
                    label=f"{point}/feature_dimensions/{key}",
                )
            payload["scalar"] = require(payload["scalar"], label=f"{point}/scalar")
            payload["audits"] = audits
            payload["valid_masks"] = valid_masks
            payload["audit_forward_count"] = int(audit_forward_count)
            payload["corruption_audit_reused"] = bool(audit_reused)
            payload["grouped_baseline_forward_count"] = int(grouped_baseline_forward_count)
            payload["baseline_shared_across_intervention_points"] = len(self.points) > 1
        return payloads


def patching_recovery_alignment_suite(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    corruption: str,
    intervention_points: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    minimum_relative_effect: float = 0.05,
    minimum_prediction_retention: float = 0.80,
    minimum_common_valid_samples: int = 32,
    minimum_block_recovery_fraction: float = 0.01,
    minimum_median_recovery_fraction: float = 0.0,
    minimum_positive_recovery_sample_fraction: float = 0.50,
    same_head: bool,
    label_semantics_valid: bool,
    minimum_signal_rms_absolute: float = 1e-8,
    minimum_signal_rms_relative_to_median: float = 0.05,
    noise_sigma: float = 0.03,
    noise_seed: int = 2026080602,
    blur_sigma: float = 0.8,
    blur_kernel_size: int = 3,
    blur_padding: str = "reflect",
    mask_size: int = 8,
    mask_positions: Sequence[int] = (4, 12, 20),
    mask_fill: str = "channel_mean",
    cache_directory: str | Path | None = None,
    precomputed_left_corruption_audit: Mapping[str, Any] | None = None,
    precomputed_right_corruption_audit: Mapping[str, Any] | None = None,
    external_common_valid_masks: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate one corruption at several patch points with shared baselines."""

    points = _normalize_causal_intervention_points(intervention_points)
    if label_semantics_valid and not same_head:
        raise ValueError("DiR label semantics cannot be valid without a shared head")
    depth = len(left_model.transformer_blocks)
    if depth != len(right_model.transformer_blocks):
        raise ValueError("DiR patching comparison requires equal depth")
    final_output_key = f"block_{depth - 1:02d}_output"
    cache_group_size = 2
    point_capture_name = {
        "block_update": "update",
        "post_o_attention_output": "post_o_attention_output",
        "post_w2_mlp_output": "post_w2_mlp_output",
    }
    base_feature_keys = [
        "post_layernorm_cls",
        "post_layernorm_patch",
        "pre_layernorm_cls",
        "pre_layernorm_patch",
    ]
    if same_head:
        base_feature_keys.append("logits")
    audit_feature_keys = [
        "post_layernorm_full",
        "post_layernorm_cls",
        "post_layernorm_patch",
        "pre_layernorm_full",
        "pre_layernorm_cls",
        "pre_layernorm_patch",
    ]
    if same_head:
        audit_feature_keys.append("logits")

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
        prefix=f"dir_patching_suite_{corruption}_",
        parent=cache_parent,
        cleanup_status=cleanup_status,
    ) as temporary_root:
        cleanup_status["cache_backend"] = cache_backend
        cleanup_status["cache_plan"] = dict(cache_plan)
        cleanup_status["cache_runtime"] = cache_tracker


        runtime = _PatchingRecoveryRuntime(
            device=device,
            batches=batches,
            corruption=corruption,
            mean=mean,
            std=std,
            same_head=same_head,
            minimum_relative_effect=minimum_relative_effect,
            minimum_prediction_retention=minimum_prediction_retention,
            noise_sigma=noise_sigma,
            noise_seed=noise_seed,
            blur_sigma=blur_sigma,
            blur_kernel_size=blur_kernel_size,
            blur_padding=blur_padding,
            mask_size=mask_size,
            mask_positions=mask_positions,
            mask_fill=mask_fill,
            points=points,
            base_feature_keys=base_feature_keys,
            depth=depth,
            cache_group_size=cache_group_size,
            final_output_key=final_output_key,
            point_capture_name=point_capture_name,
            label_semantics_valid=label_semantics_valid,
            temporary_root=temporary_root,
            cache_backend=cache_backend,
            cache_tracker=cache_tracker,
        )
        left_by_point = runtime.recovery_responses(
            left_model, side="left", precomputed_corruption_audit=precomputed_left_corruption_audit
        )
        right_by_point = runtime.recovery_responses(
            right_model, side="right", precomputed_corruption_audit=precomputed_right_corruption_audit
        )
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
                    corruption=corruption,
                )
                continue
            try:
                results[point] = _finalize_patching_response_pair(
                    left_by_point[point],
                    right_by_point[point],
                    corruption=corruption,
                    intervention_point=point,
                    depth=depth,
                    cache_group_size=cache_group_size,
                    same_head=bool(same_head),
                    label_semantics_valid=bool(label_semantics_valid),
                    minimum_common_valid_samples=int(minimum_common_valid_samples),
                    minimum_block_recovery_fraction=float(minimum_block_recovery_fraction),
                    minimum_median_recovery_fraction=float(minimum_median_recovery_fraction),
                    minimum_positive_recovery_sample_fraction=float(
                        minimum_positive_recovery_sample_fraction
                    ),
                    minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
                    minimum_signal_rms_relative_to_median=float(
                        minimum_signal_rms_relative_to_median
                    ),
                    cleanup_status=cleanup_status,
                    external_common_valid_masks=external_common_valid_masks,
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
                    corruption=corruption,
                )
        return results
