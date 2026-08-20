"""Internal VJP operator alignment measurement."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import forward_with_capture_and_interventions
from .representation_similarity import _jacobian_gram_similarity_matrix, _degenerate_aware_subspace_overlap, pairwise_cka_matrix, pairwise_biased_cka_matrix
from .representation_cache import pairwise_paired_output_metric_matrices, sample_variation_rms, combined_signal_variation_validity
from .jacobian_common import (
    _measurement_status_from_primary_validity,
    _numerical_svd_rank,
    _flat_rank_correlation,
    _rademacher,
    _shared_probe_projection_seed,
)

def _compact_operator_descriptors(
    values: Sequence[torch.Tensor],
    *,
    dominant_subspace_rank: int,
    minimum_signal_rms_absolute: float = 1e-8,
) -> list[dict[str, Any]]:
    """Build finite, rank-aware descriptors for projected internal VJP sketches."""

    descriptors: list[dict[str, Any]] = []
    for value in values:
        response = value.detach().double()
        if not torch.isfinite(response).all():
            raise ValueError("DiR internal VJP sketch contains non-finite values")
        raw_rms = float(response.square().mean().sqrt().cpu())
        width = int(response.shape[1])
        if raw_rms <= float(minimum_signal_rms_absolute):
            descriptors.append(
                {
                    "kernel": torch.zeros(width, width, dtype=torch.float64),
                    "spectrum": torch.zeros(width, dtype=torch.float64),
                    "subspace": torch.empty(width, 0, dtype=torch.float64),
                    "rank_used": 0,
                    "status": "completed_below_detection_operator",
                    "degenerate": True,
                    "response_class": "below_detection_threshold",
                    "raw_response_rms": float(raw_rms),
                    "below_detection_signal_rms_threshold": float(minimum_signal_rms_absolute),
                }
            )
            continue
        centered = response - response.mean(dim=0, keepdim=True)
        centered_rms = float(centered.square().mean().sqrt().cpu())
        constant_response = centered_rms <= float(minimum_signal_rms_absolute)
        descriptor_response = response if constant_response else centered
        kernel_raw = descriptor_response.T @ descriptor_response
        trace = float(kernel_raw.trace().abs().cpu())
        if trace <= 1e-30:
            raise ValueError("DiR nonzero internal VJP response produced a zero descriptor kernel")
        kernel = kernel_raw / trace
        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        eigenvectors = eigenvectors[:, order]
        numerical_rank, tolerance = _numerical_svd_rank(
            eigenvalues,
            row_count=int(kernel.shape[0]),
            column_count=int(kernel.shape[1]),
        )
        rank = min(
            int(dominant_subspace_rank), int(numerical_rank), int(eigenvectors.shape[1])
        )
        spectrum = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        descriptors.append(
            {
                "kernel": kernel,
                "spectrum": spectrum,
                "subspace": eigenvectors[:, :rank],
                "rank_used": int(rank),
                "numerical_rank": int(numerical_rank),
                "rank_tolerance": float(tolerance),
                "status": (
                    "completed_constant_response"
                    if constant_response
                    else ("completed" if rank > 0 else "completed_numerical_rank_zero_descriptor")
                ),
                "degenerate": bool(rank == 0),
                "response_class": "constant_response" if constant_response else "sample_varying_response",
                "raw_response_rms": float(raw_rms),
                "centered_response_rms": float(centered_rms),
                "below_detection_signal_rms_threshold": float(minimum_signal_rms_absolute),
            }
        )
    return descriptors
def _descriptor_similarity_matrix(
    left_values: Sequence[torch.Tensor],
    right_values: Sequence[torch.Tensor],
    *,
    key: str,
    dominant_subspace_rank: int,
    minimum_signal_rms_absolute: float = 1e-8,
) -> list[list[float]]:
    left_descriptors = _compact_operator_descriptors(
        left_values,
        dominant_subspace_rank=dominant_subspace_rank,
        minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
    )
    right_descriptors = _compact_operator_descriptors(
        right_values,
        dominant_subspace_rank=dominant_subspace_rank,
        minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
    )
    output: list[list[float]] = []
    for left in left_descriptors:
        row: list[float] = []
        for right in right_descriptors:
            left_rank = int(left["rank_used"])
            right_rank = int(right["rank_used"])
            if left_rank == 0 or right_rank == 0:
                score = float("nan")
            elif key == "subspace":
                score, _classification = _degenerate_aware_subspace_overlap(
                    left[key], right[key]
                )
            else:
                left_tensor = left[key].reshape(-1)
                right_tensor = right[key].reshape(-1)
                if int(left_tensor.numel()) != int(right_tensor.numel()):
                    raise ValueError("DiR internal VJP descriptor dimensions differ")
                score = float(
                    (
                        torch.dot(left_tensor, right_tensor)
                        / (left_tensor.norm() * right_tensor.norm()).clamp_min(1e-12)
                    ).cpu()
                )
            row.append(float(score))
        output.append(row)
    return output
def _descriptor_rank_classification_matrix(
    left_values: Sequence[torch.Tensor],
    right_values: Sequence[torch.Tensor],
    *,
    dominant_subspace_rank: int,
    minimum_signal_rms_absolute: float = 1e-8,
) -> list[list[str]]:
    left = _compact_operator_descriptors(
        left_values,
        dominant_subspace_rank=dominant_subspace_rank,
        minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
    )
    right = _compact_operator_descriptors(
        right_values,
        dominant_subspace_rank=dominant_subspace_rank,
        minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
    )
    output: list[list[str]] = []
    for left_item in left:
        row: list[str] = []
        for right_item in right:
            left_rank = int(left_item["rank_used"])
            right_rank = int(right_item["rank_used"])
            if left_rank == 0 and right_rank == 0:
                row.append("both_rank_zero_inconclusive")
            elif left_rank == 0 or right_rank == 0:
                row.append("one_rank_zero_inconclusive")
            elif left_rank < int(dominant_subspace_rank) or right_rank < int(dominant_subspace_rank):
                row.append("valid_low_rank")
            else:
                row.append("positive_rank")
        output.append(row)
    return output
def _vjp_split_half_audit(
    left_values: Sequence[torch.Tensor],
    right_values: Sequence[torch.Tensor],
    *,
    probe_count: int,
    spearman_minimum: float,
    diagonal_difference_maximum: float,
    norm_relative_difference_maximum: float,
) -> dict[str, Any]:
    half = int(probe_count) // 2
    if half < 1:
        return {"stable": False, "reason": "fewer_than_two_probes"}
    first_left = [value[:, :half] for value in left_values]
    first_right = [value[:, :half] for value in right_values]
    second_left = [value[:, half:] for value in left_values]
    second_right = [value[:, half:] for value in right_values]
    first = np.asarray(pairwise_cka_matrix(first_left, first_right), dtype=np.float64)
    second = np.asarray(pairwise_cka_matrix(second_left, second_right), dtype=np.float64)
    correlation = _flat_rank_correlation(first, second)
    correlation_defined = bool(np.isfinite(correlation))
    diagonal_difference = float(abs(np.diag(first).mean() - np.diag(second).mean()))

    def mean_norm(values: Sequence[torch.Tensor]) -> float:
        return float(np.mean([float(value.float().square().mean().sqrt()) for value in values]))

    first_norm = 0.5 * (mean_norm(first_left) + mean_norm(first_right))
    second_norm = 0.5 * (mean_norm(second_left) + mean_norm(second_right))
    norm_difference = abs(first_norm - second_norm) / max(
        1e-12, 0.5 * (first_norm + second_norm)
    )
    return {
        "spearman_matrix_correlation": correlation,
        "spearman_defined": correlation_defined,
        "diagonal_mean_difference": diagonal_difference,
        "rms_sensitivity_relative_difference": float(norm_difference),
        "stable": bool(
            correlation_defined
            and correlation >= float(spearman_minimum)
            and diagonal_difference <= float(diagonal_difference_maximum)
            and norm_difference <= float(norm_relative_difference_maximum)
        ),
        "thresholds": {
            "spearman_minimum": float(spearman_minimum),
            "diagonal_difference_maximum": float(diagonal_difference_maximum),
            "norm_relative_difference_maximum": float(norm_relative_difference_maximum),
        },
    }
def jacobian_internal_vjp_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    probe_count: int,
    probe_seed: int,
    split_half_spearman_minimum: float = 0.80,
    split_half_diagonal_difference_maximum: float = 0.05,
    split_half_norm_relative_difference_maximum: float = 0.15,
    dominant_subspace_rank: int = 4,
    minimum_signal_rms_absolute: float = 1e-8,
    minimum_signal_rms_relative_to_median: float = 0.05,
) -> dict[str, Any]:
    """Compact VJP sketches from internal residual sites to final normalized CLS.

    For each output Rademacher direction, the exact VJP is computed and then
    projected onto a fixed Rademacher direction at the internal tensor. This
    keeps the report compact while using identical probes in all models.
    ``block_output`` is used for the block-update path because an additive
    residual update and its resulting block output have the same suffix
    derivative.
    """

    depth = len(left_model.transformer_blocks)
    if depth != len(right_model.transformer_blocks):
        raise ValueError("DiR internal VJP comparison requires equal depth")
    point_to_suffix = {
        "block_update_to_final_representation": "output",
        "post_o_attention_output_to_final_representation": "post_o_attention_output",
        "post_w2_mlp_output_to_final_representation": "post_w2_mlp_output",
    }

    def model_sketch(model: nn.Module) -> dict[str, list[torch.Tensor]]:
        model.eval().to(device)
        collected: dict[str, list[list[torch.Tensor]]] = {
            name: [[] for _ in range(depth)] for name in point_to_suffix
        }
        sample_offset = 0
        for images_cpu, _labels, _ids in batches:
            images = images_cpu.to(device).float()
            batch_size = int(images.shape[0])
            batch_rows: dict[str, list[list[torch.Tensor]]] = {
                name: [[] for _ in range(depth)] for name in point_to_suffix
            }
            for probe_index in range(int(probe_count)):
                capture_points = [
                    "final_cls",
                    *[
                        f"block_{block_index:02d}_{suffix}"
                        for suffix in point_to_suffix.values()
                        for block_index in range(depth)
                    ],
                ]
                _logits, taps = forward_with_capture_and_interventions(
                    model, images, capture_points=capture_points
                )
                final_cls = taps["final_cls"]
                output_direction = _rademacher(
                    (1, *final_cls.shape[1:]),
                    seed=int(probe_seed) + 1000003 * probe_index,
                    device=device,
                    dtype=final_cls.dtype,
                ).expand_as(final_cls)
                scalar = (final_cls * output_direction).sum()
                internal_tensors: list[torch.Tensor] = []
                keys: list[tuple[str, int]] = []
                for name, suffix in point_to_suffix.items():
                    for block_index in range(depth):
                        internal_tensors.append(taps[f"block_{block_index:02d}_{suffix}"])
                        keys.append((name, block_index))
                gradients = torch.autograd.grad(
                    scalar,
                    internal_tensors,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for tensor_index, (name, block_index) in enumerate(keys):
                    gradient = gradients[tensor_index].detach()
                    internal_direction = _rademacher(
                        (1, *gradient.shape[1:]),
                        seed=_shared_probe_projection_seed(
                            probe_seed,
                            probe_index,
                            family_offset=(
                                2000003 + 271 * list(point_to_suffix).index(name)
                            ),
                        ),
                        device=device,
                        dtype=gradient.dtype,
                    ).expand_as(gradient)
                    projected = (gradient * internal_direction).reshape(batch_size, -1).sum(dim=1, keepdim=True)
                    batch_rows[name][block_index].append(projected.cpu())
                del taps, internal_tensors, gradients
            for name in point_to_suffix:
                for block_index in range(depth):
                    collected[name][block_index].append(
                        torch.cat(batch_rows[name][block_index], dim=1)
                    )
            sample_offset += batch_size
        return {
            name: [torch.cat(parts, dim=0) for parts in block_parts]
            for name, block_parts in collected.items()
        }

    left = model_sketch(left_model)
    right = model_sketch(right_model)
    result: dict[str, Any] = {
        "probe_count": int(probe_count),
        "probe_seed": int(probe_seed),
        "dominant_subspace_rank": int(dominant_subspace_rank),
        "block_update_derivative_proxy": "suffix derivative with respect to block output; identical to additive block-update derivative",
        "probe_direction_contract": "one_shared_output_cotangent_and_one_shared_internal_projection_direction_broadcast_across_all_samples_and_models_per_probe",
        "cka_contract": "U_centered_debiased_primary_biased_auxiliary",
        "absolute_norm_interpretation": "rms_projected_vjp_sensitivity_not_unscaled_frobenius_norm",
        "descriptor_interpretation": "random_projected_VJP_probe_sketch_not_an_explicit_internal_Jacobian_spectrum_or_singular_subspace",
        "paths": {},
    }
    stable_flags: list[bool] = []
    primary_measurement_status: dict[str, str] = {}
    primary_valid_flags: list[bool] = []
    for name in point_to_suffix:
        left_values = left[name]
        right_values = right[name]
        split = _vjp_split_half_audit(
            left_values,
            right_values,
            probe_count=int(probe_count),
            spearman_minimum=float(split_half_spearman_minimum),
            diagonal_difference_maximum=float(split_half_diagonal_difference_maximum),
            norm_relative_difference_maximum=float(split_half_norm_relative_difference_maximum),
        )
        stable_flags.append(bool(split.get("stable", False)))
        left_rms = [float(value.float().square().mean().sqrt()) for value in left_values]
        right_rms = [float(value.float().square().mean().sqrt()) for value in right_values]
        left_signal = combined_signal_variation_validity(
            left_rms,
            [sample_variation_rms(value) for value in left_values],
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        right_signal = combined_signal_variation_validity(
            right_rms,
            [sample_variation_rms(value) for value in right_values],
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        for value in [*left_values, *right_values]:
            if not torch.isfinite(value).all():
                raise ValueError("DiR internal VJP primary sketch contains non-finite values")
        left_gram_values = [
            {
                "gram": value.double() @ value.double().T,
                "rms": float(value.double().square().mean().sqrt()),
                "projection_features": value,
            }
            for value in left_values
        ]
        right_gram_values = [
            {
                "gram": value.double() @ value.double().T,
                "rms": float(value.double().square().mean().sqrt()),
                "projection_features": value,
            }
            for value in right_values
        ]
        primary_matrix, degenerate_classification, constant_projection_cosine = _jacobian_gram_similarity_matrix(
            left_gram_values,
            right_gram_values,
            minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
        )
        primary_mask = [
            [str(classification) == "nondegenerate_debiased_cka" for classification in row]
            for row in degenerate_classification
        ]
        primary_valid = bool(np.diag(np.asarray(primary_mask, dtype=bool)).any())
        primary_valid_flags.append(primary_valid)
        primary_key = f"paths.{name}.sample_gram_debiased_cka_12x12"
        primary_measurement_status[primary_key] = (
            "valid" if primary_valid else "inconclusive_no_detectable_internal_vjp"
        )
        left_signal["role"] = (
            "diagnostic_signal_strength_and_variation; absolute_below_detection_responses_are_inconclusive_for_Jacobian_similarity"
        )
        right_signal["role"] = left_signal["role"]
        result["paths"][name] = {
            "sample_gram_debiased_cka_12x12": primary_matrix,
            "auxiliary_sample_gram_biased_cka_12x12": pairwise_biased_cka_matrix(left_values, right_values),
            "paired_vjp_sketch_metrics": pairwise_paired_output_metric_matrices(left_values, right_values),
            "validity_masks": {
                "sample_gram_debiased_cka_12x12": primary_mask
            },
            "jacobian_response_degenerate_classification_12x12": degenerate_classification,
            "auxiliary_constant_response_shared_projection_mean_cosine_12x12": constant_projection_cosine,
            "low_signal": {"left": left_signal, "right": right_signal},
            "projected_vjp_probe_kernel_cosine_12x12": _descriptor_similarity_matrix(
                left_values,
                right_values,
                key="kernel",
                dominant_subspace_rank=int(dominant_subspace_rank),
                minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
            ),
            "projected_vjp_probe_spectrum_cosine_12x12": _descriptor_similarity_matrix(
                left_values,
                right_values,
                key="spectrum",
                dominant_subspace_rank=int(dominant_subspace_rank),
                minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
            ),
            "projected_vjp_probe_subspace_overlap_12x12": _descriptor_similarity_matrix(
                left_values,
                right_values,
                key="subspace",
                dominant_subspace_rank=int(dominant_subspace_rank),
                minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
            ),
            "projected_vjp_rank_classification_12x12": _descriptor_rank_classification_matrix(
                left_values,
                right_values,
                dominant_subspace_rank=int(dominant_subspace_rank),
                minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
            ),
            "left_rms_vjp_sensitivity": left_rms,
            "right_rms_vjp_sensitivity": right_rms,
            "split_half": split,
        }
    result["split_half"] = {
        "stable": bool(stable_flags and all(stable_flags)),
        "path_stability": {
            name: bool(result["paths"][name]["split_half"].get("stable", False))
            for name in point_to_suffix
        },
    }
    result["primary_measurement_status"] = primary_measurement_status
    result["measurement_status"] = _measurement_status_from_primary_validity(
        primary_valid_flags,
        no_valid_status="inconclusive_no_valid_internal_vjp",
    )
    result["jacobian_degenerate_similarity_contract"] = (
        "sample_varying_paths_use_debiased_CKA; below_detection_and_detected_constant_"
        "responses_are_inconclusive_for_primary_CKA; constant_response_shared_"
        "projection_mean_direction_cosine_is_auxiliary_only; positive_low_rank_"
        "descriptors_remain_valid"
    )
    return result
