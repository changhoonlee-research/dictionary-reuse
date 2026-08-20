"""Natural-update and direct block-function correspondence measurements."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import forward_with_capture_and_interventions
from .representation_similarity import (
    _feature_view,
    _paired_output_metrics_from_components,
    paired_output_metrics,
)
from .representation_cache import (
    _component_sample_variation_rms,
    _component_signal_rms,
    _finite_elementwise_mean,
    _gram_from_feature_chunks,
    _native_update_grams_and_norms,
    _pairwise_biased_gram_cka_matrix,
    _pairwise_gram_cka_matrix,
    combined_signal_variation_validity,
    gram_variation_strength,
    outer_validity_mask,
    sample_variation_rms,
)

def block_update_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    minimum_signal_rms_absolute: float = 1e-8,
    minimum_signal_rms_relative_to_median: float = 0.05,
    capture_block_group_size: int = 3,
) -> dict[str, Any]:
    left = _native_update_grams_and_norms(
        left_model,
        batches,
        device=device,
        capture_block_group_size=int(capture_block_group_size),
    )
    right = _native_update_grams_and_norms(
        right_model,
        batches,
        device=device,
        capture_block_group_size=int(capture_block_group_size),
    )
    result: dict[str, Any] = {
        "block_update_full_token_debiased_cka_12x12": _pairwise_gram_cka_matrix(
            left["full_token_grams"], right["full_token_grams"]
        ),
        "block_update_cls_debiased_cka_12x12": _pairwise_gram_cka_matrix(
            left["cls_grams"], right["cls_grams"]
        ),
        "block_update_patch_debiased_cka_12x12": _pairwise_gram_cka_matrix(
            left["patch_grams"], right["patch_grams"]
        ),
        "auxiliary_biased_cka": {
            "block_update_full_token_biased_cka_12x12": _pairwise_biased_gram_cka_matrix(
                left["full_token_grams"], right["full_token_grams"]
            ),
            "block_update_cls_biased_cka_12x12": _pairwise_biased_gram_cka_matrix(
                left["cls_grams"], right["cls_grams"]
            ),
            "block_update_patch_biased_cka_12x12": _pairwise_biased_gram_cka_matrix(
                left["patch_grams"], right["patch_grams"]
            ),
        },
        "left_block_update_full_token_norm": left["full_token_norms"],
        "right_block_update_full_token_norm": right["full_token_norms"],
        "left_block_update_cls_norm": left["cls_norms"],
        "right_block_update_cls_norm": right["cls_norms"],
        "left_block_update_patch_norm": left["patch_norms"],
        "right_block_update_patch_norm": right["patch_norms"],
        "capture_execution": {
            "left_forward_count": int(left["capture_forward_count"]),
            "right_forward_count": int(right["capture_forward_count"]),
            "contract": str(left["capture_contract"]),
            "block_group_size": int(left["capture_block_group_size"]),
            "group_count": int(left["capture_group_count"]),
        },
        "cka_contract": "U_centered_debiased_primary_biased_auxiliary",
        "token_contract": "CLS_and_patch_reported_separately_full_token_is_auxiliary_combination",
        "token_stage_contract": "native_residual_block_update_space_before_next_block_pre_norm",
        "memory_policy": "capture_bounded_block_groups_then_build_grams_and_release_raw_activations",
    }
    validity_masks: dict[str, Any] = {}
    low_signal: dict[str, Any] = {}
    for mode in ("full_token", "cls", "patch"):
        left_validity = combined_signal_variation_validity(
            left[f"{mode}_norms"],
            [gram_variation_strength(value) for value in left[f"{mode}_grams"]],
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        right_validity = combined_signal_variation_validity(
            right[f"{mode}_norms"],
            [gram_variation_strength(value) for value in right[f"{mode}_grams"]],
            absolute_minimum=float(minimum_signal_rms_absolute),
            relative_to_median=float(minimum_signal_rms_relative_to_median),
        )
        key = f"block_update_{mode}_debiased_cka_12x12"
        validity_masks[key] = outer_validity_mask(
            left_validity["valid_by_block"], right_validity["valid_by_block"]
        )
        low_signal[mode] = {"left": left_validity, "right": right_validity}
    result["validity_masks"] = validity_masks
    result["low_signal"] = low_signal
    result["primary_metrics"] = [
        "block_update_cls_debiased_cka_12x12",
        "block_update_patch_debiased_cka_12x12",
    ]
    result["full_token_role"] = "auxiliary_only"
    return result



def prepare_block_update_cka_reference(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    capture_block_group_size: int = 3,
) -> dict[str, Any]:
    """Cache one fixed Source block-update reference for a training trajectory."""

    original_device = next(model.parameters()).device
    was_training = bool(model.training)
    try:
        captured = _native_update_grams_and_norms(
            model,
            batches,
            device=device,
            capture_block_group_size=int(capture_block_group_size),
        )
        return {
            "cls_grams": captured["cls_grams"],
            "patch_grams": captured["patch_grams"],
            "sample_count": int(captured["cls_grams"][0].shape[0]),
        }
    finally:
        model.to(original_device)
        model.train(was_training)


def _trajectory_cka_summary(matrix: Sequence[Sequence[float]]) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    diagonal = np.diag(values)
    off_diagonal = values[~np.eye(values.shape[0], dtype=bool)]

    def finite_mean(items: np.ndarray) -> float:
        finite = items[np.isfinite(items)]
        return float(finite.mean()) if finite.size else float("nan")

    same_index = finite_mean(diagonal)
    off_index = finite_mean(off_diagonal)
    return {
        "same_index_mean": same_index,
        "off_index_mean": off_index,
        "alignment_margin": float(same_index - off_index),
    }


def block_update_cka_against_reference(
    reference: Mapping[str, Any],
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    capture_block_group_size: int = 3,
) -> dict[str, Any]:
    """Measure one Target epoch against a fixed Source block-update CKA reference."""

    original_device = next(model.parameters()).device
    was_training = bool(model.training)
    try:
        captured = _native_update_grams_and_norms(
            model,
            batches,
            device=device,
            capture_block_group_size=int(capture_block_group_size),
        )
        cls_matrix = _pairwise_gram_cka_matrix(
            reference["cls_grams"], captured["cls_grams"]
        )
        patch_matrix = _pairwise_gram_cka_matrix(
            reference["patch_grams"], captured["patch_grams"]
        )
    finally:
        model.to(original_device)
        model.train(was_training)

    cls_summary = _trajectory_cka_summary(cls_matrix)
    patch_summary = _trajectory_cka_summary(patch_matrix)
    return {
        "same_task_trajectory_sample_count": int(reference["sample_count"]),
        "same_task_trajectory_cls_debiased_cka_12x12": cls_matrix,
        "same_task_trajectory_patch_debiased_cka_12x12": patch_matrix,
        "same_task_trajectory_cls_same_index_mean": cls_summary["same_index_mean"],
        "same_task_trajectory_cls_off_index_mean": cls_summary["off_index_mean"],
        "same_task_trajectory_cls_alignment_margin": cls_summary["alignment_margin"],
        "same_task_trajectory_patch_same_index_mean": patch_summary["same_index_mean"],
        "same_task_trajectory_patch_off_index_mean": patch_summary["off_index_mean"],
        "same_task_trajectory_patch_alignment_margin": patch_summary["alignment_margin"],
    }

def _run_block_window(
    blocks: Sequence[nn.Module],
    start: int,
    width: int,
    block_input: torch.Tensor,
) -> torch.Tensor:
    value = block_input
    for block in blocks[int(start) : int(start) + int(width)]:
        value, _taps = block.forward_with_measurement_intermediates(value)
    return value - block_input


def _run_block_windows(
    blocks: Sequence[nn.Module],
    start: int,
    widths: Sequence[int],
    block_input: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Evaluate all requested prefix widths with one donor-window traversal.

    This is exactly equivalent to repeated ``_run_block_window`` calls for the
    same start/input, but shared prefixes are executed only once.
    """

    start_index = int(start)
    requested = sorted({int(width) for width in widths})
    if not requested or requested[0] < 1:
        raise ValueError("DiR block-window widths must be positive")
    maximum_width = requested[-1]
    if start_index < 0 or start_index + maximum_width > len(blocks):
        raise ValueError("DiR block-window request exceeds donor depth")
    requested_set = set(requested)
    value = block_input
    outputs: dict[int, torch.Tensor] = {}
    for offset, block in enumerate(
        blocks[start_index : start_index + maximum_width], start=1
    ):
        value, _taps = block.forward_with_measurement_intermediates(value)
        if offset in requested_set:
            outputs[offset] = value - block_input
    if set(outputs) != requested_set:
        raise RuntimeError("DiR shared-prefix block-window evaluation is incomplete")
    return outputs


def _weighted_paired_metric_vector(
    value_groups: Sequence[np.ndarray],
    weight_groups: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Average paired metrics only over finite positions with positive support."""

    if not value_groups or len(value_groups) != len(weight_groups):
        raise ValueError("paired metric aggregation requires matched nonempty groups")
    values = np.concatenate(
        [np.asarray(group, dtype=np.float64) for group in value_groups], axis=0
    )
    weights = np.concatenate(
        [np.asarray(group, dtype=np.float64) for group in weight_groups], axis=0
    )
    if values.shape != weights.shape or values.ndim != 2:
        raise ValueError("paired metric values and weights must be matching 2D arrays")
    valid_contributions = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    weighted_sum = np.where(valid_contributions, values * weights, 0.0).sum(axis=0)
    valid_weight = np.where(valid_contributions, weights, 0.0).sum(axis=0)
    aggregate = np.full(valid_weight.shape, np.nan, dtype=np.float64)
    np.divide(weighted_sum, valid_weight, out=aggregate, where=valid_weight > 0)
    valid_positions = valid_weight > 0
    return {
        "aggregate": aggregate,
        "valid_weight": valid_weight,
        "valid_positions": valid_positions,
        "invalid_contribution_count": int((~valid_contributions).sum()),
    }


def _direct_condition_storage(
    depth: int,
    widths: Sequence[int],
    modes: Sequence[str],
    metric_names: Sequence[str],
    paired_count_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "debiased": {
            mode: {width: [None] * (depth - width + 1) for width in widths}
            for mode in modes
        },
        "biased": {
            mode: {width: [None] * (depth - width + 1) for width in widths}
            for mode in modes
        },
        "paired": {
            mode: {
                name: {width: [None] * (depth - width + 1) for width in widths}
                for name in metric_names
            }
            for mode in modes
        },
        "paired_counts": {
            mode: {
                name: {width: [None] * (depth - width + 1) for width in widths}
                for name in paired_count_names
            }
            for mode in modes
        },
        "signal": {
            mode: {
                side: {width: [None] * (depth - width + 1) for width in widths}
                for side in ("left", "right")
            }
            for mode in modes
        },
        "variation": {
            mode: {
                side: {width: [None] * (depth - width + 1) for width in widths}
                for side in ("left", "right")
            }
            for mode in modes
        },
    }


def _capture_direct_receiver_inputs(
    receiver_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    receiver_indices: Sequence[int],
    *,
    device: torch.device,
) -> dict[int, list[torch.Tensor]]:
    input_chunks = {int(index): [] for index in receiver_indices}
    capture_points = [f"block_{int(index):02d}_input" for index in receiver_indices]
    for images_cpu, _labels, _ids in batches:
        logits, taps = forward_with_capture_and_interventions(
            receiver_model,
            images_cpu.to(device),
            capture_points=capture_points,
        )
        for receiver_index in receiver_indices:
            input_chunks[int(receiver_index)].append(
                taps[f"block_{int(receiver_index):02d}_input"].detach().cpu()
            )
        del taps, logits
    return input_chunks


def _new_direct_width_state(
    valid_widths: Sequence[int],
    modes: Sequence[str],
    metric_names: Sequence[str],
    paired_count_names: Sequence[str],
) -> dict[int, dict[str, Any]]:
    return {
        int(width): {
            "left_grams": {mode: [] for mode in modes},
            "right_grams": {mode: [] for mode in modes},
            "paired": {
                mode: {name: [] for name in metric_names} for mode in modes
            },
            "paired_counts": {
                mode: {name: [] for name in paired_count_names} for mode in modes
            },
            "signal": {mode: {"left": [], "right": []} for mode in modes},
            "variation": {mode: {"left": [], "right": []} for mode in modes},
        }
        for width in valid_widths
    }


def _direct_window_statistics(
    left_parts: Mapping[str, Sequence[torch.Tensor]],
    right_parts: Mapping[str, Sequence[torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, dict[str, Any]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    base_modes = ("cls", "patch")
    left_features = {
        mode: torch.cat(list(left_parts[mode]), dim=0) for mode in base_modes
    }
    right_features = {
        mode: torch.cat(list(right_parts[mode]), dim=0) for mode in base_modes
    }
    left_grams = {
        mode: _gram_from_feature_chunks(left_parts[mode], device=device)
        for mode in base_modes
    }
    right_grams = {
        mode: _gram_from_feature_chunks(right_parts[mode], device=device)
        for mode in base_modes
    }
    left_grams["full_token"] = left_grams["cls"] + left_grams["patch"]
    right_grams["full_token"] = right_grams["cls"] + right_grams["patch"]

    metrics_by_mode = {
        "cls": paired_output_metrics(left_features["cls"], right_features["cls"]),
        "patch": paired_output_metrics(
            left_features["patch"], right_features["patch"]
        ),
        "full_token": _paired_output_metrics_from_components(
            [left_features["cls"], left_features["patch"]],
            [right_features["cls"], right_features["patch"]],
        ),
    }
    signal_by_mode = {
        "cls": {
            "left": float(left_features["cls"].float().square().mean().sqrt()),
            "right": float(right_features["cls"].float().square().mean().sqrt()),
        },
        "patch": {
            "left": float(left_features["patch"].float().square().mean().sqrt()),
            "right": float(right_features["patch"].float().square().mean().sqrt()),
        },
        "full_token": {
            "left": _component_signal_rms(
                [left_features["cls"], left_features["patch"]]
            ),
            "right": _component_signal_rms(
                [right_features["cls"], right_features["patch"]]
            ),
        },
    }
    variation_by_mode = {
        "cls": {
            "left": sample_variation_rms(left_features["cls"]),
            "right": sample_variation_rms(right_features["cls"]),
        },
        "patch": {
            "left": sample_variation_rms(left_features["patch"]),
            "right": sample_variation_rms(right_features["patch"]),
        },
        "full_token": {
            "left": _component_sample_variation_rms(
                [left_features["cls"], left_features["patch"]]
            ),
            "right": _component_sample_variation_rms(
                [right_features["cls"], right_features["patch"]]
            ),
        },
    }
    return (
        left_grams,
        right_grams,
        metrics_by_mode,
        signal_by_mode,
        variation_by_mode,
    )


def _accumulate_direct_receiver_windows(
    *,
    left_model: nn.Module,
    right_model: nn.Module,
    input_chunks: Sequence[torch.Tensor],
    width_state: dict[int, dict[str, Any]],
    valid_widths: Sequence[int],
    depth: int,
    device: torch.device,
    modes: Sequence[str],
    metric_names: Sequence[str],
    paired_count_names: Sequence[str],
) -> int:
    base_modes = ("cls", "patch")
    nonfinite_count = 0
    for start_index in range(depth):
        start_widths = [
            width for width in valid_widths if start_index < depth - width + 1
        ]
        if not start_widths:
            continue

        # Full-token features are virtual concatenations of these two disjoint parts.
        left_parts_by_width = {
            width: {mode: [] for mode in base_modes} for width in start_widths
        }
        right_parts_by_width = {
            width: {mode: [] for mode in base_modes} for width in start_widths
        }
        for native_input_cpu in input_chunks:
            native_input = native_input_cpu.to(device)
            left_responses = _run_block_windows(
                left_model.transformer_blocks,
                start_index,
                start_widths,
                native_input,
            )
            right_responses = _run_block_windows(
                right_model.transformer_blocks,
                start_index,
                start_widths,
                native_input,
            )
            for width in start_widths:
                for mode in base_modes:
                    left_parts_by_width[width][mode].append(
                        _feature_view(left_responses[width], mode).detach().cpu()
                    )
                    right_parts_by_width[width][mode].append(
                        _feature_view(right_responses[width], mode).detach().cpu()
                    )
            del native_input, left_responses, right_responses

        for width in start_widths:
            state = width_state[width]
            (
                left_grams,
                right_grams,
                metrics_by_mode,
                signal_by_mode,
                variation_by_mode,
            ) = _direct_window_statistics(
                left_parts_by_width[width],
                right_parts_by_width[width],
                device=device,
            )
            for mode in modes:
                state["left_grams"][mode].append(left_grams[mode])
                state["right_grams"][mode].append(right_grams[mode])
                metrics = metrics_by_mode[mode]
                nonfinite_count += int(metrics["nonfinite_sample_count"])
                for name in metric_names:
                    state["paired"][mode][name].append(float(metrics[name]))
                for name in paired_count_names:
                    state["paired_counts"][mode][name].append(int(metrics[name]))
                for side in ("left", "right"):
                    state["signal"][mode][side].append(
                        float(signal_by_mode[mode][side])
                    )
                    state["variation"][mode][side].append(
                        float(variation_by_mode[mode][side])
                    )
            del left_grams, right_grams, metrics_by_mode
            del signal_by_mode, variation_by_mode
        del left_parts_by_width, right_parts_by_width
    return nonfinite_count


def _store_direct_receiver_results(
    storage: dict[str, Any],
    width_state: Mapping[int, Mapping[str, Any]],
    *,
    receiver_index: int,
    valid_widths: Sequence[int],
    depth: int,
    modes: Sequence[str],
    metric_names: Sequence[str],
    paired_count_names: Sequence[str],
) -> None:
    for width in valid_widths:
        state = width_state[width]
        expected_count = depth - width + 1
        for mode in modes:
            if (
                len(state["left_grams"][mode]) != expected_count
                or len(state["right_grams"][mode]) != expected_count
            ):
                raise RuntimeError(
                    "DiR shared-prefix direct window incomplete "
                    f"mode={mode} width={width}"
                )
            storage["debiased"][mode][width][receiver_index] = (
                _pairwise_gram_cka_matrix(
                    state["left_grams"][mode],
                    state["right_grams"][mode],
                    invalid_as_nan=True,
                )
            )
            storage["biased"][mode][width][receiver_index] = (
                _pairwise_biased_gram_cka_matrix(
                    state["left_grams"][mode],
                    state["right_grams"][mode],
                )
            )
            for name in metric_names:
                storage["paired"][mode][name][width][receiver_index] = state[
                    "paired"
                ][mode][name]
            for name in paired_count_names:
                storage["paired_counts"][mode][name][width][receiver_index] = state[
                    "paired_counts"
                ][mode][name]
            for side in ("left", "right"):
                storage["signal"][mode][side][width][receiver_index] = state[
                    "signal"
                ][mode][side]
                storage["variation"][mode][side][width][receiver_index] = state[
                    "variation"
                ][mode][side]


def _collect_direct_condition(
    receiver_model: nn.Module,
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    depth: int,
    widths: Sequence[int],
    modes: Sequence[str],
    metric_names: Sequence[str],
    paired_count_names: Sequence[str],
    capture_block_group_size: int,
) -> tuple[dict[str, Any], int]:
    storage = _direct_condition_storage(
        depth,
        widths,
        modes,
        metric_names,
        paired_count_names,
    )
    nonfinite_count = 0
    group_size = max(1, min(int(capture_block_group_size), depth))
    for group_start in range(0, depth, group_size):
        receiver_indices = list(range(group_start, min(depth, group_start + group_size)))
        input_chunks_by_receiver = _capture_direct_receiver_inputs(
            receiver_model,
            batches,
            receiver_indices,
            device=device,
        )
        for receiver_index in receiver_indices:
            valid_widths = [
                width for width in widths if receiver_index < depth - width + 1
            ]
            width_state = _new_direct_width_state(
                valid_widths,
                modes,
                metric_names,
                paired_count_names,
            )
            nonfinite_count += _accumulate_direct_receiver_windows(
                left_model=left_model,
                right_model=right_model,
                input_chunks=input_chunks_by_receiver[receiver_index],
                width_state=width_state,
                valid_widths=valid_widths,
                depth=depth,
                device=device,
                modes=modes,
                metric_names=metric_names,
                paired_count_names=paired_count_names,
            )
            _store_direct_receiver_results(
                storage,
                width_state,
                receiver_index=receiver_index,
                valid_widths=valid_widths,
                depth=depth,
                modes=modes,
                metric_names=metric_names,
                paired_count_names=paired_count_names,
            )
            del width_state
        del input_chunks_by_receiver
    return storage, nonfinite_count


def _publish_direct_condition(
    condition_name: str,
    condition_storage: Mapping[str, Any],
    *,
    conditioned_debiased: dict[str, Any],
    conditioned_biased: dict[str, Any],
    conditioned_paired: dict[str, Any],
    conditioned_paired_counts: dict[str, Any],
    conditioned_signal: dict[str, Any],
    conditioned_variation: dict[str, Any],
    modes: Sequence[str],
    widths: Sequence[int],
    metric_names: Sequence[str],
    paired_count_names: Sequence[str],
) -> None:
    for mode in modes:
        for width in widths:
            for target, source_name in (
                (conditioned_debiased, "debiased"),
                (conditioned_biased, "biased"),
            ):
                matrices = condition_storage[source_name][mode][width]
                if any(value is None for value in matrices):
                    raise RuntimeError(
                        "DiR direct function matrix incomplete "
                        f"mode={mode} width={width}"
                    )
                target[condition_name][mode][str(width)] = [
                    value for value in matrices if value is not None
                ]
            for name in metric_names:
                vectors = condition_storage["paired"][mode][name][width]
                if any(value is None for value in vectors):
                    raise RuntimeError(
                        "DiR direct paired metric incomplete "
                        f"mode={mode} metric={name} width={width}"
                    )
                conditioned_paired[condition_name][mode][name][str(width)] = [
                    value for value in vectors if value is not None
                ]
            for name in paired_count_names:
                vectors = condition_storage["paired_counts"][mode][name][width]
                if any(value is None for value in vectors):
                    raise RuntimeError(
                        "DiR direct paired count incomplete "
                        f"mode={mode} metric={name} width={width}"
                    )
                conditioned_paired_counts[condition_name][mode][name][str(width)] = [
                    value for value in vectors if value is not None
                ]
            for side in ("left", "right"):
                signal_vectors = condition_storage["signal"][mode][side][width]
                if any(value is None for value in signal_vectors):
                    raise RuntimeError(
                        "DiR direct signal incomplete "
                        f"mode={mode} side={side} width={width}"
                    )
                conditioned_signal[condition_name][mode][side][str(width)] = [
                    value for value in signal_vectors if value is not None
                ]
                variation_vectors = condition_storage["variation"][mode][side][width]
                if any(value is None for value in variation_vectors):
                    raise RuntimeError(
                        "DiR direct sample variation incomplete "
                        f"mode={mode} side={side} width={width}"
                    )
                conditioned_variation[condition_name][mode][side][str(width)] = [
                    value for value in variation_vectors if value is not None
                ]


def direct_block_function_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    window_widths: Sequence[int] = (1, 2, 3, 4, 6, 12),
    include_single_if_missing: bool = True,
    minimum_signal_rms_absolute: float = 1e-8,
    minimum_signal_rms_relative_to_median: float = 0.05,
    capture_block_group_size: int = 3,
) -> dict[str, Any]:
    """Compare donor blocks/windows on identical receiver-native tensors."""

    depth = len(left_model.transformer_blocks)
    if depth != len(right_model.transformer_blocks):
        raise ValueError("DiR direct function comparison requires equal depth")
    widths = sorted({int(value) for value in window_widths if 1 <= int(value) <= depth})
    if bool(include_single_if_missing) and 1 not in widths:
        widths.insert(0, 1)
    if not widths:
        raise ValueError("DiR direct function comparison requires at least one window width")

    left_model.eval().to(device)
    right_model.eval().to(device)
    modes = ("full_token", "cls", "patch")
    metric_names = (
        "signed_cosine_mean",
        "normalized_l2_mean",
        "symmetric_norm_ratio_mean",
    )
    paired_count_names = (
        "cosine_valid_sample_count",
        "distance_scale_valid_sample_count",
        "nonfinite_sample_count",
        "total_sample_count",
    )
    conditioned_debiased = {
        "left": {mode: {} for mode in modes},
        "right": {mode: {} for mode in modes},
    }
    conditioned_biased = {
        "left": {mode: {} for mode in modes},
        "right": {mode: {} for mode in modes},
    }
    conditioned_paired = {
        "left": {mode: {name: {} for name in metric_names} for mode in modes},
        "right": {mode: {name: {} for name in metric_names} for mode in modes},
    }
    conditioned_paired_counts = {
        "left": {mode: {name: {} for name in paired_count_names} for mode in modes},
        "right": {mode: {name: {} for name in paired_count_names} for mode in modes},
    }
    conditioned_signal = {
        "left": {mode: {"left": {}, "right": {}} for mode in modes},
        "right": {mode: {"left": {}, "right": {}} for mode in modes},
    }
    conditioned_variation = {
        "left": {mode: {"left": {}, "right": {}} for mode in modes},
        "right": {mode: {"left": {}, "right": {}} for mode in modes},
    }
    nonfinite_sample_count_total = 0

    with torch.no_grad():
        for condition_name, receiver_model in (
            ("left", left_model),
            ("right", right_model),
        ):
            condition_storage, nonfinite_count = _collect_direct_condition(
                receiver_model,
                left_model,
                right_model,
                batches,
                device=device,
                depth=depth,
                widths=widths,
                modes=modes,
                metric_names=metric_names,
                paired_count_names=paired_count_names,
                capture_block_group_size=capture_block_group_size,
            )
            nonfinite_sample_count_total += nonfinite_count
            _publish_direct_condition(
                condition_name,
                condition_storage,
                conditioned_debiased=conditioned_debiased,
                conditioned_biased=conditioned_biased,
                conditioned_paired=conditioned_paired,
                conditioned_paired_counts=conditioned_paired_counts,
                conditioned_signal=conditioned_signal,
                conditioned_variation=conditioned_variation,
                modes=modes,
                widths=widths,
                metric_names=metric_names,
                paired_count_names=paired_count_names,
            )

    return _aggregate_direct_alignment(
        conditioned_debiased=conditioned_debiased,
        conditioned_biased=conditioned_biased,
        conditioned_paired=conditioned_paired,
        conditioned_paired_counts=conditioned_paired_counts,
        conditioned_signal=conditioned_signal,
        conditioned_variation=conditioned_variation,
        modes=modes,
        metric_names=metric_names,
        paired_count_names=paired_count_names,
        widths=widths,
        minimum_signal_rms_absolute=minimum_signal_rms_absolute,
        minimum_signal_rms_relative_to_median=minimum_signal_rms_relative_to_median,
        nonfinite_sample_count_total=nonfinite_sample_count_total,
        capture_block_group_size=capture_block_group_size,
        depth=depth,
    )


def _aggregate_direct_alignment(
    *,
    conditioned_debiased: dict[str, Any],
    conditioned_biased: dict[str, Any],
    conditioned_paired: dict[str, Any],
    conditioned_paired_counts: dict[str, Any],
    conditioned_signal: dict[str, Any],
    conditioned_variation: dict[str, Any],
    modes: Sequence[str],
    metric_names: Sequence[str],
    paired_count_names: Sequence[str],
    widths: Sequence[int],
    minimum_signal_rms_absolute: float,
    minimum_signal_rms_relative_to_median: float,
    nonfinite_sample_count_total: int,
    capture_block_group_size: int,
    depth: int,
) -> dict[str, Any]:
    """Aggregate receiver-conditioned direct measurements into report matrices."""
    conditioned_cka_validity_masks: dict[str, dict[str, dict[str, list[list[list[bool]]]]]] = {
        "left": {mode: {} for mode in modes},
        "right": {mode: {} for mode in modes},
    }
    for condition_name in ("left", "right"):
        for mode in modes:
            for width in widths:
                key = str(width)
                matrices = conditioned_debiased[condition_name][mode][key]
                masks: list[list[list[bool]]] = []
                left_signal_rows = conditioned_signal[condition_name][mode]["left"][key]
                right_signal_rows = conditioned_signal[condition_name][mode]["right"][key]
                left_variation_rows = conditioned_variation[condition_name][mode]["left"][key]
                right_variation_rows = conditioned_variation[condition_name][mode]["right"][key]
                if not (
                    len(matrices)
                    == len(left_signal_rows)
                    == len(right_signal_rows)
                    == len(left_variation_rows)
                    == len(right_variation_rows)
                ):
                    raise RuntimeError(
                        f"DiR direct condition validity shape mismatch condition={condition_name} mode={mode} width={width}"
                    )
                for matrix, left_signal_values, right_signal_values, left_variation_values, right_variation_values in zip(
                    matrices,
                    left_signal_rows,
                    right_signal_rows,
                    left_variation_rows,
                    right_variation_rows,
                ):
                    left_validity = combined_signal_variation_validity(
                        left_signal_values,
                        left_variation_values,
                        absolute_minimum=float(minimum_signal_rms_absolute),
                        relative_to_median=float(minimum_signal_rms_relative_to_median),
                    )
                    right_validity = combined_signal_variation_validity(
                        right_signal_values,
                        right_variation_values,
                        absolute_minimum=float(minimum_signal_rms_absolute),
                        relative_to_median=float(minimum_signal_rms_relative_to_median),
                    )
                    signal_mask = np.asarray(
                        outer_validity_mask(
                            left_validity["valid_by_block"],
                            right_validity["valid_by_block"],
                        ),
                        dtype=bool,
                    )
                    finite_mask = np.isfinite(np.asarray(matrix, dtype=np.float64))
                    masks.append((signal_mask & finite_mask).tolist())
                conditioned_cka_validity_masks[condition_name][mode][key] = masks

    def average_matrices(
        source: Mapping[str, Any],
        *,
        condition_name: str,
    ) -> dict[str, dict[str, list[list[float]]]]:
        output = {mode: {} for mode in modes}
        for mode in modes:
            for width in widths:
                key = str(width)
                matrices = []
                for value, mask in zip(
                    source[mode][key],
                    conditioned_cka_validity_masks[condition_name][mode][key],
                ):
                    array = np.asarray(value, dtype=np.float64).copy()
                    valid = np.asarray(mask, dtype=bool)
                    if array.shape != valid.shape:
                        raise RuntimeError(
                            f"DiR direct condition matrix/mask mismatch condition={condition_name} mode={mode} width={width}"
                        )
                    array[~valid] = np.nan
                    matrices.append(array)
                output[mode][key] = _finite_elementwise_mean(matrices).tolist()
        return output

    source_debiased = average_matrices(
        conditioned_debiased["left"], condition_name="left"
    )
    target_debiased = average_matrices(
        conditioned_debiased["right"], condition_name="right"
    )
    source_biased = average_matrices(
        conditioned_biased["left"], condition_name="left"
    )
    target_biased = average_matrices(
        conditioned_biased["right"], condition_name="right"
    )
    bidirectional_debiased = {mode: {} for mode in modes}
    bidirectional_biased = {mode: {} for mode in modes}
    paired_bidirectional = {mode: {name: {} for name in metric_names} for mode in modes}
    paired_cosine_valid_sample_count = {mode: {} for mode in modes}
    paired_cosine_validity_mask = {mode: {} for mode in modes}
    paired_cosine_inconclusive_position_count = 0
    paired_cosine_total_position_count = 0
    paired_cosine_invalid_contribution_count = 0
    low_signal: dict[str, Any] = {mode: {} for mode in modes}
    validity_masks: dict[str, Any] = {}
    for mode in modes:
        for width in widths:
            key = str(width)
            bidirectional_debiased[mode][key] = _finite_elementwise_mean(
                [
                    np.asarray(source_debiased[mode][key], dtype=np.float64),
                    np.asarray(target_debiased[mode][key], dtype=np.float64),
                ]
            ).tolist()
            bidirectional_biased[mode][key] = _finite_elementwise_mean(
                [
                    np.asarray(source_biased[mode][key], dtype=np.float64),
                    np.asarray(target_biased[mode][key], dtype=np.float64),
                ]
            ).tolist()
            for name in metric_names:
                source_values = np.asarray(
                    conditioned_paired["left"][mode][name][key], dtype=np.float64
                )
                target_values = np.asarray(
                    conditioned_paired["right"][mode][name][key], dtype=np.float64
                )
                if name == "signed_cosine_mean":
                    source_weights = np.asarray(
                        conditioned_paired_counts["left"][mode][
                            "cosine_valid_sample_count"
                        ][key],
                        dtype=np.float64,
                    )
                    target_weights = np.asarray(
                        conditioned_paired_counts["right"][mode][
                            "cosine_valid_sample_count"
                        ][key],
                        dtype=np.float64,
                    )
                    aggregate = _weighted_paired_metric_vector(
                        (source_values, target_values),
                        (source_weights, target_weights),
                    )
                    aggregated = aggregate["aggregate"]
                    valid_weight = aggregate["valid_weight"]
                    valid_positions = aggregate["valid_positions"]
                    paired_bidirectional[mode][name][key] = aggregated.tolist()
                    paired_cosine_valid_sample_count[mode][key] = (
                        valid_weight.astype(np.int64).tolist()
                    )
                    paired_cosine_validity_mask[mode][key] = valid_positions.tolist()
                    paired_cosine_inconclusive_position_count += int((~valid_positions).sum())
                    paired_cosine_total_position_count += int(valid_positions.size)
                    paired_cosine_invalid_contribution_count += int(
                        aggregate["invalid_contribution_count"]
                    )
                else:
                    count_name = "distance_scale_valid_sample_count"
                    source_weights = np.asarray(
                        conditioned_paired_counts["left"][mode][count_name][key],
                        dtype=np.float64,
                    )
                    target_weights = np.asarray(
                        conditioned_paired_counts["right"][mode][count_name][key],
                        dtype=np.float64,
                    )
                    aggregate = _weighted_paired_metric_vector(
                        (source_values, target_values),
                        (source_weights, target_weights),
                    )
                    paired_bidirectional[mode][name][key] = aggregate["aggregate"].tolist()
            left_signal = np.mean(
                [
                    np.asarray(conditioned_signal[condition][mode]["left"][key], dtype=np.float64)
                    for condition in ("left", "right")
                ],
                axis=(0, 1),
            )
            right_signal = np.mean(
                [
                    np.asarray(conditioned_signal[condition][mode]["right"][key], dtype=np.float64)
                    for condition in ("left", "right")
                ],
                axis=(0, 1),
            )
            left_variation = np.mean(
                [
                    np.asarray(conditioned_variation[condition][mode]["left"][key], dtype=np.float64)
                    for condition in ("left", "right")
                ],
                axis=(0, 1),
            )
            right_variation = np.mean(
                [
                    np.asarray(conditioned_variation[condition][mode]["right"][key], dtype=np.float64)
                    for condition in ("left", "right")
                ],
                axis=(0, 1),
            )
            left_validity = combined_signal_variation_validity(
                left_signal.tolist(),
                left_variation.tolist(),
                absolute_minimum=float(minimum_signal_rms_absolute),
                relative_to_median=float(minimum_signal_rms_relative_to_median),
            )
            right_validity = combined_signal_variation_validity(
                right_signal.tolist(),
                right_variation.tolist(),
                absolute_minimum=float(minimum_signal_rms_absolute),
                relative_to_median=float(minimum_signal_rms_relative_to_median),
            )
            low_signal[mode][key] = {"left": left_validity, "right": right_validity}
            matrix_key = f"window_{key}_{mode}_debiased_cka"
            # Every receiver-conditioned contribution was already masked by its
            # own signal/variation validity before aggregation. Therefore a
            # finite cell here means that at least one scientifically valid
            # conditioned contribution survived. Re-thresholding an average
            # signal vector can incorrectly erase such cells, especially when
            # only one receiver condition is estimable.
            cka_finite = np.isfinite(
                np.asarray(bidirectional_debiased[mode][key], dtype=np.float64)
            )
            validity_masks[matrix_key] = cka_finite.tolist()

    result = {
        "window_widths": widths,
        "source_conditioned_debiased_cka": source_debiased,
        "target_conditioned_debiased_cka": target_debiased,
        "bidirectional_mean_debiased_cka": bidirectional_debiased,
        "same_index_paired_output_metrics": paired_bidirectional,
        "paired_output_cosine_valid_sample_count": paired_cosine_valid_sample_count,
        "paired_output_cosine_validity_mask": paired_cosine_validity_mask,
        "paired_output_cosine_inconclusive_position_count": int(
            paired_cosine_inconclusive_position_count
        ),
        "paired_output_cosine_total_position_count": int(
            paired_cosine_total_position_count
        ),
        "paired_output_cosine_invalid_contribution_count": int(
            paired_cosine_invalid_contribution_count
        ),
        "cosine_measurement_status": (
            "completed"
            if int(paired_cosine_inconclusive_position_count) == 0
            else "cosine_inconclusive"
        ),
        "auxiliary_biased_cka": {
            "source_conditioned": source_biased,
            "target_conditioned": target_biased,
            "bidirectional_mean": bidirectional_biased,
        },
        "single_bidirectional_mean_full_token_debiased_cka_12x12": bidirectional_debiased["full_token"].get("1", []),
        "single_bidirectional_mean_cls_debiased_cka_12x12": bidirectional_debiased["cls"].get("1", []),
        "single_bidirectional_mean_patch_debiased_cka_12x12": bidirectional_debiased["patch"].get("1", []),
        "adjacent_bidirectional_mean_full_token_debiased_cka_11x11": bidirectional_debiased["full_token"].get("2", []),
        "validity_masks": {
            "single_bidirectional_mean_full_token_debiased_cka_12x12": validity_masks.get("window_1_full_token_debiased_cka", []),
            "single_bidirectional_mean_cls_debiased_cka_12x12": validity_masks.get("window_1_cls_debiased_cka", []),
            "single_bidirectional_mean_patch_debiased_cka_12x12": validity_masks.get("window_1_patch_debiased_cka", []),
            "adjacent_bidirectional_mean_full_token_debiased_cka_11x11": validity_masks.get("window_2_full_token_debiased_cka", []),
        },
        "low_signal": low_signal,
        "receiver_conditioned_debiased_matrices": conditioned_debiased,
        "receiver_conditioned_debiased_validity_masks": conditioned_cka_validity_masks,
        "cka_contract": "U_centered_debiased_primary_biased_auxiliary",
        "paired_output_contract": "exact_same_index_signed_cosine_normalized_l2_and_symmetric_norm_ratio",
        "token_contract": "CLS_and_patch_primary_views_reported_separately_full_token_not_allowed_to_hide_CLS",
        "token_stage_contract": "native_block_window_residual_output_space_before_next_block_pre_norm",
        "primary_metrics": [
            "single_bidirectional_mean_cls_debiased_cka_12x12",
            "single_bidirectional_mean_patch_debiased_cka_12x12",
        ],
        "full_token_role": "auxiliary_only",
        "paired_output_nonfinite_sample_count_total": int(
            nonfinite_sample_count_total
        ),
        # Filled below from the independent CLS/patch primary-view masks.
        # Auxiliary cosine validity never determines primary CKA completion.
        "measurement_status": "pending_primary_view_validation",
        "degenerate_cka_contract": "nonconstant_sample_variation_required_and_invalid_cells_masked_inconclusive",
        "memory_policy": "bounded_receiver_capture_cls_patch_only_full_token_derived_from_component_grams_and_statistics_u_centering_reused_per_pairwise_matrix",
        "capture_block_group_size": max(1, min(int(capture_block_group_size), depth)),
    }
    primary_view_status: dict[str, str] = {}
    for key in result["primary_metrics"]:
        matrix = np.asarray(result[key], dtype=np.float64)
        mask = np.asarray(result["validity_masks"].get(key, []), dtype=bool)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            primary_view_status[key] = "inconclusive_invalid_primary_matrix"
        elif mask.shape != matrix.shape:
            primary_view_status[key] = "inconclusive_invalid_primary_mask"
        elif bool((np.diag(mask) & np.isfinite(np.diag(matrix))).any()):
            primary_view_status[key] = "valid"
        else:
            primary_view_status[key] = "inconclusive_no_valid_diagonal"
    valid_primary_view_count = sum(
        value == "valid" for value in primary_view_status.values()
    )
    result["primary_view_status"] = primary_view_status
    if int(nonfinite_sample_count_total) > 0:
        result["measurement_status"] = "inconclusive_nonfinite_paired_outputs"
    elif valid_primary_view_count == len(primary_view_status):
        result["measurement_status"] = "completed"
    elif valid_primary_view_count > 0:
        result["measurement_status"] = "partial_primary_views"
    else:
        result["measurement_status"] = "inconclusive_no_valid_primary_cka"
    return result
