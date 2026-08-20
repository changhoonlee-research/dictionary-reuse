"""Supplementary atom and block intervention measurements."""

from __future__ import annotations


# Atom interventions
import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ..interventions import forward_with_capture_and_interventions
from ..measurements.ablation import atom_output_mask
from ..model.dictionary_operator import iter_dictionary_layers

def _block_index_from_layer_name(name: str) -> int | None:
    match = re.search(r"transformer_blocks[._](\d+)", name)
    return int(match.group(1)) if match else None


def atom_group_ablation(
    source_model: nn.Module,
    target_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    source_active_masks: Mapping[str, torch.Tensor],
    maximum_relative_mass_mismatch: float = 0.10,
) -> dict[str, Any]:
    """Compare shared/new atom groups under count- and mass-matched removal.

    A block is scientifically comparable only when both shared and new Target-active
    atoms exist. Empty matched selections are *not* interpreted as zero effect: that
    would confuse "nothing was ablated" with "the ablated atoms had no effect".
    Invalid blocks therefore retain their audit metadata but emit NaN effects and an
    inconclusive status, while valid blocks alone contribute to shared-vs-new results.
    """

    _ = source_model  # Source identity is represented by ``source_active_masks``.
    target_layers = dict(iter_dictionary_layers(target_model))
    depth = len(target_model.transformer_blocks)
    layer_records: list[dict[str, Any]] = []
    for name, layer in target_layers.items():
        block = _block_index_from_layer_name(name)
        if block is None or name not in source_active_masks:
            continue
        source_active = source_active_masks[name].to(
            device=layer.coefficient_magnitude.device, dtype=torch.bool
        )
        method = getattr(layer, "_actual_forward_support_mask_for_metrics", None)
        target_active = (
            method(device=layer.coefficient_magnitude.device).detach().bool()
            if callable(method)
            else layer.coefficient_commit_mask.detach().bool()
        )
        mass = layer.atom_contribution_masses().detach().float()
        layer_records.append(
            {
                "name": name,
                "layer": layer,
                "block": int(block),
                "mass": mass,
                "shared": source_active & target_active,
                "new": (~source_active) & target_active,
                "unused": ~target_active,
            }
        )
    if not layer_records:
        return {
            "available": False,
            "measurement_status": "inconclusive_no_transformer_dictionary_layers",
            "reason": "no_transformer_dictionary_layers",
        }

    def candidates(block_index: int, group: str) -> list[tuple[str, int, float]]:
        items: list[tuple[str, int, float]] = []
        for record in layer_records:
            if int(record["block"]) != int(block_index):
                continue
            indices = torch.nonzero(record[group], as_tuple=False).flatten()
            for index in indices.tolist():
                items.append(
                    (record["name"], int(index), float(record["mass"][index].cpu()))
                )
        return sorted(items, key=lambda item: (-item[2], item[0], item[1]))

    def select_by_mass(
        items: list[tuple[str, int, float]],
        target_mass: float,
    ) -> list[tuple[str, int, float]]:
        if not items or target_mass <= 0:
            return []
        cumulative = 0.0
        best_count = 0
        best_error = abs(float(target_mass))
        for count, item in enumerate(items, start=1):
            cumulative += float(item[2])
            error = abs(float(target_mass) - cumulative)
            if error < best_error:
                best_error = error
                best_count = count
        return items[:best_count]

    selections: dict[str, dict[int, dict[str, list[tuple[str, int, float]]]]] = {
        "equal_count": {},
        "equal_contribution_mass": {},
    }
    selection_audit: dict[str, Any] = {}
    for block_index in range(depth):
        shared = candidates(block_index, "shared")
        new = candidates(block_index, "new")
        unused = candidates(block_index, "unused")
        matched_count = min(len(shared), len(new))
        shared_mass = float(sum(item[2] for item in shared))
        new_mass = float(sum(item[2] for item in new))
        matched_mass = min(shared_mass, new_mass)

        count_status = (
            "completed"
            if matched_count > 0
            else "inconclusive_no_comparable_atoms"
        )
        selections["equal_count"][block_index] = {
            "shared": shared[:matched_count],
            "new": new[:matched_count],
            "unused": unused[:matched_count],
        }

        mass_shared = select_by_mass(shared, matched_mass)
        mass_new = select_by_mass(new, matched_mass)
        sanity_count = min(len(unused), max(len(mass_shared), len(mass_new)))
        selections["equal_contribution_mass"][block_index] = {
            "shared": mass_shared,
            "new": mass_new,
            "unused": unused[:sanity_count],
        }
        selected_shared_mass = float(sum(item[2] for item in mass_shared))
        selected_new_mass = float(sum(item[2] for item in mass_new))

        if matched_count == 0:
            mass_status = "inconclusive_no_comparable_atoms"
            relative_mass_mismatch = float("nan")
        elif matched_mass <= 0.0 or not mass_shared or not mass_new:
            mass_status = "inconclusive_no_positive_comparable_mass"
            relative_mass_mismatch = float("nan")
        else:
            selected_mass_reference = max(
                1e-12, 0.5 * (selected_shared_mass + selected_new_mass)
            )
            relative_mass_mismatch = (
                abs(selected_shared_mass - selected_new_mass)
                / selected_mass_reference
            )
            mass_status = (
                "completed"
                if relative_mass_mismatch <= float(maximum_relative_mass_mismatch)
                else "inconclusive_mass_mismatch"
            )

        selection_audit[str(block_index)] = {
            "available_counts": {
                "shared": len(shared),
                "new": len(new),
                "unused": len(unused),
            },
            "available_mass": {"shared": shared_mass, "new": new_mass},
            "matched_count": matched_count,
            "equal_count_status": count_status,
            "matched_contribution_mass_target": matched_mass,
            "selected_contribution_mass": {
                "shared": selected_shared_mass,
                "new": selected_new_mass,
            },
            "relative_selected_mass_mismatch": float(relative_mass_mismatch),
            "maximum_relative_mass_mismatch": float(maximum_relative_mass_mismatch),
            "equal_contribution_mass_status": mass_status,
        }

    def masks_from_selection(
        selection: list[tuple[str, int, float]],
    ) -> dict[str, torch.Tensor]:
        result = {
            name: torch.ones(int(layer.atom_count), dtype=torch.float32)
            for name, layer in target_layers.items()
        }
        for name, index, _mass in selection:
            result[name][int(index)] = 0.0
        return result

    policy_status_key = {
        "equal_count": "equal_count_status",
        "equal_contribution_mass": "equal_contribution_mass_status",
    }
    policy_valid_blocks = {
        policy: [
            block_index
            for block_index in range(depth)
            if selection_audit[str(block_index)][status_key] == "completed"
        ]
        for policy, status_key in policy_status_key.items()
    }
    valid_policy_block_count = sum(len(value) for value in policy_valid_blocks.values())
    total_policy_block_count = depth * len(policy_valid_blocks)
    if valid_policy_block_count == 0:
        measurement_status = "inconclusive_no_comparable_atoms"
    elif valid_policy_block_count < total_policy_block_count:
        measurement_status = "partial_comparable_blocks"
    else:
        measurement_status = "completed"

    target_model.eval().to(device)
    output: dict[str, Any] = {
        "available": True,
        "measurement_status": measurement_status,
        "selection_audit": selection_audit,
        "maximum_relative_mass_mismatch": float(maximum_relative_mass_mismatch),
        "valid_blocks_by_policy": {
            policy: list(indices) for policy, indices in policy_valid_blocks.items()
        },
        "valid_policy_block_count": int(valid_policy_block_count),
        "total_policy_block_count": int(total_policy_block_count),
        "comparison_contract": (
            "shared_vs_new_effects_are_reported_only_for_blocks_with_nonempty_"
            "comparable_selections;_invalid_blocks_are_NaN_not_numeric_zero"
        ),
        "policies": {},
        "baseline_forward_reuse": (
            "one_clean_pre_classifier_forward_per_batch_shared_by_all_atom_groups"
        ),
    }
    baseline_representations: list[torch.Tensor] = []
    if valid_policy_block_count > 0:
        with torch.no_grad():
            for images_cpu, _labels, _ids in batches:
                _base_logits, base = forward_with_capture_and_interventions(
                    target_model,
                    images_cpu.to(device),
                    capture_points=["pre_classifier"],
                )
                baseline_representations.append(base["pre_classifier"].detach().cpu())

    for policy in ("equal_count", "equal_contribution_mass"):
        valid_blocks = set(policy_valid_blocks[policy])
        policy_output: dict[str, Any] = {
            "shared": [],
            "new": [],
            "unused": [],
            "selected": {},
            "valid_block_mask": [index in valid_blocks for index in range(depth)],
        }
        for group_name in ("shared", "new", "unused"):
            policy_output["selected"][group_name] = []
            for block_index in range(depth):
                selection = selections[policy][block_index][group_name]
                status = selection_audit[str(block_index)][policy_status_key[policy]]

                # Shared/new are primary to this supplementary comparison. For an
                # invalid block, or an unavailable unused-atom sanity control, do
                # not run an identity forward and misreport the resulting zero.
                selection_is_valid = block_index in valid_blocks and bool(selection)
                if not selection_is_valid:
                    policy_output[group_name].append(float("nan"))
                    policy_output["selected"][group_name].append(
                        {
                            "count": len(selection),
                            "contribution_mass": float(
                                sum(item[2] for item in selection)
                            ),
                            "status": (
                                status
                                if group_name in {"shared", "new"}
                                else (
                                    "inconclusive_no_unused_control_atoms"
                                    if not selection
                                    else f"inconclusive_parent_policy_{status}"
                                )
                            ),
                        }
                    )
                    continue

                masks = masks_from_selection(selection)
                effects: list[torch.Tensor] = []
                with torch.no_grad():
                    for batch_index, (images_cpu, _labels, _ids) in enumerate(batches):
                        images = images_cpu.to(device)
                        baseline_representation = baseline_representations[batch_index].to(
                            device
                        )
                        with atom_output_mask(target_model, masks):
                            _changed_logits, changed = forward_with_capture_and_interventions(
                                target_model,
                                images,
                                capture_points=["pre_classifier"],
                            )
                        effects.append(
                            (changed["pre_classifier"] - baseline_representation)
                            .reshape(images.shape[0], -1)
                            .square()
                            .mean(1)
                            .sqrt()
                            .cpu()
                        )
                policy_output[group_name].append(
                    float(torch.cat(effects).mean()) if effects else float("nan")
                )
                policy_output["selected"][group_name].append(
                    {
                        "count": len(selection),
                        "contribution_mass": float(sum(item[2] for item in selection)),
                        "status": "completed",
                    }
                )
        output["policies"][policy] = policy_output
    return output


# Block interventions
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import Intervention, forward_from_block_input_with_interventions, forward_with_capture_and_interventions
from ..measurements.direct import _run_block_window

def _swap_direction(
    donor_model: nn.Module,
    receiver_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    depth = len(receiver_model.transformer_blocks)
    effects = np.zeros((depth, depth), dtype=np.float64)
    prediction_match = np.zeros((depth, depth), dtype=np.float64)
    counts = np.zeros((depth, depth), dtype=np.float64)
    donor_model.eval().to(device)
    receiver_model.eval().to(device)
    with torch.no_grad():
        for images_cpu, _labels, _ids in batches:
            images = images_cpu.to(device)
            points = ["pre_classifier", *[f"block_{j:02d}_input" for j in range(depth)]]
            baseline_logits, baseline = forward_with_capture_and_interventions(
                receiver_model, images, capture_points=points
            )
            for receiver_index in range(depth):
                native_input = baseline[f"block_{receiver_index:02d}_input"]
                for donor_index in range(depth):
                    donor_update = _run_block_window(
                        donor_model.transformer_blocks, donor_index, 1, native_input
                    )
                    changed_logits, changed = forward_from_block_input_with_interventions(
                        receiver_model,
                        native_input,
                        start_block_index=receiver_index,
                        capture_points=["pre_classifier"],
                        interventions=[
                            Intervention(
                                receiver_index,
                                "block_update",
                                replacement=donor_update,
                            )
                        ],
                    )
                    effect = (changed["pre_classifier"] - baseline["pre_classifier"]).reshape(images.shape[0], -1).square().mean(1).sqrt()
                    effects[donor_index, receiver_index] += float(effect.sum().cpu())
                    prediction_match[donor_index, receiver_index] += float((changed_logits.argmax(1) == baseline_logits.argmax(1)).sum().cpu())
                    counts[donor_index, receiver_index] += int(images.shape[0])
    return {
        "execution_contract": "receiver_native_prefix_captured_once_then_exact_suffix_only_per_swap",
        "final_representation_rms_effect_12x12": (effects / np.maximum(counts, 1)).tolist(),
        "receiver_prediction_retention_12x12": (prediction_match / np.maximum(counts, 1)).tolist(),
    }


def full_block_swap_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "left_donor_to_right_receiver": _swap_direction(left_model, right_model, batches, device=device),
        "right_donor_to_left_receiver": _swap_direction(right_model, left_model, batches, device=device),
    }


def _activation_patch_direction(
    donor_model: nn.Module,
    receiver_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    intervention_point: str,
) -> dict[str, Any]:
    """Directly insert donor activations into a receiver without adapters."""

    depth = len(receiver_model.transformer_blocks)
    effect = np.zeros((depth, depth), dtype=np.float64)
    retention = np.zeros((depth, depth), dtype=np.float64)
    counts = np.zeros((depth, depth), dtype=np.float64)
    donor_model.eval().to(device)
    receiver_model.eval().to(device)
    with torch.no_grad():
        for images_cpu, _labels, _ids in batches:
            images = images_cpu.to(device)
            capture_suffix = {
                "block_update": "update",
                "post_o_attention_output": "post_o_attention_output",
                "post_w2_mlp_output": "post_w2_mlp_output",
            }.get(str(intervention_point))
            if capture_suffix is None:
                raise ValueError(f"Unknown activation-patching intervention point: {intervention_point}")
            donor_points = [f"block_{index:02d}_{capture_suffix}" for index in range(depth)]
            _donor_logits, donor_taps = forward_with_capture_and_interventions(
                donor_model, images, capture_points=donor_points
            )
            baseline_points = [
                "pre_classifier", *[f"block_{index:02d}_input" for index in range(depth)]
            ]
            baseline_logits, baseline = forward_with_capture_and_interventions(
                receiver_model, images, capture_points=baseline_points
            )
            for donor_index in range(depth):
                donor_value = donor_taps[f"block_{donor_index:02d}_{capture_suffix}"]
                for receiver_index in range(depth):
                    native_input = baseline[f"block_{receiver_index:02d}_input"]
                    changed_logits, changed = forward_from_block_input_with_interventions(
                        receiver_model,
                        native_input,
                        start_block_index=receiver_index,
                        capture_points=["pre_classifier"],
                        interventions=[
                            Intervention(
                                receiver_index,
                                intervention_point,
                                replacement=donor_value,
                            )
                        ],
                    )
                    difference = (
                        changed["pre_classifier"] - baseline["pre_classifier"]
                    ).reshape(images.shape[0], -1).square().mean(1).sqrt()
                    effect[donor_index, receiver_index] += float(difference.sum().cpu())
                    retention[donor_index, receiver_index] += float(
                        (changed_logits.argmax(1) == baseline_logits.argmax(1)).sum().cpu()
                    )
                    counts[donor_index, receiver_index] += int(images.shape[0])
    return {
        "execution_contract": "receiver_native_prefix_captured_once_then_exact_suffix_only_per_activation_patch",
        "receiver_final_representation_rms_effect_12x12": (
            effect / np.maximum(counts, 1)
        ).tolist(),
        "receiver_prediction_retention_12x12": (
            retention / np.maximum(counts, 1)
        ).tolist(),
    }


def cross_model_activation_patching_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for point in ("block_update", "post_o_attention_output", "post_w2_mlp_output"):
        output[point] = {
            "left_donor_to_right_receiver": _activation_patch_direction(
                left_model,
                right_model,
                batches,
                device=device,
                intervention_point=point,
            ),
            "right_donor_to_left_receiver": _activation_patch_direction(
                right_model,
                left_model,
                batches,
                device=device,
                intervention_point=point,
            ),
        }
    return output
