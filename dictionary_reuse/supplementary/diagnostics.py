"""Supplementary gradient, probe, and representation diagnostics."""

from __future__ import annotations


# Gradient profiles
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..interventions import forward_with_capture_and_interventions

from ..measurements.representation_similarity import pairwise_cka_matrix

def _gradient_signatures(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    objective: str,
) -> list[torch.Tensor]:
    depth = len(model.transformer_blocks)
    rows: list[list[torch.Tensor]] = [[] for _ in range(depth)]
    model.eval().to(device)
    for images_cpu, labels_cpu, _ids in batches:
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)
        logits, taps = forward_with_capture_and_interventions(
            model,
            images,
            capture_points=[
                "final_cls", *[f"block_{i:02d}_output" for i in range(depth)]
            ],
        )
        if objective == "task_loss":
            per_sample_objective = F.cross_entropy(logits, labels, reduction="none")
        elif objective == "representation_norm":
            per_sample_objective = 0.5 * taps["final_cls"].float().square().sum(dim=1)
        else:
            raise ValueError(objective)
        block_outputs = [taps[f"block_{i:02d}_output"] for i in range(depth)]
        gradients = torch.autograd.grad(
            per_sample_objective.sum(),
            block_outputs,
            retain_graph=False,
            create_graph=False,
        )
        for index, gradient in enumerate(gradients):
            value = gradient.float()
            cls = value[:, 0]
            patch = value[:, 1:]
            feature = torch.stack(
                [
                    value.square().mean(dim=(1, 2)).sqrt(),
                    cls.square().mean(dim=1).sqrt(),
                    patch.square().mean(dim=(1, 2)).sqrt(),
                ],
                dim=1,
            )
            rows[index].append(feature.detach().cpu())
    return [torch.cat(values, dim=0) for values in rows]


def gradient_profile_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    include_task_loss: bool,
    task_loss_scope: str,
) -> dict[str, Any]:
    """Compare label-free gradients everywhere and CE gradients only on a shared native task."""

    result: dict[str, Any] = {}
    objectives = ["representation_norm"]
    if bool(include_task_loss):
        objectives.insert(0, "task_loss")
    for objective in objectives:
        left = _gradient_signatures(left_model, batches, device=device, objective=objective)
        right = _gradient_signatures(right_model, batches, device=device, objective=objective)
        result[f"{objective}_gradient_profile_debiased_cka_12x12"] = pairwise_cka_matrix(left, right)
        result[f"left_{objective}_gradient_norm_profile"] = [float(value[:, 0].mean()) for value in left]
        result[f"right_{objective}_gradient_norm_profile"] = [float(value[:, 0].mean()) for value in right]
    result["task_loss_gradient_status"] = "included" if include_task_loss else "not_applicable"
    result["task_loss_scope"] = str(task_loss_scope)
    result["task_loss_exclusion_reason"] = (
        None
        if include_task_loss
        else "numeric class labels do not denote the same semantic task for both compared models"
    )
    result["label_free_primary"] = "representation_norm_gradient_profile_debiased_cka_12x12"
    result["gradient_target_contract"] = (
        "gradient_with_respect_to_block_output_equals_gradient_with_respect_to_additive_block_update"
    )
    result["batching_contract"] = (
        "one_backward_of_summed_per_sample_objectives_is_exact_because_eval_transformer_has_no_cross_sample_coupling"
    )
    return result


# Probe diagnostics
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..measurements.representation_similarity import collect_native_block_features

def _ridge_fit(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(labels).sort().values
    mapped = torch.searchsorted(classes, labels)
    x = torch.cat([features.float(), torch.ones(features.shape[0], 1)], dim=1)
    y = F.one_hot(mapped, num_classes=int(classes.numel())).float()
    identity = torch.eye(x.shape[1])
    identity[-1, -1] = 0
    weight = torch.linalg.solve(x.T @ x + float(ridge) * identity, x.T @ y)
    return weight, classes


def _probe_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    classes: torch.Tensor,
) -> float:
    x = torch.cat([features.float(), torch.ones(features.shape[0], 1)], dim=1)
    prediction = classes[(x @ weight).argmax(dim=1)]
    return float((prediction == labels).float().mean())


def _collect_cls_and_labels(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    features = collect_native_block_features(model, batches, device=device, tap_suffix="output", feature_mode="cls")
    labels = torch.cat([batch[1].detach().cpu().long() for batch in batches])
    return features, labels


def _nonlinear_probe_result(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    """Fit one fixed small nonlinear probe without hyperparameter selection."""

    torch.manual_seed(int(seed))
    classes = torch.unique(train_labels).sort().values
    train_targets = torch.searchsorted(classes, train_labels)
    mean = train_features.float().mean(dim=0, keepdim=True)
    scale = train_features.float().std(dim=0, keepdim=True).clamp_min(1e-5)

    def normalize(value: torch.Tensor) -> torch.Tensor:
        return (value.float() - mean) / scale

    model = nn.Sequential(
        nn.Linear(int(train_features.shape[1]), 64),
        nn.GELU(),
        nn.Linear(64, int(classes.numel())),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    normalized_train = normalize(train_features)
    batch_size = 256
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1)
    model.train()
    for _epoch in range(20):
        order = torch.randperm(int(normalized_train.shape[0]), generator=generator)
        for start in range(0, int(order.numel()), batch_size):
            selected = order[start : start + batch_size]
            features = normalized_train[selected].to(device)
            targets = train_targets[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features), targets)
            loss.backward()
            optimizer.step()

    def accuracy(features: torch.Tensor, labels: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            prediction_index = model(normalize(features).to(device)).argmax(dim=1).cpu()
        prediction = classes[prediction_index]
        return float((prediction == labels).float().mean())

    return {
        "validation_accuracy": accuracy(validation_features, validation_labels),
        "test_accuracy": accuracy(test_features, test_labels),
        "hidden_width": 64,
        "epochs": 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "selection_policy": "fixed_single_configuration_no_sweep",
    }


def _selected_nonlinear_probe_depths(depth: int) -> tuple[int, ...]:
    """Select start/mid/end probe depths without assuming a 12-block backbone.

    The middle location preserves the release ViT-12 choice (block 5) by scaling
    its relative depth, so the current 12-block experiment is numerically unchanged.
    """

    if depth <= 0:
        raise ValueError("nonlinear probe requires at least one representation depth")
    raw = (0, int(round((depth - 1) * (5.0 / 11.0))), depth - 1)
    return tuple(dict.fromkeys(raw))


def linear_probe_profiles(
    left_model: nn.Module,
    right_model: nn.Module,
    train_batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    validation_batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    test_batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    same_task: bool,
    nonlinear_seed: int = 2026080671,
) -> dict[str, Any]:
    left_train, train_labels = _collect_cls_and_labels(left_model, train_batches, device=device)
    right_train, right_train_labels = _collect_cls_and_labels(right_model, train_batches, device=device)
    left_val, val_labels = _collect_cls_and_labels(left_model, validation_batches, device=device)
    right_val, right_val_labels = _collect_cls_and_labels(right_model, validation_batches, device=device)
    left_test, test_labels = _collect_cls_and_labels(left_model, test_batches, device=device)
    right_test, right_test_labels = _collect_cls_and_labels(right_model, test_batches, device=device)
    output: dict[str, Any] = {
        "same_task": bool(same_task),
        "independent_linear": {"left": [], "right": []},
        "independent_nonlinear": {"left": {}, "right": {}},
    }
    direct: list[float] = []
    for index in range(len(left_train)):
        left_weight, left_classes = _ridge_fit(left_train[index], train_labels)
        right_weight, right_classes = _ridge_fit(right_train[index], right_train_labels)
        output["independent_linear"]["left"].append({
            "validation_accuracy": _probe_accuracy(left_val[index], val_labels, left_weight, left_classes),
            "test_accuracy": _probe_accuracy(left_test[index], test_labels, left_weight, left_classes),
        })
        output["independent_linear"]["right"].append({
            "validation_accuracy": _probe_accuracy(right_val[index], right_val_labels, right_weight, right_classes),
            "test_accuracy": _probe_accuracy(right_test[index], right_test_labels, right_weight, right_classes),
        })
        if same_task:
            direct.append(_probe_accuracy(right_test[index], right_test_labels, left_weight, left_classes))
    if same_task:
        output["source_trained_direct_transfer_to_target_test_accuracy"] = direct

    representation_depth = len(left_train)
    if not all(
        len(values) == representation_depth
        for values in (right_train, left_val, right_val, left_test, right_test)
    ):
        raise ValueError("left/right nonlinear probe representation depths must match")
    selected_depths = _selected_nonlinear_probe_depths(representation_depth)
    for side_index, (side, train, validation, test, labels_train, labels_validation, labels_test) in enumerate(
        (
            ("left", left_train, left_val, left_test, train_labels, val_labels, test_labels),
            ("right", right_train, right_val, right_test, right_train_labels, right_val_labels, right_test_labels),
        )
    ):
        for depth_index in selected_depths:
            output["independent_nonlinear"][side][str(depth_index)] = _nonlinear_probe_result(
                train[depth_index],
                labels_train,
                validation[depth_index],
                labels_validation,
                test[depth_index],
                labels_test,
                device=device,
                seed=int(nonlinear_seed) + 100 * side_index + depth_index,
            )
    output["nonlinear_probe_depths"] = list(selected_depths)
    output["nonlinear_probe_policy"] = (
        "independent_start_scaled_middle_end_single_hidden64_gelu_20epochs_single_seed_no_sweep"
    )
    return output


# Representation geometry
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..interventions import forward_with_capture_and_interventions
from ..measurements.representation_similarity import _feature_view, pairwise_cka_matrix
from ..measurements.representation_cache import combined_signal_variation_validity, sample_variation_rms

def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rankdata(left.reshape(-1))
    right_rank = _rankdata(right.reshape(-1))
    if float(left_rank.std()) == 0.0 or float(right_rank.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _sample_distance_rank(feature: torch.Tensor) -> np.ndarray:
    value = feature.detach().float()
    value = value - value.mean(dim=0, keepdim=True)
    value = F.normalize(value, dim=1)
    distance = (1.0 - value @ value.T).clamp_min(0)
    indices = torch.triu_indices(int(distance.shape[0]), int(distance.shape[0]), offset=1)
    return _rankdata(distance[indices[0], indices[1]].cpu().numpy())


def _rsa_signal_validity(features: Sequence[torch.Tensor]) -> dict[str, Any]:
    return combined_signal_variation_validity(
        [float(value.detach().float().square().mean().sqrt()) for value in features],
        [sample_variation_rms(value) for value in features],
        absolute_minimum=1e-8,
        relative_to_median=0.05,
    )


def _rsa_matrix_summary(matrix: np.ndarray) -> dict[str, float]:
    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]

    def finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    same_index = finite_mean(diagonal)
    off_index = finite_mean(off_diagonal)
    return {
        "same_index_mean": same_index,
        "off_index_mean": off_index,
        "alignment_margin": float(same_index - off_index),
    }


def prepare_representation_rsa_reference(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Cache the fixed Source representation ranks used for the e0 RSA audit."""

    original_device = next(model.parameters()).device
    was_training = bool(model.training)
    try:
        modes = _collect_output_feature_modes(model, batches, device=device)
        reference: dict[str, Any] = {
            "sample_count": int(modes["cls"][0].shape[0]),
            "modes": {},
        }
        for mode in ("cls", "patch_mean_rms"):
            features = modes[mode]
            signal = _rsa_signal_validity(features)
            valid = np.asarray(signal["valid_by_block"], dtype=bool)
            reference["modes"][mode] = {
                "signal": signal,
                "valid": valid,
                "ranks": [
                    _sample_distance_rank(value) if valid[index] else None
                    for index, value in enumerate(features)
                ],
            }
        return reference
    finally:
        model.to(original_device)
        model.train(was_training)


def representation_rsa_against_reference(
    reference: dict[str, Any],
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Measure Target block-output RSA against one fixed Source reference."""

    original_device = next(model.parameters()).device
    was_training = bool(model.training)
    try:
        modes = _collect_output_feature_modes(model, batches, device=device)
    finally:
        model.to(original_device)
        model.train(was_training)

    output: dict[str, Any] = {
        "same_task_trajectory_rsa_sample_count": int(reference["sample_count"]),
    }
    for mode, output_name in (("cls", "cls"), ("patch_mean_rms", "patch")):
        source = reference["modes"][mode]
        target_features = modes[mode]
        target_signal = _rsa_signal_validity(target_features)
        target_valid = np.asarray(target_signal["valid_by_block"], dtype=bool)
        source_valid = np.asarray(source["valid"], dtype=bool)
        matrix = np.full((len(source_valid), len(target_valid)), np.nan, dtype=np.float64)
        target_ranks = [
            _sample_distance_rank(value) if target_valid[index] else None
            for index, value in enumerate(target_features)
        ]
        for i, source_rank in enumerate(source["ranks"]):
            if source_rank is None:
                continue
            for j, target_rank in enumerate(target_ranks):
                if target_rank is None:
                    continue
                matrix[i, j] = _spearman(source_rank, target_rank)

        summary = _rsa_matrix_summary(matrix)
        output[f"same_task_trajectory_{output_name}_rsa_spearman_12x12"] = matrix.tolist()
        output[f"same_task_trajectory_{output_name}_rsa_same_index_mean"] = summary["same_index_mean"]
        output[f"same_task_trajectory_{output_name}_rsa_off_index_mean"] = summary["off_index_mean"]
        output[f"same_task_trajectory_{output_name}_rsa_alignment_margin"] = summary["alignment_margin"]
    return output


def _pca_basis(
    feature: torch.Tensor,
    *,
    variance_fraction: float = 0.99,
    max_rank: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = feature.detach().float()
    value = value - value.mean(dim=0, keepdim=True)
    u, singular, _vh = torch.linalg.svd(value, full_matrices=False)
    energy = singular.square()
    total_energy = energy.sum()
    if not bool(torch.isfinite(total_energy)) or float(total_energy.cpu()) <= 1e-12:
        raise ValueError("SVCCA/weighted CCA proxy input has no estimable sample variation")
    cumulative = energy.cumsum(0) / total_energy
    rank = int(torch.searchsorted(cumulative, torch.tensor(float(variance_fraction), device=cumulative.device)).item()) + 1
    rank = max(1, min(rank, int(max_rank), int(u.shape[1])))
    return u[:, :rank], singular[:rank]


def _canonical_similarity(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    left_u, left_s = left
    right_u, right_s = right
    cross = left_u.T @ right_u
    u, canonical, vh = torch.linalg.svd(cross, full_matrices=False)
    canonical = canonical.clamp(0, 1)
    svcca = float(canonical.mean().cpu())
    left_weights = (u.abs().T @ left_s[: u.shape[0]]).clamp_min(0)
    right_weights = (vh.abs() @ right_s[: vh.shape[1]]).clamp_min(0)
    left_weights = left_weights[: canonical.numel()]
    right_weights = right_weights[: canonical.numel()]
    left_weighted_cca = (canonical * left_weights).sum() / left_weights.sum().clamp_min(1e-12)
    right_weighted_cca = (canonical * right_weights).sum() / right_weights.sum().clamp_min(1e-12)
    return svcca, float((0.5 * (left_weighted_cca + right_weighted_cca)).cpu())


def _collect_output_feature_modes(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    depth = len(model.transformer_blocks)
    cls_parts: list[list[torch.Tensor]] = [[] for _ in range(depth)]
    patch_parts: list[list[torch.Tensor]] = [[] for _ in range(depth)]
    points = [f"block_{index:02d}_output" for index in range(depth)]
    model.eval().to(device)
    with torch.no_grad():
        for images_cpu, _labels, _ids in batches:
            _logits, taps = forward_with_capture_and_interventions(
                model, images_cpu.to(device), capture_points=points
            )
            for index in range(depth):
                value = taps[f"block_{index:02d}_output"]
                cls_parts[index].append(_feature_view(value, "cls").detach().cpu())
                patch_parts[index].append(
                    _feature_view(value, "patch_mean_rms").detach().cpu()
                )
    return {
        "cls": [torch.cat(parts, dim=0) for parts in cls_parts],
        "patch_mean_rms": [torch.cat(parts, dim=0) for parts in patch_parts],
    }


def representation_geometry_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    output: dict[str, Any] = {"validity_masks": {}, "low_signal": {}}
    left_modes = _collect_output_feature_modes(left_model, batches, device=device)
    right_modes = _collect_output_feature_modes(right_model, batches, device=device)
    for mode in ("cls", "patch_mean_rms"):
        left = left_modes[mode]
        right = right_modes[mode]
        left_signal = combined_signal_variation_validity(
            [float(value.detach().float().square().mean().sqrt()) for value in left],
            [sample_variation_rms(value) for value in left],
            absolute_minimum=1e-8,
            relative_to_median=0.05,
        )
        right_signal = combined_signal_variation_validity(
            [float(value.detach().float().square().mean().sqrt()) for value in right],
            [sample_variation_rms(value) for value in right],
            absolute_minimum=1e-8,
            relative_to_median=0.05,
        )
        left_valid = np.asarray(left_signal["valid_by_block"], dtype=bool)
        right_valid = np.asarray(right_signal["valid_by_block"], dtype=bool)
        pair_valid = np.logical_and.outer(left_valid, right_valid)
        output["low_signal"][mode] = {"left": left_signal, "right": right_signal}

        cka_key = f"{mode}_linear_cka_12x12"
        cka = np.asarray(pairwise_cka_matrix(left, right), dtype=np.float64)
        cka[~pair_valid] = np.nan
        output[cka_key] = cka.tolist()
        output["validity_masks"][cka_key] = (pair_valid & np.isfinite(cka)).tolist()

        left_ranks = [_sample_distance_rank(value) if left_valid[index] else None for index, value in enumerate(left)]
        right_ranks = [_sample_distance_rank(value) if right_valid[index] else None for index, value in enumerate(right)]
        rsa = np.full((len(left), len(right)), np.nan, dtype=np.float64)
        for i, left_rank in enumerate(left_ranks):
            if left_rank is None:
                continue
            for j, right_rank in enumerate(right_ranks):
                if right_rank is None:
                    continue
                rsa[i, j] = _spearman(left_rank, right_rank)
        rsa_key = f"{mode}_rsa_spearman_12x12"
        output[rsa_key] = rsa.tolist()
        output["validity_masks"][rsa_key] = (pair_valid & np.isfinite(rsa)).tolist()

        left_pca = [_pca_basis(value) if left_valid[index] else None for index, value in enumerate(left)]
        right_pca = [_pca_basis(value) if right_valid[index] else None for index, value in enumerate(right)]
        svcca = np.full((len(left), len(right)), np.nan, dtype=np.float64)
        weighted_cca_proxy = np.full((len(left), len(right)), np.nan, dtype=np.float64)
        for i, left_value in enumerate(left_pca):
            if left_value is None:
                continue
            for j, right_value in enumerate(right_pca):
                if right_value is None:
                    continue
                svcca_value, weighted_proxy_value = _canonical_similarity(left_value, right_value)
                svcca[i, j] = svcca_value
                weighted_cca_proxy[i, j] = weighted_proxy_value
        svcca_key = f"{mode}_svcca_12x12"
        weighted_key = f"{mode}_weighted_cca_proxy_12x12"
        output[svcca_key] = svcca.tolist()
        output[weighted_key] = weighted_cca_proxy.tolist()
        output["validity_masks"][svcca_key] = (pair_valid & np.isfinite(svcca)).tolist()
        output["validity_masks"][weighted_key] = (
            pair_valid & np.isfinite(weighted_cca_proxy)
        ).tolist()
        output[f"{mode}_weighted_cca_proxy_contract"] = (
            "symmetric_singular_value_weighted_canonical_correlation_proxy"
        )
    output["degenerate_signal_contract"] = (
        "RSA_SVCCA_and_weighted_CCA_proxy_are_NaN_inconclusive_when_signal_or_sample_variation_is_below_the_same_style_of_threshold_used_by_core_CKA"
    )
    output["forward_reuse_contract"] = (
        "one_native_forward_per_model_batch_shared_by_cls_and_patch_mean_rms_profiles"
    )
    return output
