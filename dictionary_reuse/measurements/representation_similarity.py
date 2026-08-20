"""Feature-space similarity primitives used by DiR measurements."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from ..interventions import forward_with_capture_and_interventions

def _subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    """Symmetric containment-normalized overlap for possibly unequal ranks."""

    if left.ndim != 2 or right.ndim != 2 or int(left.shape[0]) != int(right.shape[0]):
        return float("nan")
    normalizer = min(int(left.shape[1]), int(right.shape[1]))
    if normalizer < 1:
        return float("nan")
    # Use every valid basis vector on both sides. Truncating the larger
    # subspace by column order can report zero even when the smaller space
    # is fully contained in it. The Frobenius norm is basis-order invariant.
    overlap = left.T @ right
    score = overlap.square().sum() / float(normalizer)
    return float(score.clamp(min=0.0, max=1.0).cpu())


def _padded_singular_spectrum_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    """Cosine between singular spectra without assuming equal numerical rank.

    Randomized-SVD descriptors store a fixed-width singular-value vector with
    zeros beyond the numerical rank. Comparing those full vectors preserves
    unequal-rank information and avoids shape-dependent ``torch.dot`` errors.
    """

    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("DiR singular spectra must be one-dimensional")
    width = max(int(left.numel()), int(right.numel()))
    if width < 1:
        return 1.0
    left_full = torch.zeros(width, dtype=left.dtype, device=left.device)
    right_full = torch.zeros(width, dtype=right.dtype, device=right.device)
    left_full[: int(left.numel())] = left
    right_full[: int(right.numel())] = right.to(device=left.device, dtype=left.dtype)
    left_norm = left_full.norm()
    right_norm = right_full.norm()
    if float(left_norm) == 0.0 and float(right_norm) == 0.0:
        return 1.0
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return 0.0
    return float((torch.dot(left_full, right_full) / (left_norm * right_norm)).clamp(-1.0, 1.0).cpu())


def _degenerate_aware_subspace_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, str]:
    """Extend subspace overlap to the zero-rank Jacobian by explicit convention.

    The zero operator has the trivial subspace. Two zero-rank operators are
    treated as identical for role-similarity reporting; a zero-rank operator
    compared with a nonzero subspace receives zero overlap. Positive-rank
    comparisons use the ordinary containment-normalized overlap.
    """

    if left.ndim != 2 or right.ndim != 2 or int(left.shape[0]) != int(right.shape[0]):
        raise ValueError("DiR Jacobian subspaces must have matching ambient dimensions")
    left_rank = int(left.shape[1])
    right_rank = int(right.shape[1])
    if left_rank == 0 and right_rank == 0:
        return 1.0, "both_zero_rank"
    if left_rank == 0 or right_rank == 0:
        return 0.0, "one_zero_rank"
    return _subspace_overlap(left, right), "positive_rank"


def _projection_mean_direction_cosine(
    left_projection_features: torch.Tensor | None,
    right_projection_features: torch.Tensor | None,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Auxiliary signed cosine between mean shared-projection responses."""

    if left_projection_features is None or right_projection_features is None:
        return float("nan")
    left_projection = left_projection_features.detach().double()
    right_projection = right_projection_features.detach().double()
    if left_projection.ndim != 2 or right_projection.ndim != 2:
        raise ValueError("DiR Jacobian projection features must be 2-D")
    if int(left_projection.shape[1]) != int(right_projection.shape[1]):
        raise ValueError("DiR Jacobian projection feature widths differ")
    if not torch.isfinite(left_projection).all() or not torch.isfinite(right_projection).all():
        raise ValueError("DiR Jacobian projection features contain non-finite values")
    left_mean = left_projection.mean(dim=0)
    right_mean = right_projection.mean(dim=0)
    left_norm = left_mean.norm()
    right_norm = right_mean.norm()
    if float(left_norm.cpu()) <= float(epsilon) or float(right_norm.cpu()) <= float(epsilon):
        return float("nan")
    return float(
        (
            torch.dot(left_mean, right_mean)
            / (left_norm * right_norm).clamp_min(float(epsilon))
        ).cpu()
    )


def _jacobian_debiased_gram_cka(
    left_gram: torch.Tensor,
    right_gram: torch.Tensor,
    *,
    left_response_rms: float | None = None,
    right_response_rms: float | None = None,
    left_projection_features: torch.Tensor | None = None,
    right_projection_features: torch.Tensor | None = None,
    minimum_signal_rms_absolute: float = 1e-8,
    epsilon: float = 1e-12,
) -> tuple[float, str]:
    """Primary Jacobian response similarity using only debiased CKA.

    Uncentered response magnitude is used only as a detection threshold. A
    below-threshold response is inconclusive rather than evidence for a true
    zero Jacobian. Detected but sample-constant responses are also
    inconclusive for U-centered/debiased CKA; their shared-projection mean
    direction cosine is reported separately as an auxiliary diagnostic.
    """

    if tuple(left_gram.shape) != tuple(right_gram.shape):
        raise ValueError("DiR Jacobian Gram shapes differ")
    if not torch.isfinite(left_gram).all() or not torch.isfinite(right_gram).all():
        raise ValueError("DiR Jacobian Gram contains non-finite values")

    if left_response_rms is None:
        left_response_rms = float(left_gram.diagonal().clamp_min(0).mean().sqrt().cpu())
    if right_response_rms is None:
        right_response_rms = float(right_gram.diagonal().clamp_min(0).mean().sqrt().cpu())
    if not math.isfinite(float(left_response_rms)) or not math.isfinite(float(right_response_rms)):
        raise ValueError("DiR Jacobian response RMS is non-finite")
    threshold = float(minimum_signal_rms_absolute)
    left_below_detection = float(left_response_rms) <= threshold
    right_below_detection = float(right_response_rms) <= threshold
    if left_below_detection and right_below_detection:
        return float("nan"), "both_below_detection_threshold"
    if left_below_detection or right_below_detection:
        return float("nan"), "one_below_detection_threshold"

    left = _u_center_gram(left_gram)
    right = _u_center_gram(right_gram)
    left_norm = float(left.square().sum().sqrt().cpu())
    right_norm = float(right.square().sum().sqrt().cpu())
    left_constant = left_norm <= float(epsilon)
    right_constant = right_norm <= float(epsilon)
    if left_constant or right_constant:
        classification = (
            "both_detected_constant_response_debiased_cka_inconclusive"
            if left_constant and right_constant
            else "one_detected_constant_response_debiased_cka_inconclusive"
        )
        return float("nan"), classification

    score = float(((left * right).sum() / (left_norm * right_norm)).cpu())
    return score, "nondegenerate_debiased_cka"


def _jacobian_gram_similarity_matrix(
    left_values: Sequence[dict[str, Any]],
    right_values: Sequence[dict[str, Any]],
    *,
    key: str = "gram",
    minimum_signal_rms_absolute: float = 1e-8,
) -> tuple[list[list[float]], list[list[str]], list[list[float]]]:
    rms_key = {"gram": "rms", "first_gram": "first_rms", "second_gram": "second_rms"}.get(
        key, "rms"
    )
    projection_key = {
        "gram": "projection_features",
        "first_gram": "first_projection_features",
        "second_gram": "second_projection_features",
    }.get(key, "projection_features")
    scores: list[list[float]] = []
    classes: list[list[str]] = []
    auxiliary_mean_projection_cosines: list[list[float]] = []
    for left_item in left_values:
        score_row: list[float] = []
        class_row: list[str] = []
        cosine_row: list[float] = []
        for right_item in right_values:
            score, classification = _jacobian_debiased_gram_cka(
                left_item[key],
                right_item[key],
                left_response_rms=float(left_item.get(rms_key, float("nan"))),
                right_response_rms=float(right_item.get(rms_key, float("nan"))),
                left_projection_features=left_item.get(projection_key),
                right_projection_features=right_item.get(projection_key),
                minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
            )
            auxiliary_cosine = float("nan")
            if "detected_constant_response" in str(classification):
                auxiliary_cosine = _projection_mean_direction_cosine(
                    left_item.get(projection_key), right_item.get(projection_key)
                )
            score_row.append(float(score))
            class_row.append(str(classification))
            cosine_row.append(float(auxiliary_cosine))
        scores.append(score_row)
        classes.append(class_row)
        auxiliary_mean_projection_cosines.append(cosine_row)
    return scores, classes, auxiliary_mean_projection_cosines


def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    """Biased double-centering retained only as an auxiliary diagnostic."""

    value = gram.detach().double()
    if value.ndim != 2 or int(value.shape[0]) != int(value.shape[1]):
        raise ValueError(f"Expected a square Gram matrix, got {tuple(value.shape)}")
    row_mean = value.mean(dim=1, keepdim=True)
    column_mean = value.mean(dim=0, keepdim=True)
    return value - row_mean - column_mean + value.mean()


def _u_center_gram(gram: torch.Tensor) -> torch.Tensor:
    """U-center a Gram matrix for the unbiased HSIC/CKA estimator."""

    value = gram.detach().double().clone()
    if value.ndim != 2 or int(value.shape[0]) != int(value.shape[1]):
        raise ValueError(f"Expected a square Gram matrix, got {tuple(value.shape)}")
    sample_count = int(value.shape[0])
    if sample_count < 4:
        raise ValueError("Debiased CKA requires at least four paired samples")
    value.fill_diagonal_(0.0)
    row_sum = value.sum(dim=1, keepdim=True)
    column_sum = value.sum(dim=0, keepdim=True)
    total_sum = value.sum()
    centered = (
        value
        - row_sum / float(sample_count - 2)
        - column_sum / float(sample_count - 2)
        + total_sum / float((sample_count - 1) * (sample_count - 2))
    )
    centered.fill_diagonal_(0.0)
    return centered


def centered_linear_cka(left: torch.Tensor, right: torch.Tensor, *, epsilon: float = 1e-12) -> float:
    """Biased linear CKA retained as an explicitly auxiliary diagnostic."""

    x = left.detach().float().reshape(left.shape[0], -1)
    y = right.detach().float().reshape(right.shape[0], -1)
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("CKA sample counts differ")
    return centered_gram_cka(x @ x.T, y @ y.T, epsilon=epsilon)


def centered_gram_cka(left_gram: torch.Tensor, right_gram: torch.Tensor, *, epsilon: float = 1e-12) -> float:
    """Biased Gram CKA retained for comparison with prior report conventions."""

    if tuple(left_gram.shape) != tuple(right_gram.shape):
        raise ValueError("Gram shapes differ")
    left = _center_gram(left_gram)
    right = _center_gram(right_gram)
    numerator = (left * right).sum()
    denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(float(epsilon))
    return float((numerator / denominator).cpu())


def debiased_linear_cka(left: torch.Tensor, right: torch.Tensor, *, epsilon: float = 1e-12) -> float:
    """U-centered linear CKA, primary under high-dimension/low-sample DiR."""

    x = left.detach().float().reshape(left.shape[0], -1)
    y = right.detach().float().reshape(right.shape[0], -1)
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("CKA sample counts differ")
    return debiased_gram_cka(x @ x.T, y @ y.T, epsilon=epsilon)


def debiased_gram_cka(left_gram: torch.Tensor, right_gram: torch.Tensor, *, epsilon: float = 1e-12) -> float:
    if tuple(left_gram.shape) != tuple(right_gram.shape):
        raise ValueError("Gram shapes differ")
    left = _u_center_gram(left_gram)
    right = _u_center_gram(right_gram)
    numerator = (left * right).sum()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if float(denominator) <= float(epsilon):
        # Undefined U-centered/debiased CKA is scientifically inconclusive.
        # Never encode it as numeric zero because downstream code may mistake
        # that placeholder for genuine orthogonality.
        return float("nan")
    return float((numerator / denominator).cpu())


def _feature_view(value: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode)
    if mode == "full_token":
        return value.reshape(value.shape[0], -1)
    if mode == "cls":
        return value[:, 0].reshape(value.shape[0], -1)
    if mode == "patch":
        return value[:, 1:].reshape(value.shape[0], -1)
    if mode == "patch_mean_rms":
        patch = value[:, 1:].float()
        return torch.cat([patch.mean(dim=1), patch.square().mean(dim=1).sqrt()], dim=1)
    raise ValueError(f"Unknown DiR feature mode: {mode}")


def collect_native_block_features(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    tap_suffix: str = "update",
    feature_mode: str = "full_token",
) -> list[torch.Tensor]:
    model.eval().to(device)
    collected: list[list[torch.Tensor]] = [[] for _ in range(len(model.transformer_blocks))]
    capture_points = [f"block_{index:02d}_{tap_suffix}" for index in range(len(model.transformer_blocks))]
    with torch.no_grad():
        for images_cpu, _labels, _ids in batches:
            _logits, taps = forward_with_capture_and_interventions(
                model, images_cpu.to(device), capture_points=capture_points
            )
            for index in range(len(collected)):
                value = _feature_view(taps[f"block_{index:02d}_{tap_suffix}"], feature_mode)
                collected[index].append(value.detach().cpu())
    return [torch.cat(parts, dim=0) for parts in collected]


def _centered_feature_gram(
    value: torch.Tensor,
    *,
    debiased: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and center one feature Gram once for pairwise CKA reuse."""

    features = value.detach().float().reshape(value.shape[0], -1)
    gram = features @ features.T
    centered = _u_center_gram(gram) if debiased else _center_gram(gram)
    squared_norm = centered.square().sum()
    return centered, squared_norm


def _pairwise_feature_cka_matrix(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    *,
    debiased: bool,
    epsilon: float = 1e-12,
) -> list[list[float]]:
    """Pairwise CKA with each feature Gram and centering computed only once."""

    right_cache = [
        _centered_feature_gram(value, debiased=debiased) for value in right
    ]
    output: list[list[float]] = []
    for left_value in left:
        left_centered, left_squared_norm = _centered_feature_gram(
            left_value, debiased=debiased
        )
        row: list[float] = []
        for right_centered, right_squared_norm in right_cache:
            if tuple(left_centered.shape) != tuple(right_centered.shape):
                raise ValueError("CKA sample counts differ")
            denominator = torch.sqrt(left_squared_norm * right_squared_norm)
            if debiased and float(denominator) <= float(epsilon):
                row.append(float("nan"))
                continue
            denominator = denominator.clamp_min(float(epsilon))
            row.append(
                float(
                    ((left_centered * right_centered).sum() / denominator).cpu()
                )
            )
        output.append(row)
    return output


def pairwise_cka_matrix(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> list[list[float]]:
    """Primary pairwise U-centered/debiased CKA matrix."""

    return _pairwise_feature_cka_matrix(left, right, debiased=True)


def pairwise_biased_cka_matrix(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> list[list[float]]:
    return _pairwise_feature_cka_matrix(left, right, debiased=False)


def paired_output_metrics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Exact paired-sample agreement with explicit non-finite exclusion.

    A sample is finite only when every value in both paired outputs is finite.
    Non-finite samples never masquerade as zero outputs and make the paired
    metric scientifically inconclusive even though finite-sample diagnostics
    are still returned for debugging.
    """

    x_all = left.detach().float().reshape(left.shape[0], -1)
    y_all = right.detach().float().reshape(right.shape[0], -1)
    if tuple(x_all.shape) != tuple(y_all.shape):
        raise ValueError(
            f"Paired output metric shape mismatch: {tuple(x_all.shape)} vs {tuple(y_all.shape)}"
        )
    finite_sample = torch.isfinite(x_all).all(dim=1) & torch.isfinite(y_all).all(dim=1)
    nonfinite_count = int((~finite_sample).sum().cpu())
    x = x_all[finite_sample]
    y = y_all[finite_sample]
    finite_count = int(x.shape[0])

    if finite_count == 0:
        return {
            "signed_cosine_mean": float("nan"),
            "normalized_l2_mean": float("nan"),
            "symmetric_norm_ratio_mean": float("nan"),
            "valid_sample_count": 0,
            "cosine_valid_sample_count": 0,
            "distance_scale_valid_sample_count": 0,
            "both_zero_sample_count": 0,
            "one_zero_sample_count": 0,
            "finite_sample_count": 0,
            "nonfinite_sample_count": nonfinite_count,
            "total_sample_count": int(x_all.shape[0]),
            "quality_passed": False,
            "cosine_quality_passed": False,
            "cosine_measurement_status": "inconclusive_nonfinite_paired_outputs",
            "measurement_status": "inconclusive_nonfinite_paired_outputs",
        }

    left_norm = x.norm(dim=1)
    right_norm = y.norm(dim=1)
    left_nonzero = left_norm > float(epsilon)
    right_nonzero = right_norm > float(epsilon)
    cosine_valid = left_nonzero & right_nonzero
    one_zero = left_nonzero ^ right_nonzero
    both_zero = ~(left_nonzero | right_nonzero)

    if bool(cosine_valid.any()):
        xv = x[cosine_valid]
        yv = y[cosine_valid]
        ln = left_norm[cosine_valid]
        rn = right_norm[cosine_valid]
        cosine_mean = float(
            ((xv * yv).sum(dim=1) / (ln * rn).clamp_min(float(epsilon))).mean().cpu()
        )
    else:
        cosine_mean = float("nan")

    denominator = 0.5 * (left_norm + right_norm)
    normalized_l2 = torch.zeros_like(denominator)
    nontrivial_distance = denominator > float(epsilon)
    normalized_l2[nontrivial_distance] = (
        (x[nontrivial_distance] - y[nontrivial_distance]).norm(dim=1)
        / denominator[nontrivial_distance]
    )
    normalized_l2[one_zero] = 2.0

    maximum_norm = torch.maximum(left_norm, right_norm)
    norm_ratio = torch.ones_like(maximum_norm)
    nonzero_scale = maximum_norm > float(epsilon)
    norm_ratio[nonzero_scale] = (
        torch.minimum(left_norm[nonzero_scale], right_norm[nonzero_scale])
        / maximum_norm[nonzero_scale]
    )
    norm_ratio[one_zero] = 0.0
    quality_passed = nonfinite_count == 0
    cosine_quality_passed = bool(cosine_valid.any()) and quality_passed
    if bool(cosine_valid.any()):
        cosine_measurement_status = "completed"
    elif bool(both_zero.all()):
        cosine_measurement_status = "inconclusive_all_zero_outputs"
    else:
        cosine_measurement_status = "inconclusive_no_pair_with_two_nonzero_outputs"
    if not quality_passed:
        measurement_status = "inconclusive_nonfinite_paired_outputs"
    elif not cosine_quality_passed:
        measurement_status = "completed_distance_scale_metrics_cosine_inconclusive"
    else:
        measurement_status = "completed"
    return {
        "signed_cosine_mean": cosine_mean,
        "normalized_l2_mean": float(normalized_l2.mean().cpu()),
        "symmetric_norm_ratio_mean": float(norm_ratio.mean().cpu()),
        "valid_sample_count": int(cosine_valid.sum().cpu()),
        "cosine_valid_sample_count": int(cosine_valid.sum().cpu()),
        "distance_scale_valid_sample_count": finite_count,
        "both_zero_sample_count": int(both_zero.sum().cpu()),
        "one_zero_sample_count": int(one_zero.sum().cpu()),
        "finite_sample_count": finite_count,
        "nonfinite_sample_count": nonfinite_count,
        "total_sample_count": int(x_all.shape[0]),
        "quality_passed": quality_passed,
        "cosine_quality_passed": cosine_quality_passed,
        "cosine_measurement_status": cosine_measurement_status,
        "measurement_status": measurement_status,
    }


def _paired_output_metrics_from_components(
    left_components: Sequence[torch.Tensor],
    right_components: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Exact paired metrics for a feature represented by concatenated components.

    This avoids materializing large CLS+patch concatenations while preserving
    exactly the same dot products, norms, distances, and finite-sample policy.
    """

    if not left_components or len(left_components) != len(right_components):
        raise ValueError("DiR paired component metrics require matched nonempty components")
    left_parts = [
        value.detach().float().reshape(value.shape[0], -1) for value in left_components
    ]
    right_parts = [
        value.detach().float().reshape(value.shape[0], -1) for value in right_components
    ]
    sample_count = int(left_parts[0].shape[0])
    if any(int(value.shape[0]) != sample_count for value in (*left_parts, *right_parts)):
        raise ValueError("DiR paired component sample counts differ")
    for left_value, right_value in zip(left_parts, right_parts):
        if tuple(left_value.shape) != tuple(right_value.shape):
            raise ValueError(
                f"Paired component shape mismatch: {tuple(left_value.shape)} vs {tuple(right_value.shape)}"
            )
    finite_sample = torch.ones(sample_count, dtype=torch.bool)
    for left_value, right_value in zip(left_parts, right_parts):
        finite_sample &= torch.isfinite(left_value).all(dim=1)
        finite_sample &= torch.isfinite(right_value).all(dim=1)
    nonfinite_count = int((~finite_sample).sum().cpu())
    finite_count = int(finite_sample.sum().cpu())
    if finite_count == 0:
        return {
            "signed_cosine_mean": float("nan"),
            "normalized_l2_mean": float("nan"),
            "symmetric_norm_ratio_mean": float("nan"),
            "valid_sample_count": 0,
            "cosine_valid_sample_count": 0,
            "distance_scale_valid_sample_count": 0,
            "both_zero_sample_count": 0,
            "one_zero_sample_count": 0,
            "finite_sample_count": 0,
            "nonfinite_sample_count": nonfinite_count,
            "total_sample_count": sample_count,
            "quality_passed": False,
            "cosine_quality_passed": False,
            "cosine_measurement_status": "inconclusive_nonfinite_paired_outputs",
            "measurement_status": "inconclusive_nonfinite_paired_outputs",
        }

    left_norm_square = torch.zeros(finite_count, dtype=torch.float32)
    right_norm_square = torch.zeros(finite_count, dtype=torch.float32)
    difference_norm_square = torch.zeros(finite_count, dtype=torch.float32)
    dot = torch.zeros(finite_count, dtype=torch.float32)
    for left_value, right_value in zip(left_parts, right_parts):
        x = left_value[finite_sample]
        y = right_value[finite_sample]
        left_norm_square += x.square().sum(dim=1)
        right_norm_square += y.square().sum(dim=1)
        difference_norm_square += (x - y).square().sum(dim=1)
        dot += (x * y).sum(dim=1)
    left_norm = left_norm_square.sqrt()
    right_norm = right_norm_square.sqrt()
    left_nonzero = left_norm > float(epsilon)
    right_nonzero = right_norm > float(epsilon)
    cosine_valid = left_nonzero & right_nonzero
    one_zero = left_nonzero ^ right_nonzero
    both_zero = ~(left_nonzero | right_nonzero)
    cosine_mean = (
        float((dot[cosine_valid] / (left_norm[cosine_valid] * right_norm[cosine_valid]).clamp_min(float(epsilon))).mean().cpu())
        if bool(cosine_valid.any())
        else float("nan")
    )
    denominator = 0.5 * (left_norm + right_norm)
    normalized_l2 = torch.zeros_like(denominator)
    nontrivial_distance = denominator > float(epsilon)
    normalized_l2[nontrivial_distance] = (
        difference_norm_square[nontrivial_distance].sqrt()
        / denominator[nontrivial_distance]
    )
    normalized_l2[one_zero] = 2.0
    maximum_norm = torch.maximum(left_norm, right_norm)
    norm_ratio = torch.ones_like(maximum_norm)
    nonzero_scale = maximum_norm > float(epsilon)
    norm_ratio[nonzero_scale] = (
        torch.minimum(left_norm[nonzero_scale], right_norm[nonzero_scale])
        / maximum_norm[nonzero_scale]
    )
    norm_ratio[one_zero] = 0.0
    quality_passed = nonfinite_count == 0
    cosine_quality_passed = bool(cosine_valid.any()) and quality_passed
    if bool(cosine_valid.any()):
        cosine_measurement_status = "completed"
    elif bool(both_zero.all()):
        cosine_measurement_status = "inconclusive_all_zero_outputs"
    else:
        cosine_measurement_status = "inconclusive_no_pair_with_two_nonzero_outputs"
    measurement_status = (
        "inconclusive_nonfinite_paired_outputs"
        if not quality_passed
        else (
            "completed"
            if cosine_quality_passed
            else "completed_distance_scale_metrics_cosine_inconclusive"
        )
    )
    return {
        "signed_cosine_mean": cosine_mean,
        "normalized_l2_mean": float(normalized_l2.mean().cpu()),
        "symmetric_norm_ratio_mean": float(norm_ratio.mean().cpu()),
        "valid_sample_count": int(cosine_valid.sum().cpu()),
        "cosine_valid_sample_count": int(cosine_valid.sum().cpu()),
        "distance_scale_valid_sample_count": finite_count,
        "both_zero_sample_count": int(both_zero.sum().cpu()),
        "one_zero_sample_count": int(one_zero.sum().cpu()),
        "finite_sample_count": finite_count,
        "nonfinite_sample_count": nonfinite_count,
        "total_sample_count": sample_count,
        "quality_passed": quality_passed,
        "cosine_quality_passed": cosine_quality_passed,
        "cosine_measurement_status": cosine_measurement_status,
        "measurement_status": measurement_status,
    }
