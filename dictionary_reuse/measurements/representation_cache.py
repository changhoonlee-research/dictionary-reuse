"""Exact-cache, validity, and Gram-matrix utilities for DiR measurements."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..interventions import forward_with_capture_and_interventions
from .representation_similarity import (
    _center_gram,
    _u_center_gram,
    paired_output_metrics,
)

def _load_feature_reference(reference: torch.Tensor | str | Path) -> torch.Tensor:
    """Load one exact cached feature chunk from RAM or the work-directory fallback."""

    if torch.is_tensor(reference):
        return reference
    return torch.load(Path(reference), map_location="cpu", weights_only=True)


def _save_feature_chunks(
    chunks: Sequence[torch.Tensor],
    *,
    root: Path,
    stem: str,
    backend: str = "disk",
    cache_tracker: dict[str, Any] | None = None,
) -> list[torch.Tensor | Path]:
    """Retain exact chunks in RAM up to a safe budget, then spill to work_dir."""

    mode = str(backend)
    if mode not in {"memory", "hybrid", "disk"}:
        raise ValueError(f"Unsupported DiR causal cache backend: {backend}")

    references: list[torch.Tensor | Path] = []
    tracker = cache_tracker if cache_tracker is not None else {}
    ram_budget = int(tracker.get("ram_budget_bytes", 0))
    ram_used = int(tracker.get("ram_bytes_used", 0))
    for chunk_index, chunk in enumerate(chunks):
        cpu_chunk = chunk.detach().cpu()
        chunk_bytes = int(cpu_chunk.numel() * cpu_chunk.element_size())
        retain_in_ram = mode == "memory" or (
            mode == "hybrid" and ram_used + chunk_bytes <= ram_budget
        )
        if retain_in_ram:
            references.append(cpu_chunk)
            ram_used += chunk_bytes
            tracker["ram_bytes_used"] = int(ram_used)
            tracker["ram_chunk_count"] = int(tracker.get("ram_chunk_count", 0)) + 1
            continue
        path = root / f"{stem}_chunk_{chunk_index:03d}.pt"
        torch.save(cpu_chunk, path)
        references.append(path)
        tracker["disk_bytes_written"] = int(tracker.get("disk_bytes_written", 0)) + int(
            path.stat().st_size
        )
        tracker["disk_chunk_count"] = int(tracker.get("disk_chunk_count", 0)) + 1
    return references


def _available_system_memory_bytes() -> int:
    """Best-effort Linux MemAvailable lookup without adding a runtime dependency."""

    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _select_exact_causal_cache_backend(*, estimated_bytes: int) -> dict[str, Any]:
    """Use exact RAM first and spill to work_dir only when conservative headroom requires it."""

    estimated = max(0, int(estimated_bytes))
    available = _available_system_memory_bytes()
    reserve = max(2 * 1024**3, int(math.ceil(0.50 * estimated)))
    ram_budget = max(0, min(estimated, available - reserve)) if available > 0 else 0
    if estimated == 0 or ram_budget >= estimated:
        backend = "memory"
    elif ram_budget >= 256 * 1024**2:
        backend = "hybrid"
    else:
        backend = "disk"
    return {
        "backend": backend,
        "estimated_raw_cache_bytes": estimated,
        "available_memory_bytes_at_suite_start": int(available),
        "required_memory_headroom_bytes": int(reserve),
        "ram_budget_bytes": int(ram_budget),
        "estimated_disk_spill_bytes": int(max(0, estimated - ram_budget)),
        "selection_contract": "exact_RAM_first_with_bounded_budget_then_exact_workdir_spill_no_quantization_no_extra_forward",
    }


def _estimate_exact_causal_raw_cache_bytes(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    depth: int,
    point_count: int,
    same_head: bool,
) -> int:
    """Estimate exact raw CLS+patch/pre+post cache bytes for both model sides."""

    response_sample_count = sum(int(images.shape[0]) for images, _labels, _ids in batches)
    embedding_dimension = int(getattr(model, "embedding_dimension"))
    patch_embedding = getattr(model, "patch_embedding")
    patch_count = int(getattr(patch_embedding, "number_of_patches"))
    token_count = patch_count + 1
    class_count = int(getattr(getattr(model, "classification_head"), "out_features", 100))
    return int(
        response_sample_count
        * int(depth)
        * int(point_count)
        * 2  # model sides
        * (2 * token_count * embedding_dimension + (class_count if same_head else 0))
        * 4
    )


@contextmanager
def _failsoft_temporary_directory(
    *,
    prefix: str,
    parent: Path | None,
    cleanup_status: dict[str, Any],
):
    """Create a bounded cache directory without letting cleanup erase results."""

    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(
            prefix=str(prefix),
            dir=str(parent) if parent is not None else None,
        )
    )
    cleanup_status.clear()
    cleanup_status.update({"status": "active", "path": str(root), "warning": ""})
    try:
        yield root
    finally:
        try:
            shutil.rmtree(root)
            cleanup_status.update({"status": "completed", "path": str(root), "warning": ""})
        except Exception as exc:
            cleanup_status.update(
                {
                    "status": "warning_cleanup_failed",
                    "path": str(root),
                    "warning": f"{type(exc).__name__}: {exc}",
                }
            )


def _feature_rms_and_variation_from_gram(
    gram: torch.Tensor,
    *,
    feature_dimension: int,
) -> tuple[float, float]:
    """Recover exact element RMS and across-sample variation from an exact Gram."""

    dimension = int(feature_dimension)
    sample_count = int(gram.shape[0])
    if dimension <= 0 or sample_count <= 0:
        raise ValueError("DiR feature moment recovery requires positive dimensions")
    value = gram.detach().double()
    trace = float(torch.diagonal(value).sum().cpu())
    total = float(value.sum().cpu())
    denominator = float(sample_count * dimension)
    signal_square = max(0.0, trace / denominator)
    centered_square_sum = max(0.0, trace - total / float(sample_count))
    variation_square = centered_square_sum / denominator
    return math.sqrt(signal_square), math.sqrt(max(0.0, variation_square))


def _streaming_cached_paired_output_metric_matrices(
    left_references: Mapping[str, Sequence[Sequence[torch.Tensor | str | Path]]],
    right_references: Mapping[str, Sequence[Sequence[torch.Tensor | str | Path]]],
    *,
    output_components: Mapping[str, Sequence[str]],
    sample_masks: Mapping[str, torch.Tensor | None] | None = None,
    epsilon: float = 1e-12,
) -> dict[str, dict[str, Any]]:
    """Compute paired diagnostics with one dot/norm pass per base component.

    Each cached raw feature chunk is loaded once. The expensive depth-by-depth
    dot products and norms are computed exactly once for every base component,
    then composite views such as ``full = CLS + patch`` are formed by adding
    those sufficient statistics. Per-output sample masks are applied only after
    component statistics exist, so different CLS/patch/full masks do not force
    repeated patch dot products.
    """

    if not output_components:
        return {}
    required_keys = sorted({key for keys in output_components.values() for key in keys})
    if not required_keys:
        return {}
    for key in required_keys:
        if key not in left_references or key not in right_references:
            raise KeyError(f"DiR paired cache missing feature key: {key}")
    depth = len(left_references[required_keys[0]])
    if depth < 1 or len(right_references[required_keys[0]]) != depth:
        raise ValueError("DiR paired cache depth mismatch")
    for key in required_keys:
        if len(left_references[key]) != depth or len(right_references[key]) != depth:
            raise ValueError(f"DiR paired cache depth mismatch for {key}")
    chunk_count = len(left_references[required_keys[0]][0])
    if chunk_count < 1:
        raise ValueError("DiR paired cache has no sample chunks")
    for key in required_keys:
        for block_index in range(depth):
            if len(left_references[key][block_index]) != chunk_count:
                raise ValueError(f"DiR left paired cache chunk mismatch for {key}")
            if len(right_references[key][block_index]) != chunk_count:
                raise ValueError(f"DiR right paired cache chunk mismatch for {key}")

    accumulator_names = (
        "cosine_sum",
        "normalized_l2_sum",
        "norm_ratio_sum",
        "cosine_valid_count",
        "finite_count",
        "nonfinite_count",
        "both_zero_count",
        "one_zero_count",
    )
    accumulators: dict[str, dict[str, np.ndarray]] = {
        output_name: {
            name: np.zeros(
                (depth, depth),
                dtype=(np.float64 if name.endswith("sum") else np.int64),
            )
            for name in accumulator_names
        }
        for output_name in output_components
    }
    total_selected = {name: 0 for name in output_components}
    masks = dict(sample_masks or {})
    sample_offset = 0

    for chunk_index in range(chunk_count):
        component_stats: dict[str, dict[str, torch.Tensor]] = {}
        chunk_sample_count: int | None = None

        # Raw component tensors are discarded immediately after their
        # sufficient statistics are computed. This bounds RAM and prevents a
        # large patch dot-product from being recomputed for full/patch views.
        for key in required_keys:
            left_blocks: list[torch.Tensor] = []
            right_blocks: list[torch.Tensor] = []
            for block_index in range(depth):
                left_tensor = _load_feature_reference(
                    left_references[key][block_index][chunk_index]
                )
                right_tensor = _load_feature_reference(
                    right_references[key][block_index][chunk_index]
                )
                left_blocks.append(left_tensor.float().reshape(left_tensor.shape[0], -1))
                right_blocks.append(right_tensor.float().reshape(right_tensor.shape[0], -1))
            current_count = int(left_blocks[0].shape[0])
            if any(
                int(value.shape[0]) != current_count
                for value in (*left_blocks, *right_blocks)
            ):
                raise ValueError("DiR paired cache sample chunk mismatch")
            if chunk_sample_count is None:
                chunk_sample_count = current_count
            elif int(chunk_sample_count) != current_count:
                raise ValueError("DiR paired cache cross-view sample chunk mismatch")

            left_value = torch.stack(left_blocks, dim=0).double()
            right_value = torch.stack(right_blocks, dim=0).double()
            component_stats[key] = {
                "left_norm_square": left_value.square().sum(dim=2),
                "right_norm_square": right_value.square().sum(dim=2),
                "dot": torch.einsum("ind,jnd->ijn", left_value, right_value),
                "left_finite": torch.isfinite(left_value).all(dim=2),
                "right_finite": torch.isfinite(right_value).all(dim=2),
            }
            del left_blocks, right_blocks, left_value, right_value

        if chunk_sample_count is None:
            raise RuntimeError("DiR paired cache chunk could not be loaded")

        for output_name, component_keys in output_components.items():
            mask = masks.get(output_name)
            if mask is None:
                local_mask = torch.ones(chunk_sample_count, dtype=torch.bool)
            else:
                local_mask = (
                    mask[sample_offset : sample_offset + chunk_sample_count]
                    .bool()
                    .cpu()
                )
                if int(local_mask.numel()) != chunk_sample_count:
                    raise ValueError("DiR paired cache sample mask length mismatch")
            selected_count = int(local_mask.sum())
            total_selected[output_name] += selected_count
            if selected_count == 0:
                continue

            left_norm_square = torch.zeros(depth, selected_count, dtype=torch.float64)
            right_norm_square = torch.zeros(depth, selected_count, dtype=torch.float64)
            dot = torch.zeros(depth, depth, selected_count, dtype=torch.float64)
            left_finite = torch.ones(depth, selected_count, dtype=torch.bool)
            right_finite = torch.ones(depth, selected_count, dtype=torch.bool)
            for key in component_keys:
                stats = component_stats[key]
                left_norm_square += stats["left_norm_square"][:, local_mask]
                right_norm_square += stats["right_norm_square"][:, local_mask]
                dot += stats["dot"][:, :, local_mask]
                left_finite &= stats["left_finite"][:, local_mask]
                right_finite &= stats["right_finite"][:, local_mask]

            finite = left_finite[:, None, :] & right_finite[None, :, :]
            left_norm = left_norm_square.clamp_min(0.0).sqrt()
            right_norm = right_norm_square.clamp_min(0.0).sqrt()
            left_nonzero = left_norm > float(epsilon)
            right_nonzero = right_norm > float(epsilon)
            cosine_valid = finite & left_nonzero[:, None, :] & right_nonzero[None, :, :]
            one_zero = finite & (left_nonzero[:, None, :] ^ right_nonzero[None, :, :])
            both_zero = finite & ~(left_nonzero[:, None, :] | right_nonzero[None, :, :])

            left_norm_pair = left_norm[:, None, :]
            right_norm_pair = right_norm[None, :, :]
            cosine = dot / (left_norm_pair * right_norm_pair).clamp_min(float(epsilon))
            difference_square = (
                left_norm_square[:, None, :]
                + right_norm_square[None, :, :]
                - 2.0 * dot
            ).clamp_min(0.0)
            denominator = 0.5 * (left_norm_pair + right_norm_pair)
            normalized_l2 = torch.zeros_like(difference_square)
            nontrivial_distance = finite & (denominator > float(epsilon))
            normalized_l2[nontrivial_distance] = (
                difference_square[nontrivial_distance].sqrt()
                / denominator[nontrivial_distance]
            )
            normalized_l2[one_zero] = 2.0

            maximum_norm = torch.maximum(left_norm_pair, right_norm_pair)
            minimum_norm = torch.minimum(left_norm_pair, right_norm_pair)
            norm_ratio = torch.ones_like(maximum_norm.expand(depth, depth, selected_count))
            nonzero_scale = finite & (maximum_norm > float(epsilon))
            expanded_minimum = minimum_norm.expand(depth, depth, selected_count)
            expanded_maximum = maximum_norm.expand(depth, depth, selected_count)
            norm_ratio[nonzero_scale] = (
                expanded_minimum[nonzero_scale] / expanded_maximum[nonzero_scale]
            )
            norm_ratio[one_zero] = 0.0

            accumulator = accumulators[output_name]
            accumulator["cosine_sum"] += torch.where(
                cosine_valid, cosine, torch.zeros_like(cosine)
            ).sum(dim=2).numpy()
            accumulator["normalized_l2_sum"] += torch.where(
                finite, normalized_l2, torch.zeros_like(normalized_l2)
            ).sum(dim=2).numpy()
            accumulator["norm_ratio_sum"] += torch.where(
                finite, norm_ratio, torch.zeros_like(norm_ratio)
            ).sum(dim=2).numpy()
            accumulator["cosine_valid_count"] += cosine_valid.sum(dim=2).numpy().astype(np.int64)
            accumulator["finite_count"] += finite.sum(dim=2).numpy().astype(np.int64)
            accumulator["nonfinite_count"] += (~finite).sum(dim=2).numpy().astype(np.int64)
            accumulator["both_zero_count"] += both_zero.sum(dim=2).numpy().astype(np.int64)
            accumulator["one_zero_count"] += one_zero.sum(dim=2).numpy().astype(np.int64)
        sample_offset += chunk_sample_count

    output: dict[str, dict[str, Any]] = {}
    for output_name, accumulator in accumulators.items():
        cosine_count = accumulator["cosine_valid_count"]
        finite_count = accumulator["finite_count"]
        cosine_mean = np.full((depth, depth), np.nan, dtype=np.float64)
        normalized_l2_mean = np.full((depth, depth), np.nan, dtype=np.float64)
        norm_ratio_mean = np.full((depth, depth), np.nan, dtype=np.float64)
        cosine_mask = cosine_count > 0
        finite_mask = finite_count > 0
        cosine_mean[cosine_mask] = (
            accumulator["cosine_sum"][cosine_mask] / cosine_count[cosine_mask]
        )
        normalized_l2_mean[finite_mask] = (
            accumulator["normalized_l2_sum"][finite_mask] / finite_count[finite_mask]
        )
        norm_ratio_mean[finite_mask] = (
            accumulator["norm_ratio_sum"][finite_mask] / finite_count[finite_mask]
        )
        output[output_name] = {
            "signed_cosine_mean_12x12": cosine_mean.tolist(),
            "normalized_l2_mean_12x12": normalized_l2_mean.tolist(),
            "symmetric_norm_ratio_mean_12x12": norm_ratio_mean.tolist(),
            "cosine_valid_sample_count_12x12": cosine_count.tolist(),
            "distance_scale_valid_sample_count_12x12": finite_count.tolist(),
            "nonfinite_paired_output_sample_count_12x12": accumulator["nonfinite_count"].tolist(),
            "both_zero_paired_output_sample_count_12x12": accumulator["both_zero_count"].tolist(),
            "one_zero_paired_output_sample_count_12x12": accumulator["one_zero_count"].tolist(),
            "selected_sample_count": int(total_selected[output_name]),
            "cache_read_contract": (
                "each_raw_feature_chunk_loaded_once_and_each_base_component_"
                "dot_norm_computed_once_then_composite_views_reuse_sufficient_statistics"
            ),
        }
        output[output_name]["quality_passed"] = bool(
            np.all(finite_count == int(total_selected[output_name]))
        )
    return output


def pairwise_paired_output_metric_matrices(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> dict[str, list[list[float]]]:
    names = ("signed_cosine_mean", "normalized_l2_mean", "symmetric_norm_ratio_mean")
    count_names = (
        "cosine_valid_sample_count",
        "distance_scale_valid_sample_count",
        "both_zero_sample_count",
        "one_zero_sample_count",
        "finite_sample_count",
        "nonfinite_sample_count",
        "total_sample_count",
    )
    output = {name: [] for name in (*names, *count_names)}
    for left_value in left:
        rows = {name: [] for name in (*names, *count_names)}
        for right_value in right:
            metrics = paired_output_metrics(left_value, right_value)
            for name in names:
                rows[name].append(float(metrics[name]))
            for name in count_names:
                rows[name].append(int(metrics[name]))
        for name in (*names, *count_names):
            output[name].append(rows[name])
    return {
        "signed_cosine_12x12": output["signed_cosine_mean"],
        "normalized_l2_12x12": output["normalized_l2_mean"],
        "symmetric_norm_ratio_12x12": output["symmetric_norm_ratio_mean"],
        "cosine_valid_sample_count_12x12": output["cosine_valid_sample_count"],
        "distance_scale_valid_sample_count_12x12": output[
            "distance_scale_valid_sample_count"
        ],
        "both_zero_sample_count_12x12": output["both_zero_sample_count"],
        "one_zero_sample_count_12x12": output["one_zero_sample_count"],
        "finite_sample_count_12x12": output["finite_sample_count"],
        "nonfinite_sample_count_12x12": output["nonfinite_sample_count"],
        "total_sample_count_12x12": output["total_sample_count"],
        "finite_output_validity_mask_12x12": (
            np.asarray(output["nonfinite_sample_count"], dtype=np.int64) == 0
        ).tolist(),
        "cosine_validity_mask_12x12": (
            (np.asarray(output["nonfinite_sample_count"], dtype=np.int64) == 0)
            & (np.asarray(output["cosine_valid_sample_count"], dtype=np.int64) > 0)
        ).tolist(),
        "quality_passed": not bool(
            np.asarray(output["nonfinite_sample_count"], dtype=np.int64).any()
        ),
        "cosine_quality_passed": bool(
            (np.asarray(output["cosine_valid_sample_count"], dtype=np.int64) > 0).all()
        ) and not bool(
            np.asarray(output["nonfinite_sample_count"], dtype=np.int64).any()
        ),
        "measurement_status": (
            "inconclusive_nonfinite_paired_outputs"
            if bool(np.asarray(output["nonfinite_sample_count"], dtype=np.int64).any())
            else (
                "completed"
                if bool((np.asarray(output["cosine_valid_sample_count"], dtype=np.int64) > 0).all())
                else "completed_distance_scale_metrics_some_cosine_cells_inconclusive"
            )
        ),
    }


def signal_validity(
    rms_values: Sequence[float],
    *,
    absolute_minimum: float = 1e-8,
    relative_to_median: float = 0.05,
) -> dict[str, Any]:
    array = np.asarray(rms_values, dtype=np.float64)
    finite_positive = array[np.isfinite(array) & (array > 0)]
    median = float(np.median(finite_positive)) if finite_positive.size else 0.0
    threshold = max(float(absolute_minimum), float(relative_to_median) * median)
    valid = np.isfinite(array) & (array >= threshold)
    return {
        "rms": array.tolist(),
        "median_positive_rms": median,
        "absolute_minimum": float(absolute_minimum),
        "relative_to_median": float(relative_to_median),
        "threshold": threshold,
        "valid_by_block": valid.tolist(),
        "valid_block_count": int(valid.sum()),
    }


def sample_variation_rms(value: torch.Tensor) -> float:
    """RMS variation across paired samples after removing the sample mean.

    A non-zero activation can still be useless for CKA when it is identical for
    every sample. This quantity detects that degenerate case independently of
    the ordinary activation/effect RMS.
    """

    feature = value.detach().float().reshape(value.shape[0], -1)
    centered = feature - feature.mean(dim=0, keepdim=True)
    return float(centered.square().mean().sqrt().cpu())


def _component_signal_rms(components: Sequence[torch.Tensor]) -> float:
    """Exact RMS of a virtual concatenation without materializing it."""

    if not components:
        raise ValueError("DiR component RMS requires at least one component")
    square_sum = 0.0
    element_count = 0
    for value in components:
        feature = value.detach().float().reshape(value.shape[0], -1)
        square_sum += float(feature.square().sum().cpu())
        element_count += int(feature.numel())
    if element_count <= 0:
        return 0.0
    return float((square_sum / float(element_count)) ** 0.5)


def _component_sample_variation_rms(components: Sequence[torch.Tensor]) -> float:
    """Sample-variation RMS of a virtual concatenation without a large copy."""

    if not components:
        raise ValueError("DiR component variation requires at least one component")
    centered_square_sum = 0.0
    element_count = 0
    sample_count = int(components[0].shape[0])
    for value in components:
        feature = value.detach().float().reshape(value.shape[0], -1)
        if int(feature.shape[0]) != sample_count:
            raise ValueError("DiR component variation sample counts differ")
        centered = feature - feature.mean(dim=0, keepdim=True)
        centered_square_sum += float(centered.square().sum().cpu())
        element_count += int(centered.numel())
        del centered
    if element_count <= 0:
        return 0.0
    return float((centered_square_sum / float(element_count)) ** 0.5)


def gram_variation_strength(gram: torch.Tensor) -> float:
    """Scale-aware U-centered Gram energy used to validate CKA estimability."""

    centered = _u_center_gram(gram)
    sample_count = max(1, int(centered.shape[0]))
    return float(centered.square().mean().sqrt().cpu()) / float(sample_count)


def combined_signal_variation_validity(
    signal_values: Sequence[float],
    variation_values: Sequence[float],
    *,
    absolute_minimum: float,
    relative_to_median: float,
) -> dict[str, Any]:
    if len(signal_values) != len(variation_values):
        raise ValueError("DiR signal and sample-variation vectors must have equal length")
    signal = signal_validity(
        signal_values,
        absolute_minimum=float(absolute_minimum),
        relative_to_median=float(relative_to_median),
    )
    variation = signal_validity(
        variation_values,
        absolute_minimum=float(absolute_minimum),
        relative_to_median=float(relative_to_median),
    )
    valid = [
        bool(signal_value and variation_value)
        for signal_value, variation_value in zip(
            signal["valid_by_block"], variation["valid_by_block"]
        )
    ]
    return {
        "signal": signal,
        "sample_variation": variation,
        "valid_by_block": valid,
        "valid_block_count": int(sum(valid)),
        "contract": "nonzero_effect_and_nonconstant_across_samples_required",
    }


def recovery_validity(
    signal_valid_by_block: Sequence[bool],
    mean_clean_distance_reduction_fraction: Sequence[float],
    median_clean_distance_reduction_fraction: Sequence[float],
    positive_recovery_sample_fraction: Sequence[float],
    mean_clean_target_projection_fraction: Sequence[float],
    *,
    minimum_block_recovery_fraction: float,
    minimum_median_recovery_fraction: float,
    minimum_positive_recovery_sample_fraction: float,
) -> list[bool]:
    if not (
        len(signal_valid_by_block)
        == len(mean_clean_distance_reduction_fraction)
        == len(median_clean_distance_reduction_fraction)
        == len(positive_recovery_sample_fraction)
        == len(mean_clean_target_projection_fraction)
    ):
        raise ValueError("DiR recovery-validity vectors must have equal length")
    return [
        bool(
            signal
            and np.isfinite(fraction)
            and np.isfinite(median_fraction)
            and np.isfinite(positive_fraction)
            and np.isfinite(projection)
            and float(fraction) >= float(minimum_block_recovery_fraction)
            and float(median_fraction) > float(minimum_median_recovery_fraction)
            and float(positive_fraction) >= float(minimum_positive_recovery_sample_fraction)
            and float(projection) > 0.0
        )
        for signal, fraction, median_fraction, positive_fraction, projection in zip(
            signal_valid_by_block,
            mean_clean_distance_reduction_fraction,
            median_clean_distance_reduction_fraction,
            positive_recovery_sample_fraction,
            mean_clean_target_projection_fraction,
        )
    ]


def outer_validity_mask(left_valid: Sequence[bool], right_valid: Sequence[bool]) -> list[list[bool]]:
    left = np.asarray(left_valid, dtype=bool)
    right = np.asarray(right_valid, dtype=bool)
    return np.logical_and(left[:, None], right[None, :]).tolist()


def _gram_from_feature_chunks(
    chunks: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    sizes = [int(chunk.shape[0]) for chunk in chunks]
    offsets = np.cumsum([0, *sizes]).tolist()
    total = int(sum(sizes))
    gram = torch.empty(total, total, dtype=torch.float64)
    for row_index, row_cpu in enumerate(chunks):
        row = row_cpu.to(device=device, dtype=torch.float32)
        for column_index in range(row_index, len(chunks)):
            column = chunks[column_index].to(device=device, dtype=torch.float32)
            value = (row @ column.T).double().cpu()
            row_slice = slice(offsets[row_index], offsets[row_index + 1])
            column_slice = slice(offsets[column_index], offsets[column_index + 1])
            gram[row_slice, column_slice] = value
            if row_index != column_index:
                gram[column_slice, row_slice] = value.T
            del column, value
        del row
    return gram


def _native_update_grams_and_norms(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    capture_block_group_size: int = 3,
) -> dict[str, Any]:
    """Compute exact sample Grams with bounded grouped activation capture."""

    model.eval().to(device)
    depth = len(model.transformer_blocks)
    group_size = max(1, min(int(capture_block_group_size), depth))
    output: dict[str, Any] = {
        "full_token_grams": [None] * depth,
        "cls_grams": [None] * depth,
        "patch_grams": [None] * depth,
        "full_token_norms": [None] * depth,
        "cls_norms": [None] * depth,
        "patch_norms": [None] * depth,
        "capture_forward_count": 0,
        "capture_contract": "bounded_grouped_block_capture_forward_per_batch",
        "capture_block_group_size": group_size,
        "capture_group_count": int(math.ceil(depth / group_size)),
    }
    with torch.no_grad():
        for group_start in range(0, depth, group_size):
            block_indices = list(range(group_start, min(depth, group_start + group_size)))
            cls_chunks = {index: [] for index in block_indices}
            patch_chunks = {index: [] for index in block_indices}
            full_norms = {index: [] for index in block_indices}
            cls_norms = {index: [] for index in block_indices}
            patch_norms = {index: [] for index in block_indices}
            capture_points = [f"block_{index:02d}_update" for index in block_indices]
            for images_cpu, _labels, _ids in batches:
                _logits, taps = forward_with_capture_and_interventions(
                    model, images_cpu.to(device), capture_points=capture_points
                )
                output["capture_forward_count"] += 1
                for block_index, point in zip(block_indices, capture_points):
                    update = taps[point].detach().float()
                    cls = update[:, 0].reshape(update.shape[0], -1)
                    patch = update[:, 1:].reshape(update.shape[0], -1)
                    cls_chunks[block_index].append(cls.cpu())
                    patch_chunks[block_index].append(patch.cpu())
                    cls_squared = cls.square().sum(dim=1)
                    patch_squared = patch.square().sum(dim=1)
                    cls_norms[block_index].append(cls_squared.sqrt().cpu())
                    patch_norms[block_index].append(patch_squared.sqrt().cpu())
                    full_norms[block_index].append(
                        (cls_squared + patch_squared).sqrt().cpu()
                    )
                    del update, cls, patch, cls_squared, patch_squared
                del taps, _logits
            for block_index in block_indices:
                cls_gram = _gram_from_feature_chunks(
                    cls_chunks[block_index], device=device
                )
                patch_gram = _gram_from_feature_chunks(
                    patch_chunks[block_index], device=device
                )
                output["cls_grams"][block_index] = cls_gram
                output["patch_grams"][block_index] = patch_gram
                output["full_token_grams"][block_index] = cls_gram + patch_gram
                output["full_token_norms"][block_index] = float(
                    torch.cat(full_norms[block_index]).mean()
                )
                output["cls_norms"][block_index] = float(
                    torch.cat(cls_norms[block_index]).mean()
                )
                output["patch_norms"][block_index] = float(
                    torch.cat(patch_norms[block_index]).mean()
                )
            del cls_chunks, patch_chunks, full_norms, cls_norms, patch_norms
    return output


def _pairwise_gram_cka_matrix(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    *,
    invalid_as_nan: bool = False,
    epsilon: float = 1e-12,
) -> list[list[float]]:
    """Pairwise U-centered CKA without re-centering the same Gram repeatedly.

    The right-side U-centered Grams are cached once for the duration of this
    single matrix call. Each left Gram is centered once and released after its
    row is complete. This cuts U-centering from O(L*R) to O(L+R) while keeping
    temporary memory bounded to one left matrix plus the right-side matrices.
    """

    if not invalid_as_nan:
        # Keep the public numerical contract identical to debiased_gram_cka.
        # The cached path below uses the same U-centering and denominator rule.
        invalid_as_nan = True

    right_cache: list[tuple[torch.Tensor, float]] = []
    for right_value in right:
        centered = _u_center_gram(right_value)
        norm = float(centered.square().sum().sqrt().cpu())
        right_cache.append((centered, norm))

    output: list[list[float]] = []
    for left_value in left:
        left_centered = _u_center_gram(left_value)
        left_norm = float(left_centered.square().sum().sqrt().cpu())
        row: list[float] = []
        for right_centered, right_norm in right_cache:
            if left_norm <= float(epsilon) or right_norm <= float(epsilon):
                row.append(float("nan"))
                continue
            row.append(
                float(
                    (
                        (left_centered * right_centered).sum()
                        / (left_norm * right_norm)
                    ).cpu()
                )
            )
        output.append(row)
        del left_centered
    return output


def _finite_elementwise_mean(values: Sequence[np.ndarray]) -> np.ndarray:
    """Elementwise mean over finite contributions only; all-invalid stays NaN."""

    if not values:
        raise ValueError("DiR finite elementwise mean requires at least one array")
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays):
        raise ValueError("DiR finite elementwise mean shape mismatch")
    stacked = np.stack(arrays, axis=0)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=0)
    total = np.where(finite, stacked, 0.0).sum(axis=0)
    output = np.full(shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=output, where=count > 0)
    return output


def _pairwise_biased_gram_cka_matrix(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-12,
) -> list[list[float]]:
    """Pairwise biased Gram CKA with centering cached once per block."""

    right_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for right_value in right:
        centered = _center_gram(right_value)
        right_cache.append((centered, centered.square().sum()))

    output: list[list[float]] = []
    for left_value in left:
        left_centered = _center_gram(left_value)
        left_squared_norm = left_centered.square().sum()
        row: list[float] = []
        for right_centered, right_squared_norm in right_cache:
            if tuple(left_centered.shape) != tuple(right_centered.shape):
                raise ValueError("Gram shapes differ")
            denominator = torch.sqrt(
                left_squared_norm * right_squared_norm
            ).clamp_min(float(epsilon))
            row.append(
                float(
                    ((left_centered * right_centered).sum() / denominator).cpu()
                )
            )
        output.append(row)
    return output
