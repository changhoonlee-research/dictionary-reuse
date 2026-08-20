"""Shared Jacobian probe, rank, and status utilities."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch



def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        stop = start + 1
        while stop < flat.size and flat[order[stop]] == flat[order[start]]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks

def _flat_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_flat.size != right_flat.size or left_flat.size < 2:
        return float("nan")
    finite = np.isfinite(left_flat) & np.isfinite(right_flat)
    if int(finite.sum()) < 2:
        return float("nan")
    left_rank = _average_tie_ranks(left_flat[finite])
    right_rank = _average_tie_ranks(right_flat[finite])
    if float(left_rank.std()) == 0.0 or float(right_rank.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])

def _rademacher(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    normalize_per_sample: bool = True,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    normalized_shape = tuple(int(x) for x in shape)
    value = torch.randint(0, 2, normalized_shape, generator=generator, device=device)
    value = value.to(dtype=dtype).mul_(2).sub_(1)
    if normalize_per_sample and len(normalized_shape) >= 2:
        event_size = int(np.prod(normalized_shape[1:]))
    else:
        event_size = int(np.prod(normalized_shape))
    return value / math.sqrt(max(1, event_size))
def _shared_probe_projection_seed(
    probe_seed: int,
    probe_index: int,
    *,
    family_offset: int,
    sample_offset: int = 0,
) -> int:
    """Return a block-index-independent seed shared by all compared blocks/models."""

    return (
        int(probe_seed)
        + int(family_offset)
        + 1000003 * int(probe_index)
        + int(sample_offset)
    )
def _shared_input_probe_direction(
    images: torch.Tensor,
    *,
    seed: int,
    normalization_std: Sequence[float],
) -> torch.Tensor:
    """One raw-pixel probe broadcast identically across every batch sample."""

    shared = _rademacher(
        (1, *images.shape[1:]),
        seed=int(seed),
        device=images.device,
        dtype=images.dtype,
    ).expand_as(images)
    std_tensor = torch.tensor(
        normalization_std, device=images.device, dtype=images.dtype
    ).view(1, int(images.shape[1]), 1, 1)
    return shared / std_tensor
def _numerical_svd_rank(
    singular_values: torch.Tensor,
    *,
    row_count: int,
    column_count: int,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-12,
) -> tuple[int, float]:
    """Return a conservative numerical rank for float32-derived sketches."""

    values = singular_values.detach().double()
    sigma_max = float(values[0].abs().cpu()) if values.numel() else 0.0
    tolerance = max(
        float(absolute_tolerance),
        float(relative_tolerance) * sigma_max,
        torch.finfo(torch.float64).eps
        * max(int(row_count), int(column_count))
        * sigma_max,
    )
    rank = int((values > float(tolerance)).sum().item())
    return rank, float(tolerance)
def _finalize_randomized_svd_descriptor(
    range_matrix: torch.Tensor,
    vjp_rows: torch.Tensor,
    *,
    target_rank: int,
    range_basis: torch.Tensor | None = None,
    relative_rank_tolerance: float = 1e-6,
    absolute_rank_tolerance: float = 1e-12,
    zero_signal_rms_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Finalize randomized SVD from ``Y=JΩ`` and ``B=QᵀJ`` without forming J.

    Responses below the declared detection threshold and numerically rank-zero
    reduced operators remain rank zero and are recorded explicitly. They are not
    treated as proof that the full Jacobian is exactly zero. Null-space
    directions are never padded into the reported singular subspaces.
    """

    value = range_matrix.detach().double()
    if value.ndim != 2 or int(value.shape[1]) < 1:
        raise ValueError("DiR randomized Jacobian range matrix must be 2-D and nonempty")
    if not torch.isfinite(value).all():
        raise ValueError("DiR randomized Jacobian range sketch is non-finite")
    range_response_rms = float(value.square().mean().sqrt().cpu())
    range_u, range_s, _range_vh = torch.linalg.svd(value, full_matrices=False)
    if not torch.isfinite(range_s).all():
        raise ValueError("DiR randomized Jacobian range sketch is non-finite")
    numerical_range_rank, range_tolerance = _numerical_svd_rank(
        range_s,
        row_count=int(value.shape[0]),
        column_count=int(value.shape[1]),
        relative_tolerance=float(relative_rank_tolerance),
        absolute_tolerance=float(absolute_rank_tolerance),
    )
    below_detection_range = range_response_rms <= float(zero_signal_rms_tolerance)
    if below_detection_range:
        numerical_range_rank = 0
    if range_basis is None:
        q = range_u[:, :numerical_range_rank].contiguous()
    else:
        q = range_basis.detach().double().contiguous()
        if tuple(q.shape) != (int(value.shape[0]), numerical_range_rank):
            raise ValueError("DiR randomized Jacobian range basis shape is inconsistent")

    b = vjp_rows.detach().double()
    input_dimension = int(b.shape[1]) if b.ndim == 2 else 0
    if b.ndim != 2 or tuple(b.shape) != (numerical_range_rank, input_dimension):
        raise ValueError("DiR randomized Jacobian VJP row shape is inconsistent")

    padded = torch.zeros(int(target_rank), dtype=torch.float64)
    if numerical_range_rank == 0:
        return {
            "status": (
                "completed_below_detection_jacobian"
                if below_detection_range
                else "completed_numerical_rank_zero_jacobian"
            ),
            "valid": True,
            "degenerate": True,
            "degenerate_reason": (
                "below_detection_range_response"
                if below_detection_range
                else "detected_range_numerical_rank_zero"
            ),
            "singular_values": padded,
            "input_subspace": torch.empty(input_dimension, 0, dtype=torch.float64),
            "output_subspace": torch.empty(int(value.shape[0]), 0, dtype=torch.float64),
            "rank_used": 0,
            "numerical_range_rank": 0,
            "numerical_operator_rank": 0,
            "range_rank_tolerance": float(range_tolerance),
            "operator_rank_tolerance": float(absolute_rank_tolerance),
            "range_singular_values": range_s.cpu(),
            "range_response_rms": float(range_response_rms),
            "below_detection_signal_rms_threshold": float(zero_signal_rms_tolerance),
        }

    u_hat, singular_values, vh = torch.linalg.svd(b, full_matrices=False)
    if not torch.isfinite(singular_values).all():
        raise ValueError("DiR randomized Jacobian reduced operator is non-finite")
    numerical_operator_rank, operator_tolerance = _numerical_svd_rank(
        singular_values,
        row_count=int(b.shape[0]),
        column_count=int(b.shape[1]),
        relative_tolerance=float(relative_rank_tolerance),
        absolute_tolerance=float(absolute_rank_tolerance),
    )
    rank_used = min(
        int(target_rank),
        int(numerical_operator_rank),
        int(q.shape[1]),
        int(vh.shape[0]),
    )
    if rank_used == 0:
        return {
            "status": "completed_numerical_rank_zero_jacobian",
            "valid": True,
            "degenerate": True,
            "degenerate_reason": "detected_reduced_operator_numerical_rank_zero",
            "singular_values": padded,
            "input_subspace": torch.empty(input_dimension, 0, dtype=torch.float64),
            "output_subspace": torch.empty(int(value.shape[0]), 0, dtype=torch.float64),
            "rank_used": 0,
            "numerical_range_rank": int(numerical_range_rank),
            "numerical_operator_rank": int(numerical_operator_rank),
            "range_rank_tolerance": float(range_tolerance),
            "operator_rank_tolerance": float(operator_tolerance),
            "range_singular_values": range_s.cpu(),
            "range_response_rms": float(range_response_rms),
            "below_detection_signal_rms_threshold": float(zero_signal_rms_tolerance),
        }

    leading = singular_values[:rank_used].clamp_min(0)
    output_subspace = (q @ u_hat[:, :rank_used]).contiguous()
    input_subspace = vh[:rank_used].T.contiguous()
    padded[:rank_used] = leading.cpu()
    return {
        "status": "completed",
        "valid": True,
        "degenerate": False,
        "degenerate_reason": None,
        "singular_values": padded,
        "input_subspace": input_subspace.cpu(),
        "output_subspace": output_subspace.cpu(),
        "rank_used": int(rank_used),
        "numerical_range_rank": int(numerical_range_rank),
        "numerical_operator_rank": int(numerical_operator_rank),
        "range_rank_tolerance": float(range_tolerance),
        "operator_rank_tolerance": float(operator_tolerance),
        "range_singular_values": range_s.cpu(),
        "range_response_rms": float(range_response_rms),
        "below_detection_signal_rms_threshold": float(zero_signal_rms_tolerance),
    }
def _measurement_status_from_primary_validity(
    valid_flags: Sequence[bool], *, no_valid_status: str
) -> str:
    """Summarize completed primary-view computation without hiding low signal."""

    flags = [bool(value) for value in valid_flags]
    if flags and all(flags):
        return "completed"
    if any(flags):
        return "partial_primary_views"
    return str(no_valid_status)
