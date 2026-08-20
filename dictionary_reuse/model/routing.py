"""Entmax routing primitives and rank/statistical routing helpers."""

from __future__ import annotations

import math
from typing import Any

import torch

_ENTMAX_COMPILED_KERNEL_CACHE: dict[int, Any] = {}
_ENTMAX_COMPILED_KERNEL_FAILURES: set[int] = set()
_ENTMAX_TEMPORARY_EAGER_COUNTS: set[int] = set()
_ENTMAX_COMPILED_VALIDATION_KEYS: set[tuple[int, str, str, float]] = set()
_ENTMAX_RUNTIME_STATUS: dict[int, dict[str, Any]] = {}

def reset_entmax_runtime_state() -> None:
    """Reset compile/fallback/validation state for one scientific run."""

    _ENTMAX_COMPILED_KERNEL_CACHE.clear()
    _ENTMAX_COMPILED_KERNEL_FAILURES.clear()
    _ENTMAX_TEMPORARY_EAGER_COUNTS.clear()
    _ENTMAX_COMPILED_VALIDATION_KEYS.clear()
    _ENTMAX_RUNTIME_STATUS.clear()

def request_entmax_runtime_revalidation() -> None:
    """Recheck one compiled/eager sample at the next generic-entmax call."""

    _ENTMAX_COMPILED_VALIDATION_KEYS.clear()

def _tensor_quantile_float(values: torch.Tensor, quantile: float) -> float:
    detached = values.detach().float().flatten()
    if int(detached.numel()) <= 0:
        return 0.0
    sorted_values, _ = torch.sort(detached)
    index = int(math.ceil(float(quantile) * float(int(sorted_values.numel()))) - 1)
    index = max(0, min(index, int(sorted_values.numel()) - 1))
    return float(sorted_values[index].cpu())

def _mass95_atoms_from_sample_distribution(
    probability: torch.Tensor,
    *,
    mass_target: float = 0.95,
) -> tuple[float, float]:
    detached = probability.detach().float().reshape(-1, int(probability.shape[-1]))
    if int(detached.numel()) <= 0 or int(detached.shape[0]) <= 0:
        return 0.0, 0.0
    sorted_probability, _ = torch.sort(detached, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probability, dim=-1)
    counts = (cumulative < float(mass_target)).sum(dim=-1).to(dtype=torch.float32) + 1.0
    return float(counts.mean().cpu()), _tensor_quantile_float(counts, 0.95)

def _route_distribution_neff_and_mass95_tensors(
    mass: torch.Tensor,
    *,
    mass_target: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return on-device N_eff and mass95 count for a 1-D route-mass vector."""

    values = mass.detach().float().reshape(-1)
    if int(values.numel()) <= 0:
        zero = torch.zeros((), device=mass.device, dtype=torch.float32)
        return zero, zero
    probability = values / values.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    neff = torch.exp(entropy)
    sorted_probability, _ = torch.sort(probability, descending=True)
    cumulative = torch.cumsum(sorted_probability, dim=0)
    mass95 = (cumulative < float(mass_target)).sum().to(dtype=torch.float32) + 1.0
    return neff.detach(), mass95.detach()

def _scalar_report_float(value: Any, default: float = 0.0) -> float:
    """Convert a report scalar to float; CUDA synchronization happens only at report time."""

    if isinstance(value, torch.Tensor):
        if int(value.numel()) <= 0:
            return float(default)
        return float(value.detach().float().reshape(-1).mean().cpu())
    if value is None:
        return float(default)
    return float(value)

def _sparsemax_last_dim(logits: torch.Tensor) -> torch.Tensor:
    """Sparsemax over the final dimension for one or more score rows."""

    values = logits.float()
    if int(values.numel()) <= 0 or int(values.shape[-1]) <= 0:
        return values
    sorted_values, _ = torch.sort(values, dim=-1, descending=True)
    atom_count = int(values.shape[-1])
    range_values = torch.arange(
        1,
        atom_count + 1,
        device=values.device,
        dtype=values.dtype,
    )
    view_shape = (*([1] * max(0, int(values.ndim) - 1)), atom_count)
    range_values = range_values.reshape(view_shape)
    cumulative = torch.cumsum(sorted_values, dim=-1)
    support = 1.0 + range_values * sorted_values > cumulative
    support_count = support.to(dtype=values.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
    tau_index = support_count.to(dtype=torch.long).sub(1)
    tau = (cumulative.gather(-1, tau_index) - 1.0) / support_count
    return (values - tau).clamp_min(0.0)

def _entmax_generic_bisection_eager(
    values: torch.Tensor,
    alpha_tensor: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """Generic entmax bisection over the final dimension with no host sync."""

    scaled = (alpha_tensor - 1.0) * values
    power = 1.0 / (alpha_tensor - 1.0)
    lower = scaled.amin(dim=-1, keepdim=True) - 1.0
    upper = scaled.amax(dim=-1, keepdim=True)
    for _ in range(max(1, int(iterations))):
        tau = (lower + upper) / 2.0
        probability = (scaled - tau).clamp_min(0.0).pow(power)
        too_large = probability.sum(dim=-1, keepdim=True) > 1.0
        lower = torch.where(too_large, tau, lower)
        upper = torch.where(too_large, upper, tau)
    probability = (scaled - upper).clamp_min(0.0).pow(power)
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)

def _entmax_runtime_state(iterations: int) -> dict[str, Any]:
    count = max(1, int(iterations))
    state = _ENTMAX_RUNTIME_STATUS.get(count)
    if state is None:
        state = {
            "compile_requested": False,
            "compile_attempted": False,
            "compile_wrapper_created": False,
            "compile_succeeded": False,
            "compiled_active": False,
            "validation_count": 0,
            "fallback_count": 0,
            "compiled_eager_max_abs_error": 0.0,
            "compiled_eager_support_mismatch_count": 0,
            "fallback_reason": "",
        }
        _ENTMAX_RUNTIME_STATUS[count] = state
    return state

def _disable_compiled_entmax(iterations: int, *, reason: str) -> None:
    count = max(1, int(iterations))
    state = _entmax_runtime_state(count)
    if count not in _ENTMAX_COMPILED_KERNEL_FAILURES:
        state["fallback_count"] = int(state.get("fallback_count", 0)) + 1
    state["compiled_active"] = False
    state["fallback_reason"] = str(reason)
    _ENTMAX_COMPILED_KERNEL_FAILURES.add(count)
    _ENTMAX_COMPILED_KERNEL_CACHE.pop(count, None)

def _entmax_compiled_kernel(iterations: int) -> Any | None:
    """Return one lazily compiled CUDA kernel per static bisection count."""

    count = max(1, int(iterations))
    state = _entmax_runtime_state(count)
    state["compile_requested"] = True
    if count in _ENTMAX_TEMPORARY_EAGER_COUNTS:
        state["compiled_active"] = False
        return None
    if count in _ENTMAX_COMPILED_KERNEL_FAILURES:
        return None
    cached = _ENTMAX_COMPILED_KERNEL_CACHE.get(count)
    if cached is not None:
        return cached
    compile_function = getattr(torch, "compile", None)
    state["compile_attempted"] = True
    if not callable(compile_function):
        _disable_compiled_entmax(count, reason="torch_compile_unavailable")
        return None

    def kernel(values: torch.Tensor, alpha_tensor: torch.Tensor) -> torch.Tensor:
        return _entmax_generic_bisection_eager(values, alpha_tensor, iterations=count)

    try:
        compiled = compile_function(kernel, fullgraph=True, dynamic=False)
    except Exception as exc:
        _disable_compiled_entmax(count, reason=f"compile_exception:{type(exc).__name__}")
        return None
    state["compile_wrapper_created"] = True
    _ENTMAX_COMPILED_KERNEL_CACHE[count] = compiled
    return compiled

def _entmax_runtime_snapshot(iterations: int) -> dict[str, Any]:
    return dict(_entmax_runtime_state(iterations))

def _checked_compiled_entmax_or_eager(
    compiled_probability: torch.Tensor,
    eager_probability: torch.Tensor,
    *,
    iterations: int,
    validation_key: tuple[Any, ...],
    support_tolerance: float,
    compiled_eager_tolerance: float,
) -> torch.Tensor:
    """Validate one compiled result and return eager on any mismatch.

    Validation failure is intentionally non-fatal: it is recorded, compiled
    execution is disabled for the remaining run, and the eager reference result
    becomes the returned probability immediately.
    """

    count = max(1, int(iterations))
    state = _entmax_runtime_state(count)
    if validation_key in _ENTMAX_COMPILED_VALIDATION_KEYS:
        state["compiled_active"] = count not in _ENTMAX_COMPILED_KERNEL_FAILURES
        return compiled_probability if state["compiled_active"] else eager_probability
    difference = (compiled_probability.float() - eager_probability.float()).abs()
    max_abs_error = float(difference.amax().detach().cpu())
    support_threshold = float(max(0.0, support_tolerance))
    support_mismatch_count = int(
        ((compiled_probability > support_threshold) != (eager_probability > support_threshold))
        .sum()
        .detach()
        .cpu()
    )
    finite = bool(torch.isfinite(compiled_probability).all().detach().cpu())
    state["validation_count"] = int(state.get("validation_count", 0)) + 1
    state["compiled_eager_max_abs_error"] = max(
        float(state.get("compiled_eager_max_abs_error", 0.0)), max_abs_error
    )
    state["compiled_eager_support_mismatch_count"] = int(
        state.get("compiled_eager_support_mismatch_count", 0)
    ) + support_mismatch_count
    _ENTMAX_COMPILED_VALIDATION_KEYS.add(validation_key)
    if not finite or max_abs_error > float(compiled_eager_tolerance) or support_mismatch_count > 0:
        reason = (
            "validation_nonfinite" if not finite else
            "validation_support_mismatch" if support_mismatch_count > 0 else
            "validation_probability_error"
        )
        _disable_compiled_entmax(count, reason=reason)
        return eager_probability
    state["compile_succeeded"] = True
    state["compiled_active"] = True
    return compiled_probability

def _entmax_alpha_last_dim(
    logits: torch.Tensor,
    alpha: float,
    *,
    iterations: int = 32,
    compile_generic_cuda: bool = False,
    support_tolerance: float = 0.0,
    compiled_eager_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Entmax-alpha over the final dimension with checked compiled execution."""

    values = logits.float()
    if int(values.numel()) <= 0 or int(values.shape[-1]) <= 0:
        return values
    alpha_value = float(alpha)
    if alpha_value <= 1.0 + 1e-8:
        return torch.softmax(values, dim=-1)
    if alpha_value >= 2.0 - 1e-8:
        probability = _sparsemax_last_dim(values)
        return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    alpha_tensor = values.new_tensor(alpha_value)
    count = max(1, int(iterations))
    state = _entmax_runtime_state(count)
    state["compile_requested"] = bool(compile_generic_cuda)
    if bool(compile_generic_cuda) and values.is_cuda and count not in _ENTMAX_COMPILED_KERNEL_FAILURES:
        compiled = _entmax_compiled_kernel(count)
        if compiled is not None:
            try:
                compiled_probability = compiled(values, alpha_tensor)
            except Exception as exc:
                _disable_compiled_entmax(count, reason=f"runtime_exception:{type(exc).__name__}")
            else:
                validation_key = (
                    count,
                    str(values.device),
                    str(values.dtype),
                    tuple(int(dim) for dim in values.shape),
                    int(values.numel()),
                    round(alpha_value, 8),
                    round(float(support_tolerance), 12),
                    round(float(compiled_eager_tolerance), 12),
                    bool(values.is_contiguous()),
                )
                if validation_key in _ENTMAX_COMPILED_VALIDATION_KEYS:
                    state["compiled_active"] = True
                    return compiled_probability
                eager_probability = _entmax_generic_bisection_eager(
                    values, alpha_tensor, iterations=count
                )
                return _checked_compiled_entmax_or_eager(
                    compiled_probability,
                    eager_probability,
                    iterations=count,
                    validation_key=validation_key,
                    support_tolerance=support_tolerance,
                    compiled_eager_tolerance=compiled_eager_tolerance,
                )
    state["compiled_active"] = False
    return _entmax_generic_bisection_eager(values, alpha_tensor, iterations=count)

def _entmax_alpha_1d(
    logits: torch.Tensor,
    alpha: float,
    *,
    iterations: int = 32,
    compile_generic_cuda: bool = False,
    support_tolerance: float = 0.0,
    compiled_eager_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Single-vector wrapper around final-dimension Entmax."""

    return _entmax_alpha_last_dim(
        logits.float().reshape(-1),
        alpha,
        iterations=iterations,
        compile_generic_cuda=compile_generic_cuda,
        support_tolerance=support_tolerance,
        compiled_eager_tolerance=compiled_eager_tolerance,
    )

def _average_tie_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return zero-based average ranks with deterministic tie handling on CPU."""

    data = values.detach().float().flatten().cpu()
    count = int(data.numel())
    if count <= 0:
        return data
    sorted_values, order = torch.sort(data, stable=True)
    new_group = torch.ones(count, dtype=torch.bool)
    if count > 1:
        new_group[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = new_group.cumsum(dim=0) - 1
    group_count = int(group_ids[-1].item()) + 1
    positions = torch.arange(count, dtype=torch.float32)
    sums = torch.zeros(group_count, dtype=torch.float32)
    counts = torch.zeros(group_count, dtype=torch.float32)
    sums.scatter_add_(0, group_ids, positions)
    counts.scatter_add_(0, group_ids, torch.ones_like(positions))
    group_ranks = sums / counts.clamp_min(1.0)
    ranks = torch.empty(count, dtype=torch.float32)
    ranks[order] = group_ranks[group_ids]
    return ranks

def _spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_values = left.detach().float().flatten().cpu()
    right_values = right.detach().float().flatten().cpu()
    if int(left_values.numel()) != int(right_values.numel()) or int(left_values.numel()) < 2:
        return None
    left_rank = _average_tie_ranks(left_values)
    right_rank = _average_tie_ranks(right_values)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom) <= 1e-12:
        return None
    return float(torch.dot(left_centered, right_centered) / denom.clamp_min(1e-12))

def _rank_correlation_for_abs_values(
    left_abs: torch.Tensor,
    right_abs: torch.Tensor,
    mask: torch.Tensor,
) -> float | str:
    mask_cpu = mask.detach().bool().flatten().cpu()
    if int(mask_cpu.sum().item()) < 2:
        return ""
    left = left_abs.detach().float().flatten().cpu()[mask_cpu]
    right = right_abs.detach().float().flatten().cpu()[mask_cpu]
    correlation = _spearman_correlation(left, right)
    return "" if correlation is None else float(correlation)
