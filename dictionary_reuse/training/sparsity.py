"""Current DiR routed-support metrics and support bookkeeping."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .schema import _min_mean_max_metrics
from ..model.routing import _scalar_report_float
from ..model.dictionary_operator import (
    iter_dictionary_layers,
)


# --- Natural-sparsity policy -------------------------------------------------
def _natural_sparsity_uses_forward_solution_entropy(config: dict[str, Any] | None) -> bool:
    payload = config or {}
    objective = str(payload.get("objective", payload.get("code_unit", ""))).lower()
    return bool(payload.get("use_forward_solution_entropy", False)) or "forward_solution_entropy" in objective

def _natural_sparsity_applies_to_phase(
    config: dict[str, Any] | None,
    active_groups: Sequence[str] | None,
) -> bool:
    payload = config or {}
    phase_spec = payload.get("apply_only_in_phase", payload.get("phase", ""))
    if not phase_spec:
        return True
    if active_groups is None:
        return True
    allowed = {str(phase_spec)} if isinstance(phase_spec, str) else {str(item) for item in phase_spec}
    active = {str(item) for item in active_groups}
    return bool(active & allowed)

def _natural_sparsity_records_activation_contribution(config: dict[str, Any] | None) -> bool:
    payload = config or {}
    # current DiR uses the same bounded report pass for atom-level relative-C shortcut
    # diagnostics and actual QK/VO activation RMS, even when no contribution
    # sparsity loss is active. This remains measurement-only.
    return bool(payload.get("activation_aware_eval_metrics_enabled", False))

# --- Forward-solution entropy and routed support ----------------------------
def _empty_forward_solution_entropy_metrics() -> dict[str, float]:
    return {
        "forward_solution_entropy_mean": 0.0,
        "forward_solution_entropy_p95": 0.0,
        "forward_solution_neff_mean": 0.0,
        "forward_solution_neff_p95": 0.0,
        "forward_solution_mass95_atoms_mean": 0.0,
        "forward_solution_mass95_atoms_p95": 0.0,
        "forward_solution_loss_mass_entropy_mean": 0.0,
        "forward_solution_loss_mass_entropy_p95": 0.0,
        "forward_solution_loss_mass_neff_mean": 0.0,
        "forward_solution_loss_mass_neff_p95": 0.0,
        "forward_solution_loss_mass_mass95_atoms_mean": 0.0,
        "forward_solution_loss_mass_mass95_atoms_p95": 0.0,
        "forward_solution_used_atom_entropy": 0.0,
        "global_solution_entropy_mode": 0.0,
        "global_solution_entmax_alpha": 1.0,
        "global_solution_entropy_mean": 0.0,
        "global_solution_neff_atoms_mean": 0.0,
        "global_solution_support_active_atoms_mean": 0.0,
        "global_solution_support_inactive_atoms_mean": 0.0,
        "global_solution_hard_support_enabled": 0.0,
        "forward_routed_gate_enabled_mean": 0.0,
        "forward_routed_gate_alpha_mean": 1.0,
        "forward_routed_gate_active_atoms_mean": 0.0,
        "forward_routed_gate_active_ratio_mean": 0.0,
        "forward_routed_fixed_support_available": 0.0,
        "forward_routed_fixed_support_active_count_min": 0.0,
        "forward_routed_fixed_support_active_count_max": 0.0,
        "forward_routed_fixed_support_active_count_mean": 0.0,
        "forward_routed_fixed_support_active_count_per_layer": "",
        "forward_routed_gate_straight_through_mean": 0.0,
        "forward_routed_gate_ema_support_mean": 0.0,
        "forward_routed_gate_route_mass_skipped_mean": 0.0,
        "forward_routed_gate_support_jaccard_mean": 0.0,
        "forward_routed_gate_support_union_atoms_mean": 0.0,
        "forward_routed_gate_support_churn_mean": 0.0,
        "forward_routed_gate_pre_route_neff_mean": 0.0,
        "forward_routed_gate_pre_route_mass95_atoms_mean": 0.0,
        "forward_routed_gate_post_route_neff_mean": 0.0,
        "forward_routed_gate_post_route_mass95_atoms_mean": 0.0,
    }

def _forward_solution_entropy_mode(config: dict[str, Any] | None) -> str:
    payload = config or {}
    return str(payload.get("solution_entropy_mode", payload.get("forward_solution_entropy_mode", "local"))).lower()

def _entmax_alpha_schedule_value(
    payload: dict[str, Any],
    *,
    schedule: Any,
    epoch: int | None,
) -> float | None:
    if isinstance(schedule, dict):
        schedule = schedule.get("schedule", schedule.get("points", schedule.get("entmax_alpha_schedule")))
    if not isinstance(schedule, list) or not schedule:
        return None
    points: list[tuple[int, float]] = []
    for item in schedule:
        if not isinstance(item, dict):
            continue
        if "epoch" not in item:
            continue
        value = item.get("alpha", item.get("value"))
        if value is None:
            continue
        points.append((int(item["epoch"]), float(value)))
    if not points:
        return None
    points.sort(key=lambda pair: pair[0])
    if epoch is None:
        return float(points[-1][1])
    epoch_value = int(epoch)
    if epoch_value <= points[0][0]:
        return float(points[0][1])
    schedule_mode = str(
        payload.get("entmax_alpha_schedule_mode", payload.get("alpha_schedule_mode", "linear"))
    ).strip().lower()
    cycle_epochs = max(1, int(payload.get("entmax_alpha_cycle_epochs", payload.get("alpha_schedule_cycle_epochs", 1)) or 1))
    for (left_epoch, left_alpha), (right_epoch, right_alpha) in zip(points, points[1:]):
        if epoch_value <= right_epoch:
            if int(right_epoch) <= int(left_epoch):
                return float(right_alpha)
            if schedule_mode in {"cycle_staircase", "cycle_step_linear", "cycle_quantized_linear"}:
                step_count = max(1, int(right_epoch - left_epoch) // int(cycle_epochs))
                step_index = max(0, min(step_count, int(epoch_value - left_epoch) // int(cycle_epochs)))
                progress = float(step_index) / float(step_count)
            elif schedule_mode in {"step", "staircase", "piecewise_constant"}:
                progress = 0.0
            else:
                progress = (float(epoch_value) - float(left_epoch)) / (float(right_epoch) - float(left_epoch))
            return float(left_alpha) + (float(right_alpha) - float(left_alpha)) * max(0.0, min(1.0, progress))
    return float(points[-1][1])

def _forward_solution_kindwise_alpha_schedules(config: dict[str, Any] | None) -> dict[str, Any]:
    payload = config or {}
    schedules = payload.get("entmax_alpha_schedule_by_layer_kind", payload.get("alpha_schedule_by_layer_kind", {}))
    return schedules if isinstance(schedules, dict) else {}

def _forward_solution_has_kindwise_alpha(config: dict[str, Any] | None) -> bool:
    return bool(_forward_solution_kindwise_alpha_schedules(config))

def _dictionary_layer_kind_alpha_key(layer_name: str) -> str:
    name = str(layer_name)
    if name == "classification_head" or name.startswith("classification_head."):
        return "head"
    if name.startswith("patch_embedding."):
        return "patch"
    if name.startswith("class_token.") or name.startswith("position_embedding."):
        return "token_position"
    if ".multi_head_self_attention." in name:
        if any(token in name for token in ("query_projection", "key_projection", "value_projection")):
            return "qkv"
        if "output_projection" in name:
            return "o"
    if ".feed_forward_network." in name:
        if "first_linear_layer" in name:
            return "w1"
        if "second_linear_layer" in name:
            return "w2"
    return "default"

def _alpha_schedule_aliases_for_layer_kind(kind: str) -> tuple[str, ...]:
    normalized = str(kind).strip().lower()
    aliases = {
        "head": ("head", "classification_head"),
        "qkv": ("qkv", "q_k_v", "q/k/v", "attention_qkv", "attention_q_k_v", "reader", "q", "k", "v"),
        "o": ("o", "output", "attention_o", "attention_output"),
        "w1": ("w1", "mlp_w1", "ffn_w1", "first_linear", "first_linear_layer"),
        "w2": ("w2", "mlp_w2", "ffn_w2", "second_linear", "second_linear_layer"),
        "patch": ("patch", "patch_embedding", "patch_token_position", "patch_token_pos", "endpoint", "endpoints"),
        "token_position": (
            "token_position",
            "token",
            "class_token",
            "position",
            "position_embedding",
            "patch_token_position",
            "patch_token_pos",
            "patch_token_positional",
            "patch/token/pos",
            "endpoint",
            "endpoints",
        ),
        "patch_token_position": (
            "patch_token_position",
            "patch_token_pos",
            "patch_token_positional",
            "patch/token/pos",
            "endpoint",
            "endpoints",
            "patch",
            "token_position",
            "token",
            "class_token",
            "position",
            "position_embedding",
        ),
        "default": ("default", "all"),
    }
    return aliases.get(normalized, (normalized, "default", "all"))

def _forward_solution_entmax_alpha(config: dict[str, Any] | None, *, epoch: int | None) -> float:
    payload = config or {}
    if _forward_solution_entropy_mode(payload) not in {"global_entmax", "global_sparsemax", "a1o_global_entmax"}:
        return 1.0
    schedule = payload.get("entmax_alpha_schedule", payload.get("alpha_schedule"))
    alpha_from_schedule = _entmax_alpha_schedule_value(payload, schedule=schedule, epoch=epoch)
    if alpha_from_schedule is not None:
        return float(alpha_from_schedule)
    kind_schedules = _forward_solution_kindwise_alpha_schedules(payload)
    if kind_schedules:
        values: list[float] = []
        for value in kind_schedules.values():
            alpha = _entmax_alpha_schedule_value(payload, schedule=value, epoch=epoch)
            if alpha is not None:
                values.append(float(alpha))
        if values:
            return sum(values) / float(len(values))
    if epoch is None:
        return float(payload.get("entmax_alpha", payload.get("alpha", 1.0)))
    start_epoch = int(payload.get("entmax_schedule_start_epoch", payload.get("schedule_start_epoch", 1)) or 1)
    warmup_epochs = max(0, int(payload.get("entmax_warmup_epochs", payload.get("warmup_epochs", 0)) or 0))
    transition_epochs = max(1, int(payload.get("entmax_transition_epochs", payload.get("transition_epochs", 1)) or 1))
    alpha_start = float(payload.get("entmax_alpha_start", 1.0))
    alpha_end = float(payload.get("entmax_alpha_end", 2.0))
    if int(epoch) < start_epoch + warmup_epochs:
        return alpha_start
    progress = min(1.0, max(0.0, float(int(epoch) - (start_epoch + warmup_epochs)) / float(transition_epochs)))
    return alpha_start + (alpha_end - alpha_start) * progress

def _forward_solution_entmax_alpha_for_layer(
    config: dict[str, Any] | None,
    *,
    epoch: int | None,
    layer_name: str,
) -> float:
    payload = config or {}
    if _forward_solution_entropy_mode(payload) not in {"global_entmax", "global_sparsemax", "a1o_global_entmax"}:
        return 1.0
    kind_schedules = _forward_solution_kindwise_alpha_schedules(payload)
    if kind_schedules:
        kind = _dictionary_layer_kind_alpha_key(layer_name)
        for alias in _alpha_schedule_aliases_for_layer_kind(kind):
            if alias in kind_schedules:
                alpha = _entmax_alpha_schedule_value(payload, schedule=kind_schedules[alias], epoch=epoch)
                if alpha is not None:
                    return float(alpha)
        for alias in _alpha_schedule_aliases_for_layer_kind("default"):
            if alias in kind_schedules:
                alpha = _entmax_alpha_schedule_value(payload, schedule=kind_schedules[alias], epoch=epoch)
                if alpha is not None:
                    return float(alpha)
    return _forward_solution_entmax_alpha(payload, epoch=epoch)

def _epoch_flag_enabled(
    payload: dict[str, Any],
    *,
    key: str,
    epoch: int | None,
    default: bool = False,
) -> bool:
    if key in payload:
        return bool(payload[key])
    start_key = f"{key}_start_epoch"
    if start_key in payload and epoch is not None:
        return int(epoch) >= int(payload[start_key])
    return bool(default)

def _epoch_start_value(payload: dict[str, Any], *keys: str, default: int = 10**9) -> int:
    """Return an epoch start value while preserving valid epoch zero.

    Fixed-support profiles may intentionally start routing at epoch 0. Avoid
    ``value or default`` because Python treats 0 as false and would delay the
    configured route until an unreachable epoch.
    """

    for key in keys:
        if key in payload:
            value = payload.get(key)
            return int(default) if value is None else int(value)
    return int(default)

def prepare_forward_solution_entropy_layers(
    model: nn.Module,
    config: dict[str, Any] | None,
    *,
    active_groups: Sequence[str] | None = None,
    epoch: int | None = None,
    force_record_metrics: bool = False,
    record_metrics: bool = True,
) -> None:
    payload = config or {}
    phase_applies = _natural_sparsity_applies_to_phase(payload, active_groups)
    mode = _forward_solution_entropy_mode(payload)
    routed_start_epoch = _epoch_start_value(payload, "forward_routed_gate_start_epoch", "routed_gate_start_epoch", default=10**9)
    routed_enabled = bool(payload.get("forward_routed_gate_enabled", payload.get("routed_forward_gate_enabled", False))) and (
        epoch is None or int(epoch) >= routed_start_epoch
    )
    routed_straight_through = bool(payload.get("forward_routed_gate_straight_through", payload.get("straight_through_hard_route", True)))
    routed_support_tolerance = float(payload.get("forward_routed_gate_support_tolerance", payload.get("support_tolerance", 1e-8)) or 1e-8)
    routed_score_eps = float(payload.get("forward_routed_gate_score_eps", payload.get("solution_entropy_mass_eps", 1e-6)) or 1e-6)
    routed_eval_use_ema = _epoch_flag_enabled(
        payload,
        key="forward_routed_gate_eval_use_ema_support",
        epoch=epoch,
        default=bool(payload.get("eval_use_ema_support", False)),
    )
    routed_train_use_ema = _epoch_flag_enabled(
        payload,
        key="forward_routed_gate_train_use_ema_support",
        epoch=epoch,
        default=bool(payload.get("train_use_ema_support", False)),
    )
    routed_require_ema = _epoch_flag_enabled(
        payload,
        key="forward_routed_gate_require_ema_support",
        epoch=epoch,
        default=bool(payload.get("require_ema_support", False)),
    )
    routed_skip_route_mass_when_fixed = bool(payload.get("forward_routed_gate_skip_route_mass_when_using_fixed_support", False))
    routed_use_copied_hard_support_mask = bool(payload.get("forward_routed_gate_use_copied_hard_support_mask", True))
    if _forward_routed_fixed_support_rehearsal_started(payload, epoch):
        routed_enabled = True
        routed_straight_through = False
        routed_eval_use_ema = True
        routed_train_use_ema = True
        routed_require_ema = True
        routed_skip_route_mass_when_fixed = True
        routed_use_copied_hard_support_mask = True
    routed_entmax_iterations = max(1, int(payload.get("forward_routed_gate_entmax_iterations", 16) or 16))
    routed_compile_entmax = bool(payload.get("forward_routed_gate_compile_entmax_bisection", True))
    routed_entmax_compile_parity_tolerance = max(
        1e-9, float(payload.get("forward_routed_gate_entmax_compile_parity_tolerance", 1e-6))
    )
    routed_route_mass_parity = bool(payload.get("forward_routed_gate_route_mass_parity_check", True))
    routed_route_mass_parity_tolerance = max(1e-9, float(payload.get("forward_routed_gate_route_mass_parity_tolerance", 1e-6)))
    fixed_support_cache_output_parity = bool(payload.get("fixed_support_cache_output_parity_check", True))
    fixed_support_cache_output_parity_tolerance = max(1e-9, float(payload.get("fixed_support_cache_output_parity_tolerance", 1e-6)))
    entropy_forward_needed = bool(phase_applies and (bool(force_record_metrics) or bool(routed_enabled and record_metrics)))
    enabled = (
        bool(payload.get("enabled", False))
        and _natural_sparsity_uses_forward_solution_entropy(payload)
        and entropy_forward_needed
    )
    mass_eps = float(payload.get("mass_eps", payload.get("solution_entropy_mass_eps", payload.get("smooth_abs_eps", 1e-6))) or 1e-6)
    entropy_eps = float(payload.get("entropy_eps", 1e-8) or 1e-8)
    support_tolerance = float(payload.get("support_tolerance", payload.get("sparsemax_support_tolerance", 1e-12)) or 1e-12)
    usage_ema_decay = float(payload.get("usage_ema_decay", 0.95) or 0.95)
    use_ema_support = bool(payload.get("use_ema_support", True))
    for _name, layer in iter_dictionary_layers(model):
        layer_alpha = _forward_solution_entmax_alpha_for_layer(
            payload,
            epoch=int(epoch) if epoch is not None else None,
            layer_name=_name,
        )
        layer.forward_solution_entropy_enabled = bool(enabled)
        layer.forward_solution_entropy_mode = mode
        layer.forward_solution_entmax_alpha = float(layer_alpha)
        layer.forward_solution_support_tolerance = float(support_tolerance)
        layer.forward_solution_usage_ema_decay = float(usage_ema_decay)
        layer.forward_solution_use_ema_support = bool(use_ema_support)
        layer.forward_solution_entropy_mass_eps = mass_eps
        layer.forward_solution_entropy_eps = entropy_eps
        layer.forward_solution_entropy_record_metrics = bool(record_metrics)
        layer.forward_solution_entropy_mass_mode = str(payload.get("solution_entropy_mass", payload.get("forward_solution_entropy_mass", ""))).lower()
        layer.forward_routed_gate_enabled = bool(routed_enabled)
        layer.forward_routed_gate_straight_through = bool(routed_straight_through)
        layer.forward_routed_gate_alpha = float(layer_alpha)
        layer.forward_routed_gate_support_tolerance = float(routed_support_tolerance)
        layer.forward_routed_gate_score_eps = float(routed_score_eps)
        layer.forward_routed_gate_eval_use_ema_support = bool(routed_eval_use_ema)
        layer.forward_routed_gate_train_use_ema_support = bool(routed_train_use_ema)
        layer.forward_routed_gate_require_ema_support = bool(routed_require_ema)
        layer.forward_routed_gate_skip_route_mass_when_using_fixed_support = bool(routed_skip_route_mass_when_fixed)
        layer.forward_routed_gate_use_copied_hard_support_mask = bool(routed_use_copied_hard_support_mask)
        layer.forward_routed_gate_entmax_iterations = int(routed_entmax_iterations)
        layer.forward_routed_gate_compile_entmax_bisection = bool(routed_compile_entmax)
        layer.forward_routed_gate_entmax_compile_parity_tolerance = float(
            routed_entmax_compile_parity_tolerance
        )
        layer.forward_routed_gate_route_mass_parity_check = bool(routed_route_mass_parity)
        layer.forward_routed_gate_route_mass_parity_tolerance = float(routed_route_mass_parity_tolerance)
        layer.fixed_support_cache_output_parity_check = bool(fixed_support_cache_output_parity)
        layer.fixed_support_cache_output_parity_tolerance = float(fixed_support_cache_output_parity_tolerance)
        support_union_start_epoch = payload.get("forward_routed_support_union_start_epoch", payload.get("support_union_start_epoch"))
        if (
            routed_enabled
            and support_union_start_epoch is not None
            and epoch is not None
            and int(epoch) == int(support_union_start_epoch)
            and getattr(layer, "_forward_routed_support_union_reset_epoch", None) != int(support_union_start_epoch)
            and hasattr(layer, "reset_forward_routed_support_diagnostics")
        ):
            layer.reset_forward_routed_support_diagnostics(reset_epoch=int(support_union_start_epoch))

def _forward_routed_fixed_support_rehearsal_started(config: dict[str, Any] | None, epoch: int | None) -> bool:
    if epoch is None:
        return False
    payload = config or {}
    start = payload.get("forward_routed_fixed_support_rehearsal_start_epoch", payload.get("fixed_support_rehearsal_start_epoch"))
    return start is not None and int(epoch) >= int(start)

def refresh_forward_solution_entropy_eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    config: dict[str, Any] | None,
    epoch: int,
    max_batches: int | None = None,
) -> None:
    """Refresh forward-solution route metrics without building a training-loss graph.

    Reports actual entropy, N_eff, mass95, and used-atom entropy metrics at
    evaluation checkpoints.
    """

    payload = config or {}
    if not (bool(payload.get("enabled", False)) and _natural_sparsity_uses_forward_solution_entropy(payload)):
        return
    layers = [layer for _name, layer in iter_dictionary_layers(model)]
    if not layers:
        return
    snapshot_attrs = (
        "forward_solution_entropy_enabled",
        "forward_solution_entropy_mode",
        "forward_solution_entmax_alpha",
        "forward_solution_support_tolerance",
        "forward_solution_usage_ema_decay",
        "forward_solution_use_ema_support",
        "forward_solution_entropy_mass_eps",
        "forward_solution_entropy_eps",
        "forward_solution_entropy_record_metrics",
        "forward_solution_entropy_mass_mode",
        "forward_routed_gate_enabled",
        "forward_routed_gate_straight_through",
        "forward_routed_gate_alpha",
        "forward_routed_gate_support_tolerance",
        "forward_routed_gate_score_eps",
        "forward_routed_gate_eval_use_ema_support",
    )
    previous = [
        (layer, {name: getattr(layer, name) for name in snapshot_attrs if hasattr(layer, name)})
        for layer in layers
    ]
    fixed_mask_metrics_only = str(payload.get("solution_entropy_mass", payload.get("forward_solution_entropy_mass", ""))).lower() in {
        "fixed_source_hard_support_mask_metrics_only",
        "fixed_hard_support_mask_metrics_only",
        "copied_hard_support_mask_metrics_only",
    }
    was_training = bool(model.training)
    if fixed_mask_metrics_only and all(
        hasattr(layer, "forward_routed_fixed_support_mask")
        and bool(getattr(layer, "_forward_routed_fixed_support_is_initialized", lambda: False)())
        for layer in layers
    ):
        try:
            prepare_forward_solution_entropy_layers(
                model,
                payload,
                active_groups=None,
                epoch=int(epoch),
                force_record_metrics=True,
                record_metrics=True,
            )
            for layer in layers:
                layer._record_forward_solution_fixed_support_metrics_from_mask(layer.forward_routed_fixed_support_mask)
        finally:
            for layer, state in previous:
                for name, value in state.items():
                    setattr(layer, name, value)
            if was_training:
                model.train()
            else:
                model.eval()
        return
    try:
        model.eval()
        prepare_forward_solution_entropy_layers(
            model,
            payload,
            active_groups=None,
            epoch=int(epoch),
            force_record_metrics=True,
            record_metrics=True,
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                images = batch[0].to(device)
                _ = model(images)
    finally:
        for layer, state in previous:
            for name, value in state.items():
                setattr(layer, name, value)
        if was_training:
            model.train()
        else:
            model.eval()

def forward_solution_entropy_metrics(
    model: nn.Module,
    config: dict[str, Any] | None,
    *,
    epoch: int,
) -> dict[str, float]:
    payload = config or {}
    layers = [layer for _name, layer in iter_dictionary_layers(model)]
    if not bool(payload.get("enabled", False)) or not layers:
        return _empty_forward_solution_entropy_metrics()
    values: dict[str, list[float]] = {
        "forward_solution_entropy_mean": [],
        "forward_solution_entropy_p95": [],
        "forward_solution_neff_mean": [],
        "forward_solution_neff_p95": [],
        "forward_solution_mass95_atoms_mean": [],
        "forward_solution_mass95_atoms_p95": [],
        "forward_solution_loss_mass_entropy_mean": [],
        "forward_solution_loss_mass_entropy_p95": [],
        "forward_solution_loss_mass_neff_mean": [],
        "forward_solution_loss_mass_neff_p95": [],
        "forward_solution_loss_mass_mass95_atoms_mean": [],
        "forward_solution_loss_mass_mass95_atoms_p95": [],
        "forward_solution_used_atom_entropy": [],
        "global_solution_entropy_mean": [],
        "global_solution_neff_atoms_mean": [],
        "global_solution_support_active_atoms_mean": [],
        "global_solution_support_inactive_atoms_mean": [],
        "global_solution_hard_support_enabled": [],
        "forward_routed_gate_enabled_mean": [],
        "forward_routed_gate_alpha_mean": [],
        "forward_routed_gate_active_atoms_mean": [],
        "forward_routed_gate_active_ratio_mean": [],
        "forward_routed_fixed_support_available": [],
        "forward_routed_fixed_support_active_count_min": [],
        "forward_routed_fixed_support_active_count_max": [],
        "forward_routed_fixed_support_active_count_mean": [],
        "forward_routed_gate_straight_through_mean": [],
        "forward_routed_gate_ema_support_mean": [],
        "forward_routed_gate_route_mass_skipped_mean": [],
        "forward_routed_gate_route_mass_parity_max_abs_error_mean": [],
        "forward_routed_gate_route_mass_parity_max_relative_error_mean": [],
        "fixed_support_cache_output_parity_max_abs_error_mean": [],
        "fixed_support_cache_output_parity_mean_abs_error_mean": [],
        "fixed_support_cache_output_parity_max_relative_error_mean": [],
        "fixed_support_cache_fallback_count_max": [],
        "fixed_support_cache_disabled_mean": [],
        "forward_routed_entmax_compile_requested_mean": [],
        "forward_routed_entmax_compile_attempted_mean": [],
        "forward_routed_entmax_compile_succeeded_mean": [],
        "forward_routed_entmax_compiled_active_mean": [],
        "forward_routed_entmax_compile_validation_count_max": [],
        "forward_routed_entmax_compile_fallback_count_max": [],
        "forward_routed_entmax_compiled_eager_max_abs_error_max": [],
        "forward_routed_entmax_compiled_eager_support_mismatch_count_max": [],
        "forward_routed_gate_support_jaccard_mean": [],
        "forward_routed_gate_support_union_atoms_mean": [],
        "forward_routed_gate_support_churn_mean": [],
        "forward_routed_gate_pre_route_neff_mean": [],
        "forward_routed_gate_pre_route_mass95_atoms_mean": [],
        "forward_routed_gate_post_route_neff_mean": [],
        "forward_routed_gate_post_route_mass95_atoms_mean": [],
    }
    for layer in layers:
        values["forward_solution_entropy_mean"].append(float(getattr(layer, "_last_forward_solution_entropy_mean", 0.0)))
        values["forward_solution_entropy_p95"].append(float(getattr(layer, "_last_forward_solution_entropy_p95", 0.0)))
        values["forward_solution_neff_mean"].append(float(getattr(layer, "_last_forward_solution_neff_mean", 0.0)))
        values["forward_solution_neff_p95"].append(float(getattr(layer, "_last_forward_solution_neff_p95", 0.0)))
        values["forward_solution_mass95_atoms_mean"].append(float(getattr(layer, "_last_forward_solution_mass95_atoms_mean", 0.0)))
        values["forward_solution_mass95_atoms_p95"].append(float(getattr(layer, "_last_forward_solution_mass95_atoms_p95", 0.0)))
        values["forward_solution_loss_mass_entropy_mean"].append(float(getattr(layer, "_last_forward_solution_loss_mass_entropy_mean", 0.0)))
        values["forward_solution_loss_mass_entropy_p95"].append(float(getattr(layer, "_last_forward_solution_loss_mass_entropy_p95", 0.0)))
        values["forward_solution_loss_mass_neff_mean"].append(float(getattr(layer, "_last_forward_solution_loss_mass_neff_mean", 0.0)))
        values["forward_solution_loss_mass_neff_p95"].append(float(getattr(layer, "_last_forward_solution_loss_mass_neff_p95", 0.0)))
        values["forward_solution_loss_mass_mass95_atoms_mean"].append(float(getattr(layer, "_last_forward_solution_loss_mass_mass95_atoms_mean", 0.0)))
        values["forward_solution_loss_mass_mass95_atoms_p95"].append(float(getattr(layer, "_last_forward_solution_loss_mass_mass95_atoms_p95", 0.0)))
        values["forward_solution_used_atom_entropy"].append(float(getattr(layer, "_last_forward_solution_used_atom_entropy", 0.0)))
        values["global_solution_entropy_mean"].append(float(getattr(layer, "_last_global_solution_entropy", 0.0)))
        values["global_solution_neff_atoms_mean"].append(float(getattr(layer, "_last_global_solution_neff_atoms", 0.0)))
        values["global_solution_support_active_atoms_mean"].append(float(getattr(layer, "_last_global_solution_support_active_atoms", 0.0)))
        values["global_solution_support_inactive_atoms_mean"].append(float(getattr(layer, "_last_global_solution_support_inactive_atoms", 0.0)))
        values["global_solution_hard_support_enabled"].append(1.0 if bool(getattr(layer, "_last_global_solution_hard_support_enabled", False)) else 0.0)
        values["forward_routed_gate_enabled_mean"].append(1.0 if bool(getattr(layer, "_last_forward_routed_gate_enabled", False)) else 0.0)
        values["forward_routed_gate_alpha_mean"].append(float(getattr(layer, "_last_forward_routed_gate_alpha", 1.0)))
        values["forward_routed_gate_active_atoms_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_gate_active_atoms", 0.0)))
        values["forward_routed_gate_active_ratio_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_gate_active_ratio", 0.0), default=0.0))
        fixed_mask = getattr(layer, "forward_routed_fixed_support_mask", None)
        fixed_available = bool(getattr(layer, "_forward_routed_fixed_support_is_initialized", lambda: False)()) and isinstance(fixed_mask, torch.Tensor)
        fixed_count = float(fixed_mask.detach().float().sum().cpu().item()) if fixed_available else 0.0
        values["forward_routed_fixed_support_available"].append(1.0 if fixed_available else 0.0)
        values["forward_routed_fixed_support_active_count_min"].append(fixed_count)
        values["forward_routed_fixed_support_active_count_max"].append(fixed_count)
        values["forward_routed_fixed_support_active_count_mean"].append(fixed_count)
        values["forward_routed_gate_straight_through_mean"].append(1.0 if bool(getattr(layer, "_last_forward_routed_gate_straight_through", False)) else 0.0)
        values["forward_routed_gate_ema_support_mean"].append(1.0 if bool(getattr(layer, "_last_forward_routed_gate_ema_support", False)) else 0.0)
        values["forward_routed_gate_route_mass_skipped_mean"].append(1.0 if bool(getattr(layer, "_last_forward_routed_route_mass_skipped", False)) else 0.0)
        values["forward_routed_gate_route_mass_parity_max_abs_error_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_route_mass_parity_max_abs_error", None)))
        values["forward_routed_gate_route_mass_parity_max_relative_error_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_route_mass_parity_max_relative_error", None)))
        values["fixed_support_cache_output_parity_max_abs_error_mean"].append(_scalar_report_float(getattr(layer, "_last_fixed_support_cache_output_parity_max_abs_error", None)))
        values["fixed_support_cache_output_parity_mean_abs_error_mean"].append(_scalar_report_float(getattr(layer, "_last_fixed_support_cache_output_parity_mean_abs_error", None)))
        values["fixed_support_cache_output_parity_max_relative_error_mean"].append(_scalar_report_float(getattr(layer, "_last_fixed_support_cache_output_parity_max_relative_error", None)))
        values["fixed_support_cache_fallback_count_max"].append(float(getattr(layer, "_fixed_support_cache_fallback_count", 0)))
        values["fixed_support_cache_disabled_mean"].append(1.0 if bool(getattr(layer, "_fixed_support_cache_disabled", False)) else 0.0)
        values["forward_routed_entmax_compile_requested_mean"].append(1.0 if bool(getattr(layer, "_last_entmax_compile_requested", False)) else 0.0)
        values["forward_routed_entmax_compile_attempted_mean"].append(1.0 if bool(getattr(layer, "_last_entmax_compile_attempted", False)) else 0.0)
        values["forward_routed_entmax_compile_succeeded_mean"].append(1.0 if bool(getattr(layer, "_last_entmax_compile_succeeded", False)) else 0.0)
        values["forward_routed_entmax_compiled_active_mean"].append(1.0 if bool(getattr(layer, "_last_entmax_compiled_active", False)) else 0.0)
        values["forward_routed_entmax_compile_validation_count_max"].append(float(getattr(layer, "_last_entmax_compile_validation_count", 0)))
        values["forward_routed_entmax_compile_fallback_count_max"].append(float(getattr(layer, "_last_entmax_compile_fallback_count", 0)))
        values["forward_routed_entmax_compiled_eager_max_abs_error_max"].append(float(getattr(layer, "_last_entmax_compiled_eager_max_abs_error", 0.0)))
        values["forward_routed_entmax_compiled_eager_support_mismatch_count_max"].append(float(getattr(layer, "_last_entmax_compiled_eager_support_mismatch_count", 0)))
        support_union = getattr(layer, "forward_routed_support_union", None)
        support_union_atoms = _scalar_report_float(support_union.detach().float().sum() if isinstance(support_union, torch.Tensor) else 0.0)
        support_jaccard = _scalar_report_float(getattr(layer, "_last_forward_routed_gate_support_jaccard", None), default=1.0)
        values["forward_routed_gate_support_jaccard_mean"].append(support_jaccard)
        values["forward_routed_gate_support_union_atoms_mean"].append(support_union_atoms)
        values["forward_routed_gate_support_churn_mean"].append(1.0 - support_jaccard)
        values["forward_routed_gate_pre_route_neff_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_pre_route_neff", None)))
        values["forward_routed_gate_pre_route_mass95_atoms_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_pre_route_mass95_atoms", None)))
        values["forward_routed_gate_post_route_neff_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_post_route_neff", None)))
        values["forward_routed_gate_post_route_mass95_atoms_mean"].append(_scalar_report_float(getattr(layer, "_last_forward_routed_post_route_mass95_atoms", None)))
    mean_values = {key: (sum(items) / float(max(1, len(items)))) for key, items in values.items()}
    for key in (
        "forward_routed_entmax_compile_validation_count_max",
        "forward_routed_entmax_compile_fallback_count_max",
        "forward_routed_entmax_compiled_eager_max_abs_error_max",
        "forward_routed_entmax_compiled_eager_support_mismatch_count_max",
        "fixed_support_cache_fallback_count_max",
    ):
        mean_values[key] = max(values.get(key, [0.0]) or [0.0])
    fallback_reasons = sorted({
        str(getattr(layer, "_last_entmax_compile_fallback_reason", ""))
        for layer in layers
        if str(getattr(layer, "_last_entmax_compile_fallback_reason", ""))
    })
    mean_values["forward_routed_entmax_compile_fallback_reason"] = ";".join(fallback_reasons)
    cache_fallback_reasons = sorted({
        str(getattr(layer, "_fixed_support_cache_fallback_reason", ""))
        for layer in layers
        if str(getattr(layer, "_fixed_support_cache_fallback_reason", ""))
    })
    mean_values["fixed_support_cache_fallback_reason"] = ";".join(cache_fallback_reasons)
    fixed_counts_for_summary = list(values.get("forward_routed_fixed_support_active_count_mean", []))
    fixed_available_for_summary = list(values.get("forward_routed_fixed_support_available", []))
    if fixed_counts_for_summary:
        mean_values["forward_routed_fixed_support_available"] = 1.0 if all(value >= 1.0 for value in fixed_available_for_summary) else 0.0
        mean_values["forward_routed_fixed_support_active_count_min"] = float(min(fixed_counts_for_summary))
        mean_values["forward_routed_fixed_support_active_count_max"] = float(max(fixed_counts_for_summary))
        mean_values["forward_routed_fixed_support_active_count_mean"] = float(sum(fixed_counts_for_summary)) / float(len(fixed_counts_for_summary))
        mean_values["forward_routed_fixed_support_active_count_per_layer"] = ";".join(str(int(value)) for value in fixed_counts_for_summary)
    else:
        mean_values["forward_routed_fixed_support_available"] = 0.0
        mean_values["forward_routed_fixed_support_active_count_min"] = 0.0
        mean_values["forward_routed_fixed_support_active_count_max"] = 0.0
        mean_values["forward_routed_fixed_support_active_count_mean"] = 0.0
        mean_values["forward_routed_fixed_support_active_count_per_layer"] = ""
    return {
        "global_solution_entropy_mode": 1.0 if _forward_solution_entropy_mode(payload) in {"global_entmax", "global_sparsemax", "a1o_global_entmax"} else 0.0,
        "global_solution_entmax_alpha": float(_forward_solution_entmax_alpha(payload, epoch=epoch)),
        **mean_values,
    }

def attention_dictionary_scale_metrics(model: nn.Module) -> dict[str, Any]:
    """Record the functional QK/VO dictionary-owned scale groups actually used by forward."""

    qk_learned: list[float] = []
    qk_coordinate: list[float] = []
    qk_total: list[float] = []
    qk_weight_proxy: list[float] = []
    qk_per_block: list[str] = []
    vo_learned: list[float] = []
    vo_coordinate: list[float] = []
    vo_total: list[float] = []
    vo_weight_proxy: list[float] = []
    vo_per_block: list[str] = []
    qk_trainable = 0
    vo_trainable = 0
    for module_name, module in model.named_modules():
        qk_log = getattr(module, "dictionary_qk_log_scale", None)
        vo_log = getattr(module, "dictionary_vo_log_scale", None)
        if not isinstance(qk_log, torch.Tensor) or not isinstance(vo_log, torch.Tensor):
            continue
        qk_coord = getattr(module, "dictionary_qk_coordinate_log_scale", None)
        vo_coord = getattr(module, "dictionary_vo_coordinate_log_scale", None)
        qk_coord_value = qk_log.new_zeros(()) if not isinstance(qk_coord, torch.Tensor) else qk_coord
        vo_coord_value = vo_log.new_zeros(()) if not isinstance(vo_coord, torch.Tensor) else vo_coord
        qk_learned_value = float(qk_log.detach().float().exp().cpu())
        qk_coordinate_value = float(qk_coord_value.detach().float().exp().cpu())
        qk_total_value = qk_learned_value * qk_coordinate_value
        vo_learned_value = float(vo_log.detach().float().exp().cpu())
        vo_coordinate_value = float(vo_coord_value.detach().float().exp().cpu())
        vo_total_value = vo_learned_value * vo_coordinate_value
        q_projection = getattr(module, "query_projection", None)
        k_projection = getattr(module, "key_projection", None)
        v_projection = getattr(module, "value_projection", None)
        o_projection = getattr(module, "output_projection", None)
        q_rms = float(q_projection.current_weight().detach().float().pow(2).mean().sqrt().cpu()) if hasattr(q_projection, "current_weight") else 0.0
        k_rms = float(k_projection.current_weight().detach().float().pow(2).mean().sqrt().cpu()) if hasattr(k_projection, "current_weight") else 0.0
        v_rms = float(v_projection.current_weight().detach().float().pow(2).mean().sqrt().cpu()) if hasattr(v_projection, "current_weight") else 0.0
        o_rms = float(o_projection.current_weight().detach().float().pow(2).mean().sqrt().cpu()) if hasattr(o_projection, "current_weight") else 0.0
        qk_weight_proxy_value = q_rms * k_rms * qk_total_value
        vo_weight_proxy_value = v_rms * o_rms * vo_total_value
        qk_learned.append(qk_learned_value)
        qk_coordinate.append(qk_coordinate_value)
        qk_total.append(qk_total_value)
        qk_weight_proxy.append(qk_weight_proxy_value)
        qk_per_block.append(f"{module_name}:{qk_total_value:.9g}")
        vo_learned.append(vo_learned_value)
        vo_coordinate.append(vo_coordinate_value)
        vo_total.append(vo_total_value)
        vo_weight_proxy.append(vo_weight_proxy_value)
        vo_per_block.append(f"{module_name}:{vo_total_value:.9g}")
        qk_trainable += int(isinstance(qk_log, nn.Parameter) and qk_log.requires_grad)
        vo_trainable += int(isinstance(vo_log, nn.Parameter) and vo_log.requires_grad)

    return {
        "attention_scale_group_count": float(len(qk_total)),
        **_min_mean_max_metrics(qk_learned, "attention_qk_learned_scale"),
        **_min_mean_max_metrics(qk_coordinate, "attention_qk_coordinate_scale"),
        **_min_mean_max_metrics(qk_total, "attention_qk_total_scale"),
        **_min_mean_max_metrics(qk_weight_proxy, "attention_qk_weight_rms_scale_proxy"),
        "attention_qk_trainable_scale_count": float(qk_trainable),
        "attention_qk_total_scale_per_block": ";".join(qk_per_block),
        **_min_mean_max_metrics(vo_learned, "attention_vo_learned_scale"),
        **_min_mean_max_metrics(vo_coordinate, "attention_vo_coordinate_scale"),
        **_min_mean_max_metrics(vo_total, "attention_vo_total_scale"),
        **_min_mean_max_metrics(vo_weight_proxy, "attention_vo_weight_rms_scale_proxy"),
        "attention_vo_trainable_scale_count": float(vo_trainable),
        "attention_vo_total_scale_per_block": ";".join(vo_per_block),
    }


# --- Activation-contribution measurement helper -----------------------------
def _contribution_support_counts_from_mass(
    mass: torch.Tensor,
    *,
    threshold: float,
    mass_target: float,
) -> tuple[float, float, float]:
    probability = (mass.detach().float() / mass.detach().float().sum().clamp_min(1e-12)).flatten()
    if int(probability.numel()) <= 0:
        return 0.0, 0.0, 0.0
    hard_active = float((probability >= float(threshold)).sum().cpu())
    sorted_probability, _ = torch.sort(probability, descending=True)
    cumulative = torch.cumsum(sorted_probability, dim=0)
    if bool((cumulative >= float(mass_target)).any()):
        mass_index = int((cumulative >= float(mass_target)).nonzero(as_tuple=False)[0].item())
        mass_count = float(mass_index + 1)
    else:
        mass_count = float(int(sorted_probability.numel()))
    top100_mass = float(sorted_probability[: min(100, int(sorted_probability.numel()))].sum().cpu())
    return hard_active, mass_count, top100_mass


# --- Released sparsity/route reporting --------------------------------------
def natural_sparsity_metrics(
    model: nn.Module,
    config: dict[str, Any] | None,
    *,
    epoch: int,
) -> dict[str, float]:
    payload = config or {}
    forward_metrics = forward_solution_entropy_metrics(model, payload, epoch=epoch)
    effective_per_layer = float(forward_metrics.get("forward_solution_neff_mean", 0.0))
    layers = [layer for _name, layer in iter_dictionary_layers(model)]
    return {
        "natural_sparsity_expected_active_atoms_per_layer": effective_per_layer,
        "natural_sparsity_expected_active_atoms_total": effective_per_layer * float(max(1, len(layers))),
        **forward_metrics,
    }


# --- Recording and support enforcement --------------------------------------
def _dictionary_config_for_record(config: dict[str, Any], record: RunRecord | None) -> dict[str, Any]:
    """Return the released dictionary configuration for one DiR run."""

    dictionary_config = deepcopy(config.get("dictionary", {}))
    profile_name = str(
        getattr(record, "coefficient_quantization_profile", "") if record is not None else ""
    )
    if not profile_name:
        raise ValueError("DiR requires an explicit coefficient_quantization_profile")
    profiles = dictionary_config.get("coefficient_quantization_profiles", {})
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"Unknown coefficient_quantization_profile {profile_name!r}")
    profile_payload = profiles.get(profile_name)
    if not isinstance(profile_payload, dict):
        raise ValueError(f"coefficient_quantization_profile {profile_name!r} must be a JSON object")
    effective = deepcopy(profile_payload)
    effective.setdefault("profile_id", profile_name)
    dictionary_config["coefficient_quantization"] = effective
    return dictionary_config

def enforce_committed_sparse_support(model: nn.Module, *, enforce_coefficients: bool = True) -> None:
    for _name, layer in iter_dictionary_layers(model):
        layer.enforce_committed_sparse_support(enforce_coefficients=enforce_coefficients)

def clamp_quantized_coefficient_latents(model: nn.Module) -> None:
    for _name, layer in iter_dictionary_layers(model):
        layer.clamp_quantized_coefficient_latent_()
