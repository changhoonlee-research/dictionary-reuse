"""Per-epoch DiR optimization loop.

This module contains the hot training path. It is separated from ``trainer``
so the public training orchestration remains readable without changing tensor
operations, optimizer ordering, RNG use, or measurement timing.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .schema import (
    COEFFICIENT_DYNAMICS_FIELDS,
    LearningRateProfile,
    _console_log,
    _is_dense_model_family,
)
from ..model.routing import request_entmax_runtime_revalidation
from ..model.dictionary_operator import iter_dictionary_layers, renormalize_dictionary_layers
from .sparsity import (
    _forward_solution_entmax_alpha,
    _forward_solution_has_kindwise_alpha,
    _natural_sparsity_records_activation_contribution,
    attention_dictionary_scale_metrics,
    clamp_quantized_coefficient_latents,
    enforce_committed_sparse_support,
    natural_sparsity_metrics,
    prepare_forward_solution_entropy_layers,
    refresh_forward_solution_entropy_eval_metrics,
)
from .schedule import (
    _phase_config_with_active_parameter_scope,
    _phase_for_step,
    _phase_optimizer_param_counts,
    _phase_parameter_groups,
    _routed_gate_eval_enabled_for_epoch,
    _set_phase_trainability,
    commit_forward_routed_hard_support_masks_from_ema,
    commit_pending_relative_support_transitions_,
)
from .engine import (
    model_state_on_cpu,
    optional_max_batches,
    _snapshot_dictionary_weights,
    _clip_gradients,
    _coefficient_epoch_end_sparse_event_dynamics,
    _coefficient_scale_guard,
    _coefficient_update_dynamics,
    _effective_update_ratio,
    _empty_forward_support_commit_output_parity_metrics,
    _format_atom_usage_console_fields,
    _forward_output_parity_metrics,
    _gradient_clip_config_with_zero_inactive_allowed,
    _mean_step_metrics,
    _raise_if_nonfinite_loss,
    _snapshot_coefficient_vectors,
    _update_norms,
    activation_aware_contribution_metrics_by_layer,
    atom_usage_console_metrics,
    collect_raw_relative_c_epoch_rows,
    collect_usage_rows,
    evaluate_model,
    routed_hard_gate_eval_metrics,
)

@dataclass
class EpochLoopContext:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    eval_loader: DataLoader
    device: torch.device
    profile: LearningRateProfile
    model_family: str
    run_id: str
    task_id: str
    basis_type: str
    total_epochs: int
    max_batches_per_epoch: int
    record_epochs: set[int]
    record_eval_max_batches: int | None
    final_eval_max_batches: int | None
    console_config: dict[str, Any] | None
    natural_sparsity_config: dict[str, Any] | None
    gradient_clip_config: dict[str, Any] | None
    numerical_guard_config: dict[str, Any] | None
    phase_schedule_config: dict[str, Any] | None
    step_observer: Any | None
    epoch_start_observer: Any | None
    post_epoch_training_observer: Any | None
    support_commit_post_observer: Any | None
    curves: list[dict[str, Any]]
    usage_rows: list[dict[str, Any]]
    model_snapshots: dict[int, dict[str, torch.Tensor]]
    snapshot_epochs: set[int]
    coefficient_reference_snapshot: dict[str, torch.Tensor]
    zero_coefficient_dynamics_metrics: dict[str, float]
    zero_coefficient_epoch_event_metrics: dict[str, float]
    measured_coefficient_dynamics_epochs: set[int]
    phase_cycle: tuple[Any, ...] | list[Any]
    phase_schedule_enabled: bool
    phase_profile_id: str
    phase_epoch_pass_mode: bool
    phase_counts: dict[str, int]
    phase_global_step_index: int
    phase_last_metadata: dict[str, Any]
    previous_weights: dict[str, torch.Tensor]
    cumulative_update_snapshot: dict[str, dict[str, torch.Tensor]]
    update_reference_epoch: int
    update_reference_label: str
    global_step: int
    public_phase: str


# --- Epoch execution ---------------------------------------------------------
@dataclass
class EpochLoopState:
    phase_global_step_index: int
    phase_last_metadata: dict[str, Any]
    previous_weights: dict[str, torch.Tensor]
    global_step: int


@dataclass
class EpochTrainStats:
    total_loss: float
    total_correct: int
    total_count: int
    effective_batches: int
    coefficient_step_metrics: list[dict[str, float]]
    relative_c_runtime_path: str
    epoch_train_seconds: float
    mean_batch_milliseconds: float


def _activate_phase(
    context: EpochLoopContext,
    state: EpochLoopState,
) -> tuple[str, ...]:
    phase, phase_position = _phase_for_step(
        context.phase_cycle,
        state.phase_global_step_index,
    )
    active_groups = tuple(str(group) for group in (phase or {}).get("groups", ()))
    step_phase_config = _phase_config_with_active_parameter_scope(
        context.phase_schedule_config,
        phase,
    )
    phase_trainability = _set_phase_trainability(
        context.model,
        active_groups=active_groups,
        model_family=context.model_family,
        profile=context.profile,
        phase_config=step_phase_config,
    )
    phase_groups = _phase_parameter_groups(
        context.model,
        model_family=context.model_family,
        phase_config=step_phase_config,
    )
    phase_optimizer_counts = _phase_optimizer_param_counts(phase_groups)
    for group in active_groups:
        if group in context.phase_counts:
            context.phase_counts[group] += 1

    state.phase_last_metadata = {
        "phase_schedule_profile": context.phase_profile_id,
        "phase_cycle_position": int(phase_position),
        "phase_group": str((phase or {}).get("name", "+".join(active_groups))),
        "phase_trainable_groups": phase_trainability.get(
            "phase_trainable_groups", "+".join(active_groups)
        ),
        "phase_cumulative_coefficient_steps": context.phase_counts["C"],
        "phase_cumulative_dictionary_steps": context.phase_counts["D"],
        "phase_cumulative_backbone_steps": context.phase_counts["B"],
        "phase_trainable_C_param_count": phase_trainability.get(
            "phase_trainable_C_param_count", 0
        ),
        "phase_trainable_D_param_count": phase_trainability.get(
            "phase_trainable_D_param_count", 0
        ),
        "phase_trainable_B_param_count": phase_trainability.get(
            "phase_trainable_B_param_count", 0
        ),
        "phase_optimizer_C_param_count": phase_optimizer_counts.get("C", 0),
        "phase_optimizer_D_param_count": phase_optimizer_counts.get("D", 0),
        "phase_optimizer_B_param_count": phase_optimizer_counts.get("B", 0),
    }
    return active_groups


def _relative_c_runtime_path(model: nn.Module) -> str:
    relative_layers = [
        layer
        for _name, layer in iter_dictionary_layers(model)
        if bool(getattr(layer, "relative_coefficient_enabled", False))
    ]
    if not relative_layers:
        return "not_applicable"

    all_fixed_support = all(
        not bool(getattr(layer, "forward_routed_gate_enabled", False))
        or bool(
            getattr(
                layer,
                "_forward_routed_fixed_support_is_initialized",
                lambda: False,
            )()
        )
        for layer in relative_layers
    )
    return "source_fixed_support" if all_fixed_support else "source_dynamic_route"


def _start_epoch_timer(
    device: torch.device,
) -> tuple[bool, torch.cuda.Event | None, torch.cuda.Event | None, float | None]:
    cuda_timer = bool(device.type == "cuda" and torch.cuda.is_available())
    if cuda_timer:
        timer_start = torch.cuda.Event(enable_timing=True)
        timer_end = torch.cuda.Event(enable_timing=True)
        timer_start.record()
        return True, timer_start, timer_end, None
    return False, None, None, time.perf_counter()


def _finish_epoch_timer(
    *,
    cuda_timer: bool,
    timer_start: torch.cuda.Event | None,
    timer_end: torch.cuda.Event | None,
    cpu_start: float | None,
) -> float:
    if cuda_timer:
        assert timer_start is not None and timer_end is not None
        timer_end.record()
        timer_end.synchronize()
        return float(timer_start.elapsed_time(timer_end)) / 1000.0
    assert cpu_start is not None
    return float(time.perf_counter() - cpu_start)


def _task_loss_from_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch_metadata: Mapping[str, Any],
) -> torch.Tensor:
    task_a_count = int(batch_metadata.get("task_a_count", 0) or 0)
    task_b_count = int(batch_metadata.get("task_b_count", 0) or 0)
    if task_a_count <= 0 or task_b_count <= 0:
        return F.cross_entropy(logits, labels)

    if task_a_count + task_b_count != int(labels.numel()):
        raise RuntimeError(
            "joint batch metadata does not match concatenated label count: "
            f"task_a_count={task_a_count} task_b_count={task_b_count} "
            f"labels={int(labels.numel())}"
        )
    task_loss_a = F.cross_entropy(logits[:task_a_count], labels[:task_a_count])
    task_loss_b = F.cross_entropy(logits[task_a_count:], labels[task_a_count:])
    return task_loss_a + task_loss_b


def _train_single_epoch(
    context: EpochLoopContext,
    state: EpochLoopState,
    *,
    epoch: int,
) -> EpochTrainStats:
    model = context.model
    optimizer = context.optimizer
    measure_coefficient_dynamics = (
        not _is_dense_model_family(context.model_family)
        and int(epoch) in context.measured_coefficient_dynamics_epochs
    )
    active_groups: tuple[str, ...] = ()
    if context.phase_schedule_enabled and context.phase_epoch_pass_mode:
        active_groups = _activate_phase(context, state)
        state.phase_global_step_index += 1

    if context.epoch_start_observer is not None:
        context.epoch_start_observer(
            model=model,
            optimizer=optimizer,
            epoch=int(epoch),
            global_step=int(state.global_step),
        )

    relative_c_runtime_path = _relative_c_runtime_path(model)
    cuda_timer, timer_start, timer_end, cpu_start = _start_epoch_timer(context.device)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    effective_batches = 0
    coefficient_step_metrics: list[dict[str, float]] = []

    for batch_index, batch in enumerate(context.train_loader):
        if batch_index >= int(context.max_batches_per_epoch):
            break
        if context.phase_schedule_enabled and not context.phase_epoch_pass_mode:
            active_groups = _activate_phase(context, state)

        images = batch[0].to(context.device)
        labels = batch[1].to(context.device)
        batch_metadata = (
            batch[3] if len(batch) > 3 and isinstance(batch[3], Mapping) else {}
        )
        optimizer.zero_grad(set_to_none=True)
        record_route_diagnostics = bool(
            int(batch_index) == 0
            and (
                int(epoch) in context.record_epochs
                or int(epoch) == int(context.total_epochs)
            )
        )
        prepare_forward_solution_entropy_layers(
            model,
            context.natural_sparsity_config or {},
            active_groups=active_groups if context.phase_schedule_enabled else None,
            epoch=epoch,
            force_record_metrics=False,
            record_metrics=record_route_diagnostics,
        )
        logits = model(images)
        task_loss = _task_loss_from_batch(logits, labels, batch_metadata)
        loss = task_loss
        _raise_if_nonfinite_loss(
            loss,
            context.numerical_guard_config,
            run_id=context.run_id,
            epoch=epoch,
            batch_index=batch_index,
            phase="train",
        )

        coefficient_before = (
            _snapshot_coefficient_vectors(model) if measure_coefficient_dynamics else {}
        )
        loss.backward()
        step_gradient_clip_config = _gradient_clip_config_with_zero_inactive_allowed(
            context.gradient_clip_config or {},
            zero_inactive_allowed=False,
        )
        clip_metrics = _clip_gradients(
            model,
            step_gradient_clip_config or {},
            measure=measure_coefficient_dynamics,
        )
        optimizer.step()
        clamp_quantized_coefficient_latents(model)
        if relative_c_runtime_path in {"source_dynamic_route", "target_dynamic_support"}:
            commit_pending_relative_support_transitions_(model)
        if context.phase_schedule_enabled and not context.phase_epoch_pass_mode:
            state.phase_global_step_index += 1

        coefficient_after_optimizer = (
            _snapshot_coefficient_vectors(model) if measure_coefficient_dynamics else {}
        )
        if not _is_dense_model_family(context.model_family):
            renormalize_dictionary_layers(model)
        enforce_committed_sparse_support(model, enforce_coefficients=True)
        coefficient_after_projection = (
            _snapshot_coefficient_vectors(model) if measure_coefficient_dynamics else {}
        )

        if measure_coefficient_dynamics:
            coefficient_step_metrics.append(
                {
                    **clip_metrics,
                    **_coefficient_update_dynamics(
                        coefficient_before,
                        coefficient_after_optimizer,
                        coefficient_after_projection,
                    ),
                    **{
                        key: float(value) if isinstance(value, (int, float)) else value
                        for key, value in state.phase_last_metadata.items()
                    },
                }
            )
        elif (
            _is_dense_model_family(context.model_family)
            and int(epoch) in context.measured_coefficient_dynamics_epochs
        ):
            coefficient_step_metrics.append(
                {**context.zero_coefficient_dynamics_metrics, **clip_metrics}
            )

        state.global_step += 1
        if context.step_observer is not None:
            context.step_observer(
                model=model,
                optimizer=optimizer,
                epoch=int(epoch),
                batch_index=int(batch_index),
                global_step=int(state.global_step),
                task_loss=float(task_loss.detach().cpu()),
                total_loss=float(loss.detach().cpu()),
                clip_metrics=dict(clip_metrics),
                relative_c_runtime_path=str(relative_c_runtime_path),
            )

        effective_batches += 1
        total_loss += float(task_loss.detach().cpu()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total_count += int(labels.numel())

    epoch_train_seconds = _finish_epoch_timer(
        cuda_timer=cuda_timer,
        timer_start=timer_start,
        timer_end=timer_end,
        cpu_start=cpu_start,
    )
    mean_batch_milliseconds = (
        epoch_train_seconds * 1000.0 / max(1, int(effective_batches))
    )
    return EpochTrainStats(
        total_loss=total_loss,
        total_correct=total_correct,
        total_count=total_count,
        effective_batches=effective_batches,
        coefficient_step_metrics=coefficient_step_metrics,
        relative_c_runtime_path=relative_c_runtime_path,
        epoch_train_seconds=epoch_train_seconds,
        mean_batch_milliseconds=mean_batch_milliseconds,
    )


def _commit_support_if_due(
    context: EpochLoopContext,
    state: EpochLoopState,
    *,
    epoch: int,
) -> tuple[str, dict[str, Any]]:
    model = context.model
    config = context.natural_sparsity_config or {}
    commit_epoch = config.get("forward_routed_hard_support_commit_epoch")
    parity_metrics = _empty_forward_support_commit_output_parity_metrics()
    if (
        _is_dense_model_family(context.model_family)
        or commit_epoch is None
        or int(epoch) != int(commit_epoch)
    ):
        return "", parity_metrics

    parity_enabled = bool(
        config.get("forward_routed_hard_support_commit_output_parity_check", False)
    )
    parity_images: torch.Tensor | None = None
    parity_labels: torch.Tensor | None = None
    before_commit_logits: torch.Tensor | None = None
    was_training = bool(model.training)
    if parity_enabled:
        parity_batch = next(iter(context.eval_loader))
        sample_count = max(
            1,
            int(
                config.get(
                    "forward_routed_hard_support_commit_output_parity_sample_count",
                    128,
                )
            ),
        )
        parity_images = parity_batch[0][:sample_count].to(context.device)
        parity_labels = parity_batch[1][:sample_count].to(context.device)
        model.eval()
        with torch.no_grad():
            before_commit_logits = model(parity_images).detach()

    commit_report = commit_forward_routed_hard_support_masks_from_ema(
        model,
        alpha=(
            None
            if _forward_solution_has_kindwise_alpha(config)
            else float(_forward_solution_entmax_alpha(config, epoch=int(epoch)))
        ),
        tolerance=float(config.get("forward_routed_gate_support_tolerance", 1e-8)),
        score_eps=float(config.get("forward_routed_gate_score_eps", 1e-6)),
        entmax_iterations=int(
            config.get("forward_routed_gate_entmax_iterations", 16) or 16
        ),
        prefix="source_forward_routed_hard_support_commit",
        fold_relative_coordinate_scales=bool(
            config.get(
                "forward_routed_hard_support_commit_fold_relative_coordinate_into_"
                "dictionary_scale",
                True,
            )
        ),
    )
    if context.support_commit_post_observer is not None:
        context.support_commit_post_observer(
            model=model,
            optimizer=context.optimizer,
            epoch=int(epoch),
            global_step=int(state.global_step),
            commit_report=commit_report,
        )

    if parity_enabled:
        assert (
            parity_images is not None
            and parity_labels is not None
            and before_commit_logits is not None
        )
        model.eval()
        with torch.no_grad():
            after_commit_logits = model(parity_images).detach()
        parity_metrics = _forward_output_parity_metrics(
            before_commit_logits,
            after_commit_logits,
            parity_labels,
            max_abs_tolerance=float(
                config.get(
                    "forward_routed_hard_support_commit_output_parity_max_abs_tolerance",
                    5e-5,
                )
            ),
            relative_l2_tolerance=float(
                config.get(
                    "forward_routed_hard_support_commit_output_parity_relative_l2_tolerance",
                    1e-6,
                )
            ),
            prediction_mismatch_maximum=int(
                config.get(
                    "forward_routed_hard_support_commit_output_parity_prediction_"
                    "mismatch_maximum",
                    0,
                )
            ),
            accuracy_difference_maximum=float(
                config.get(
                    "forward_routed_hard_support_commit_output_parity_accuracy_"
                    "difference_maximum",
                    0.0,
                )
            ),
        )
        model.train(was_training)

    applied = bool(
        commit_report.get("source_forward_routed_hard_support_commit_applied", False)
    )
    return (
        "forward_routed_hard_support_commit" if applied else "",
        parity_metrics,
    )


def _record_epoch(
    context: EpochLoopContext,
    state: EpochLoopState,
    stats: EpochTrainStats,
    *,
    epoch: int,
    coefficient_epoch_event_type: str,
    coefficient_epoch_event_metrics: Mapping[str, Any],
    support_commit_output_parity_metrics: Mapping[str, Any],
) -> None:
    if epoch == int(context.total_epochs):
        eval_max_batches = context.final_eval_max_batches
    else:
        eval_max_batches = context.record_eval_max_batches

    model = context.model
    config = context.natural_sparsity_config or {}
    eval_metrics = evaluate_model(
        model,
        context.eval_loader,
        device=context.device,
        max_batches=eval_max_batches,
    )
    routed_gate_eval = routed_hard_gate_eval_metrics(
        model,
        context.eval_loader,
        device=context.device,
        max_batches=eval_max_batches,
        enabled=_routed_gate_eval_enabled_for_epoch(config, epoch=int(epoch)),
        dynamic_enabled=bool(
            config.get("forward_routed_gate_dynamic_eval_enabled", True)
        ),
        fixed_enabled=bool(config.get("forward_routed_gate_fixed_eval_enabled", True)),
        base_eval_metrics=eval_metrics,
    )
    refresh_forward_solution_entropy_eval_metrics(
        model,
        context.eval_loader,
        device=context.device,
        config=config,
        epoch=epoch,
        max_batches=eval_max_batches,
    )

    effective_update = _effective_update_ratio(model, state.previous_weights)
    state.previous_weights = _snapshot_dictionary_weights(model)
    update_norms = _update_norms(model, context.cumulative_update_snapshot)
    train_loss = stats.total_loss / max(1, stats.total_count)
    train_accuracy = stats.total_correct / max(1, stats.total_count)
    sparsity_metrics = natural_sparsity_metrics(model, config, epoch=epoch)
    atom_usage = atom_usage_console_metrics(model)
    atom_usage_fields = _format_atom_usage_console_fields(
        atom_usage,
        sparsity_metrics,
    )
    coefficient_dynamics = _mean_step_metrics(
        stats.coefficient_step_metrics,
        COEFFICIENT_DYNAMICS_FIELDS,
    )
    if not _is_dense_model_family(context.model_family):
        current_coefficients = _snapshot_coefficient_vectors(model)
        cumulative_coefficient_metrics = _coefficient_update_dynamics(
            context.coefficient_reference_snapshot,
            current_coefficients,
            current_coefficients,
        )
    else:
        cumulative_coefficient_metrics = {
            "coefficient_update_norm": 0.0,
            "coefficient_update_ratio": 0.0,
            "coefficient_radial_update_norm": 0.0,
            "coefficient_tangential_update_norm": 0.0,
            "coefficient_tangential_update_ratio": 0.0,
        }

    source_dynamic_route_seconds = (
        stats.epoch_train_seconds
        if stats.relative_c_runtime_path == "source_dynamic_route"
        else ""
    )
    source_fixed_support_seconds = (
        stats.epoch_train_seconds
        if stats.relative_c_runtime_path == "source_fixed_support"
        else ""
    )
    target_fixed_support_seconds = (
        stats.epoch_train_seconds
        if stats.relative_c_runtime_path == "target_fixed_support"
        else ""
    )
    context.curves.append(
        {
            "run_id": context.run_id,
            "task_id": context.task_id,
            "basis_type": context.basis_type,
            "epoch": epoch,
            "global_step": state.global_step,
            "effective_train_batches": state.global_step,
            "max_batches_per_epoch": int(context.max_batches_per_epoch),
            "epoch_train_seconds": stats.epoch_train_seconds,
            "mean_batch_milliseconds": stats.mean_batch_milliseconds,
            "relative_c_runtime_path": stats.relative_c_runtime_path,
            "source_dynamic_route_seconds": source_dynamic_route_seconds,
            "source_fixed_support_seconds": source_fixed_support_seconds,
            "target_fixed_support_seconds": target_fixed_support_seconds,
            "update_reference_epoch": int(context.update_reference_epoch),
            "update_reference_label": context.update_reference_label,
            "coefficient_cumulative_update_norm_from_update_reference": (
                cumulative_coefficient_metrics.get("coefficient_update_norm", 0.0)
            ),
            "coefficient_cumulative_update_ratio_from_update_reference": (
                cumulative_coefficient_metrics.get("coefficient_update_ratio", 0.0)
            ),
            "coefficient_cumulative_radial_update_norm_from_update_reference": (
                cumulative_coefficient_metrics.get(
                    "coefficient_radial_update_norm", 0.0
                )
            ),
            "coefficient_cumulative_tangential_update_norm_from_update_reference": (
                cumulative_coefficient_metrics.get(
                    "coefficient_tangential_update_norm", 0.0
                )
            ),
            "coefficient_cumulative_tangential_update_ratio_from_update_reference": (
                cumulative_coefficient_metrics.get(
                    "coefficient_tangential_update_ratio", 0.0
                )
            ),
            **sparsity_metrics,
            **attention_dictionary_scale_metrics(model),
            **atom_usage,
            **coefficient_dynamics,
            "coefficient_epoch_end_sparse_event_type": coefficient_epoch_event_type,
            **coefficient_epoch_event_metrics,
            **support_commit_output_parity_metrics,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_count": eval_metrics["count"],
            **routed_gate_eval,
            "effective_update_ratio_mean": effective_update.get("mean", 0.0),
            "effective_update_ratio_max": effective_update.get("max", 0.0),
            **update_norms,
            "coefficient_cumulative_update_norm_from_epoch0": (
                cumulative_coefficient_metrics.get("coefficient_update_norm", 0.0)
            ),
            "coefficient_cumulative_update_ratio_from_epoch0": (
                cumulative_coefficient_metrics.get("coefficient_update_ratio", 0.0)
            ),
            "coefficient_cumulative_radial_update_norm_from_epoch0": (
                cumulative_coefficient_metrics.get(
                    "coefficient_radial_update_norm", 0.0
                )
            ),
            "coefficient_cumulative_tangential_update_norm_from_epoch0": (
                cumulative_coefficient_metrics.get(
                    "coefficient_tangential_update_norm", 0.0
                )
            ),
            "coefficient_cumulative_tangential_update_ratio_from_epoch0": (
                cumulative_coefficient_metrics.get(
                    "coefficient_tangential_update_ratio", 0.0
                )
            ),
            **state.phase_last_metadata,
        }
    )
    _console_log(
        context.console_config,
        f"    {epoch}/{int(context.total_epochs)} | "
        f"acc={100.0 * float(eval_metrics['accuracy']):.2f}% | "
        f"loss={float(eval_metrics['loss']):.4f} | "
        f"train={train_loss:.4f} | "
        f"batch_ms={stats.mean_batch_milliseconds:.1f}"
        + (f" | {atom_usage_fields}" if atom_usage_fields else ""),
    )

    activation_metrics = {}
    attention_activation_metrics: dict[str, Any] = {}
    if _natural_sparsity_records_activation_contribution(config):
        activation_metrics, attention_activation_metrics = (
            activation_aware_contribution_metrics_by_layer(
                model,
                context.eval_loader,
                device=context.device,
                max_batches=optional_max_batches(
                    config.get("activation_aware_eval_max_batches"),
                    context.record_eval_max_batches,
                ),
                threshold=float(config.get("hard_active_threshold", 1e-3)),
                mass_target=float(config.get("mass_target", 0.95)),
            )
        )
    if context.curves and attention_activation_metrics:
        context.curves[-1].update(attention_activation_metrics)
    context.usage_rows.extend(
        collect_usage_rows(
            model,
            run_id=context.run_id,
            epoch=epoch,
            global_step=state.global_step,
            task_id=context.task_id,
            model_family=context.model_family,
            basis_type=context.basis_type,
            coefficient_reference_snapshot=context.coefficient_reference_snapshot,
            activation_contribution_metrics=activation_metrics,
            dictionary_entropy_config=config,
        )
    )


def run_training_epochs(context: EpochLoopContext) -> int:
    """Run all optimization epochs while keeping reporting and phase steps explicit."""

    state = EpochLoopState(
        phase_global_step_index=context.phase_global_step_index,
        phase_last_metadata=context.phase_last_metadata,
        previous_weights=context.previous_weights,
        global_step=context.global_step,
    )

    for epoch in range(1, int(context.total_epochs) + 1):
        if int(epoch) in context.record_epochs:
            request_entmax_runtime_revalidation()
        context.model.train()
        stats = _train_single_epoch(context, state, epoch=epoch)

        coefficient_epoch_event_metrics = dict(
            context.zero_coefficient_epoch_event_metrics
        )
        measure_coefficient_dynamics = (
            not _is_dense_model_family(context.model_family)
            and int(epoch) in context.measured_coefficient_dynamics_epochs
        )
        if not _is_dense_model_family(context.model_family):
            renormalize_dictionary_layers(context.model)
        coefficient_snapshot_before_epoch_event = (
            _snapshot_coefficient_vectors(context.model)
            if measure_coefficient_dynamics
            else {}
        )

        event_type, parity_metrics = _commit_support_if_due(
            context,
            state,
            epoch=epoch,
        )
        if measure_coefficient_dynamics and event_type:
            coefficient_epoch_event_metrics = (
                _coefficient_epoch_end_sparse_event_dynamics(
                    coefficient_snapshot_before_epoch_event,
                    _snapshot_coefficient_vectors(context.model),
                )
            )

        if (
            not _is_dense_model_family(context.model_family)
            and bool(
                (context.numerical_guard_config or {}).get(
                    "check_coefficients_after_epoch", True
                )
            )
        ):
            _coefficient_scale_guard(
                context.model,
                context.numerical_guard_config,
                run_id=context.run_id,
                epoch=epoch,
                phase="epoch_end",
            )

        if context.post_epoch_training_observer is not None:
            context.post_epoch_training_observer(
                model=context.model,
                optimizer=context.optimizer,
                epoch=int(epoch),
                global_step=int(state.global_step),
            )

        if (
            not _is_dense_model_family(context.model_family)
            and int(epoch) not in context.record_epochs
            and int(epoch) != int(context.total_epochs)
        ):
            context.usage_rows.extend(
                collect_raw_relative_c_epoch_rows(
                    context.model,
                    run_id=context.run_id,
                    epoch=int(epoch),
                    global_step=int(state.global_step),
                    task_id=context.task_id,
                    model_family=context.model_family,
                    basis_type=context.basis_type,
                )
            )
        if int(epoch) in context.snapshot_epochs:
            context.model_snapshots[int(epoch)] = model_state_on_cpu(context.model)

        if epoch in context.record_epochs or epoch == context.total_epochs:
            _record_epoch(
                context,
                state,
                stats,
                epoch=epoch,
                coefficient_epoch_event_type=event_type,
                coefficient_epoch_event_metrics=coefficient_epoch_event_metrics,
                support_commit_output_parity_metrics=parity_metrics,
            )

    return int(state.global_step)
