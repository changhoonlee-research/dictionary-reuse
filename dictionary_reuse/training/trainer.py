"""Top-level training loop, profiles, and device helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .schema import (
    COEFFICIENT_DYNAMICS_FIELDS,
    COEFFICIENT_EPOCH_EVENT_DYNAMICS_FIELDS,
    LearningRateProfile,
    RunRecord,
    _console_log,
    _is_dense_model_family,
    _public_training_phase,
)
from ..model.routing import reset_entmax_runtime_state
from ..model.dictionary_operator import renormalize_dictionary_layers
from .sparsity import (
    _natural_sparsity_records_activation_contribution,
    natural_sparsity_metrics,
    prepare_forward_solution_entropy_layers,
    refresh_forward_solution_entropy_eval_metrics,
)
from .schedule import (
    _phase_cycle_from_config,
    _routed_gate_eval_enabled_for_epoch,
    assert_full_dictionary_integrity,
)
from .engine import (
    _snapshot_coefficient_vectors,
    _snapshot_dictionary_weights,
    _snapshot_update_state,
    _update_norms,
    activation_aware_contribution_metrics_by_layer,
    build_optimizer,
    collect_usage_rows,
    evaluate_model,
    optional_max_batches,
    routed_hard_gate_eval_metrics,
)

from .epoch_loop import EpochLoopContext, run_training_epochs

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    device: torch.device,
    profile: LearningRateProfile,
    model_family: str,
    run_id: str,
    task_id: str,
    basis_type: str,
    total_epochs: int,
    max_batches_per_epoch: int,
    record_epochs: set[int],
    include_epoch0_eval: bool = True,
    epoch0_eval_max_batches: int | None = None,
    record_eval_max_batches: int | None = 4,
    final_eval_max_batches: int | None = None,
    console_config: dict[str, Any] | None = None,
    snapshot_epochs: set[int] | None = None,
    natural_sparsity_config: dict[str, Any] | None = None,
    gradient_clip_config: dict[str, Any] | None = None,
    numerical_guard_config: dict[str, Any] | None = None,
    phase_schedule_config: dict[str, Any] | None = None,
    skip_initial_dictionary_normalization: bool = False,
    step_observer: Any | None = None,
    epoch_start_observer: Any | None = None,
    post_epoch_training_observer: Any | None = None,
    support_commit_post_observer: Any | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    reset_entmax_runtime_state()
    model.to(device)
    if bool((phase_schedule_config or {}).get("require_full_dictionary_integrity", False)):
        assert_full_dictionary_integrity(model, phase_config=phase_schedule_config or {})
    optimizer = build_optimizer(model, profile, model_family=model_family)
    if not _is_dense_model_family(model_family) and not bool(skip_initial_dictionary_normalization):
        renormalize_dictionary_layers(model)
    coefficient_reference_snapshot = _snapshot_coefficient_vectors(model) if not _is_dense_model_family(model_family) else {}
    zero_coefficient_dynamics_metrics = {name: 0.0 for name in COEFFICIENT_DYNAMICS_FIELDS}
    zero_coefficient_epoch_event_metrics = {name: 0.0 for name in COEFFICIENT_EPOCH_EVENT_DYNAMICS_FIELDS}
    measured_coefficient_dynamics_epochs = set(int(epoch) for epoch in record_epochs) | {int(total_epochs)}
    phase_cycle = _phase_cycle_from_config(phase_schedule_config)
    phase_schedule_enabled = bool(phase_cycle)
    phase_profile_id = str((phase_schedule_config or {}).get("profile_id", "disabled"))
    phase_unit = str((phase_schedule_config or {}).get("unit", "step")).lower()
    phase_epoch_pass_mode = phase_schedule_enabled and phase_unit in {"epoch", "epoch_pass", "data_pass", "full_pass"}
    phase_counts = {"C": 0, "D": 0, "B": 0}
    phase_global_step_index = 0
    phase_last_metadata = {
        "phase_schedule_profile": phase_profile_id,
        "phase_cycle_position": "",
        "phase_group": "joint" if not phase_schedule_enabled else "",
        "phase_trainable_groups": "all" if not phase_schedule_enabled else "",
        "phase_cumulative_coefficient_steps": 0,
        "phase_cumulative_dictionary_steps": 0,
        "phase_cumulative_backbone_steps": 0,
        "phase_trainable_C_param_count": 0,
        "phase_trainable_D_param_count": 0,
        "phase_trainable_B_param_count": 0,
        "phase_optimizer_C_param_count": 0,
        "phase_optimizer_D_param_count": 0,
        "phase_optimizer_B_param_count": 0,
    }
    curves: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    model_snapshots: dict[int, dict[str, torch.Tensor]] = {}
    snapshot_epochs = {int(epoch) for epoch in (snapshot_epochs or set())}
    previous_weights = _snapshot_dictionary_weights(model)
    cumulative_update_snapshot = _snapshot_update_state(model)
    update_reference_epoch = 0
    update_reference_label = "epoch0"
    global_step = 0
    public_phase = _public_training_phase(run_id, task_id)
    _console_log(console_config, f"  {public_phase}")
    _console_log(
        console_config,
        f"    epochs={int(total_epochs)} | batches={int(max_batches_per_epoch)}",
    )
    if bool(include_epoch0_eval):
        if not _is_dense_model_family(model_family):
            prepare_forward_solution_entropy_layers(
                model,
                natural_sparsity_config or {},
                active_groups=["C", "D"],
                epoch=0,
                force_record_metrics=True,
                record_metrics=True,
            )
        eval_metrics = evaluate_model(model, eval_loader, device=device, max_batches=epoch0_eval_max_batches)
        routed_gate_eval = routed_hard_gate_eval_metrics(
            model,
            eval_loader,
            device=device,
            max_batches=epoch0_eval_max_batches,
            enabled=_routed_gate_eval_enabled_for_epoch(natural_sparsity_config or {}, epoch=0),
            dynamic_enabled=bool((natural_sparsity_config or {}).get("forward_routed_gate_dynamic_eval_enabled", True)),
            fixed_enabled=bool((natural_sparsity_config or {}).get("forward_routed_gate_fixed_eval_enabled", True)),
            base_eval_metrics=eval_metrics,
        )
        refresh_forward_solution_entropy_eval_metrics(
            model,
            eval_loader,
            device=device,
            config=natural_sparsity_config or {},
            epoch=0,
            max_batches=epoch0_eval_max_batches,
        )
        update_norms = _update_norms(model, cumulative_update_snapshot)
        epoch0_natural_sparsity_metrics = natural_sparsity_metrics(model, natural_sparsity_config or {}, epoch=0)
        curves.append(
            {
                "run_id": run_id,
                "task_id": task_id,
                "basis_type": basis_type,
                "epoch": 0,
                "global_step": global_step,
                "effective_train_batches": global_step,
                "max_batches_per_epoch": int(max_batches_per_epoch),
                "epoch_train_seconds": "",
                "mean_batch_milliseconds": "",
                "relative_c_runtime_path": "not_applicable",
                "source_dynamic_route_seconds": "",
                "source_fixed_support_seconds": "",
                "target_fixed_support_seconds": "",
                "update_reference_epoch": update_reference_epoch,
                "update_reference_label": update_reference_label,
                "coefficient_cumulative_update_norm_from_update_reference": 0.0,
                "coefficient_cumulative_update_ratio_from_update_reference": 0.0,
                "coefficient_cumulative_radial_update_norm_from_update_reference": 0.0,
                "coefficient_cumulative_tangential_update_norm_from_update_reference": 0.0,
                "coefficient_cumulative_tangential_update_ratio_from_update_reference": 0.0,
                **epoch0_natural_sparsity_metrics,
                **zero_coefficient_dynamics_metrics,
                "coefficient_epoch_end_sparse_event_type": "",
                **zero_coefficient_epoch_event_metrics,
                "train_loss": "",
                "train_accuracy": "",
                "eval_loss": eval_metrics["loss"],
                "eval_accuracy": eval_metrics["accuracy"],
                "eval_count": eval_metrics["count"],
                **routed_gate_eval,
                "effective_update_ratio_mean": 0.0,
                "effective_update_ratio_max": 0.0,
                **update_norms,
                "coefficient_cumulative_update_norm_from_epoch0": 0.0,
                "coefficient_cumulative_update_ratio_from_epoch0": 0.0,
                "coefficient_cumulative_radial_update_norm_from_epoch0": 0.0,
                "coefficient_cumulative_tangential_update_norm_from_epoch0": 0.0,
                "coefficient_cumulative_tangential_update_ratio_from_epoch0": 0.0,
                **phase_last_metadata,
            }
        )
        activation_metrics: dict[str, dict[str, Any]] = {}
        attention_activation_metrics: dict[str, Any] = {}
        if _natural_sparsity_records_activation_contribution(natural_sparsity_config or {}):
            activation_metrics, attention_activation_metrics = activation_aware_contribution_metrics_by_layer(
                model,
                eval_loader,
                device=device,
                max_batches=optional_max_batches((natural_sparsity_config or {}).get("activation_aware_eval_max_batches"), record_eval_max_batches),
                threshold=float((natural_sparsity_config or {}).get("hard_active_threshold", 1e-3)),
                mass_target=float((natural_sparsity_config or {}).get("mass_target", 0.95)),
            )
        if curves and attention_activation_metrics:
            curves[-1].update(attention_activation_metrics)
        usage_rows.extend(
            collect_usage_rows(
                model,
                run_id=run_id,
                epoch=0,
                global_step=global_step,
                task_id=task_id,
                model_family=model_family,
                basis_type=basis_type,
                coefficient_reference_snapshot=coefficient_reference_snapshot,
                activation_contribution_metrics=activation_metrics,
                dictionary_entropy_config=natural_sparsity_config or {},
            )
        )

    global_step = run_training_epochs(
        EpochLoopContext(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            profile=profile,
            model_family=model_family,
            run_id=run_id,
            task_id=task_id,
            basis_type=basis_type,
            total_epochs=total_epochs,
            max_batches_per_epoch=max_batches_per_epoch,
            record_epochs=record_epochs,
            record_eval_max_batches=record_eval_max_batches,
            final_eval_max_batches=final_eval_max_batches,
            console_config=console_config,
            natural_sparsity_config=natural_sparsity_config,
            gradient_clip_config=gradient_clip_config,
            numerical_guard_config=numerical_guard_config,
            phase_schedule_config=phase_schedule_config,
            step_observer=step_observer,
            epoch_start_observer=epoch_start_observer,
            post_epoch_training_observer=post_epoch_training_observer,
            support_commit_post_observer=support_commit_post_observer,
            curves=curves,
            usage_rows=usage_rows,
            model_snapshots=model_snapshots,
            snapshot_epochs=snapshot_epochs,
            coefficient_reference_snapshot=coefficient_reference_snapshot,
            zero_coefficient_dynamics_metrics=zero_coefficient_dynamics_metrics,
            zero_coefficient_epoch_event_metrics=zero_coefficient_epoch_event_metrics,
            measured_coefficient_dynamics_epochs=measured_coefficient_dynamics_epochs,
            phase_cycle=phase_cycle,
            phase_schedule_enabled=phase_schedule_enabled,
            phase_profile_id=phase_profile_id,
            phase_epoch_pass_mode=phase_epoch_pass_mode,
            phase_counts=phase_counts,
            phase_global_step_index=phase_global_step_index,
            phase_last_metadata=phase_last_metadata,
            previous_weights=previous_weights,
            cumulative_update_snapshot=cumulative_update_snapshot,
            update_reference_epoch=update_reference_epoch,
            update_reference_label=update_reference_label,
            global_step=global_step,
            public_phase=public_phase,
        )
    )
    _console_log(console_config, f"    done | steps={global_step}")
    return model, curves, usage_rows, [], global_step, model_snapshots, {}

def _make_profile(config: dict[str, Any], name: str) -> LearningRateProfile:
    profile_payload = config["learning_rate_profiles"][name]
    non_dictionary_lr = float(profile_payload["non_dictionary_lr"])
    return LearningRateProfile(
        name=name,
        coefficient_lr=float(profile_payload["coefficient_lr"]),
        dictionary_lr=float(profile_payload["dictionary_lr"]),
        non_dictionary_lr=non_dictionary_lr,
        head_lr=float(profile_payload["head_lr"]) if profile_payload.get("head_lr") is not None else None,
    )

def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda was requested but CUDA is not available; switch the runtime to GPU or explicitly set runtime.device=cpu for local debugging")
    if requested in {"cuda", "cuda_if_available"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda_if_available":
        return torch.device("cpu")
    if requested != "cpu":
        raise ValueError(f"Unknown runtime.device={requested!r}; expected cuda, cuda_if_available, or cpu")
    return torch.device("cpu")

def _make_record(payload: dict[str, Any], config: dict[str, Any]) -> RunRecord:
    model_family = str(payload["model_family"])
    if "profile" in payload:
        profile_name = str(payload["profile"])
    elif _is_dense_model_family(model_family):
        profile_name = "dense_adamw_0003"
    else:
        profile_name = "dir_training"
    seed = int(payload.get("seed", config["runtime"].get("base_seed", 20260527)))
    basis_type = str(payload.get("basis_type", config["dictionary"].get("basis_type", "dct")))
    run_id = str(payload.get("run_id") or f"{model_family}_{profile_name}_{basis_type}_seed{seed}")
    return RunRecord(
        run_id=run_id,
        model_family=model_family,
        profile=profile_name,
        seed=seed,
        basis_type=basis_type,
        coefficient_quantization_profile=str(payload.get("coefficient_quantization_profile", "")),
        natural_sparsity_profile=str(payload.get("natural_sparsity_profile", "")),
        gradient_clip_profile=str(payload.get("gradient_clip_profile", "")),
        phase_schedule_profile=str(payload.get("phase_schedule_profile", "")),
        data_order_seed=int(payload.get("data_order_seed", 0) or 0),
    )
