"""Final-paper training matrix and endpoint checkpoint stage."""

from __future__ import annotations


# Condition-by-condition training
from copy import deepcopy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

from ..dictionary_transfer import ActiveDictionaryFreezeController, verify_active_dictionary_ownership
from ..measurements.direct import (
    block_update_cka_against_reference,
    prepare_block_update_cka_reference,
)
from ..supplementary.diagnostics import (
    prepare_representation_rsa_reference,
    representation_rsa_against_reference,
)
from ..training import build_eval_loader, build_model
from .runtime import (
    _checkpoint_metadata_for_spec,
    _load_model_checkpoint,
    _measurement_batches,
    _offload_trained_model,
    _replace_training_rows_for_run,
    _reset_dense_head,
    _save_model,
    _train_one,
)


def record_epochs(final_epoch: int) -> set[int]:
    return {
        epoch
        for epoch in {0, 1, 2, 5, 10, 20, 52, int(final_epoch)}
        if epoch <= int(final_epoch)
    }


def same_task_trajectory_observer(
    source_model: nn.Module,
    *,
    role: dict[str, Any],
    device: torch.device,
    epochs: set[int],
) -> tuple[Callable[..., None], dict[int, dict[str, Any]]]:
    """Compare selected Target epochs with one fixed Source block-update reference."""

    batches, _sample_ids = _measurement_batches(
        role, task_key="task1", count=128
    )
    reference = prepare_block_update_cka_reference(
        source_model, batches, device=device
    )
    rsa_reference = prepare_representation_rsa_reference(
        source_model, batches, device=device
    )
    selected_epochs = {int(epoch) for epoch in epochs}
    rows: dict[int, dict[str, Any]] = {}

    def observer(**payload: Any) -> None:
        epoch = int(payload.get("epoch", 0))
        if epoch not in selected_epochs:
            return
        metrics = block_update_cka_against_reference(
            reference, payload["model"], batches, device=device
        )
        if epoch == 0:
            metrics.update(
                representation_rsa_against_reference(
                    rsa_reference, payload["model"], batches, device=device
                )
            )
        rows[epoch] = metrics

    return observer, rows


def attach_same_task_trajectory(
    curves: list[dict[str, Any]], trajectory: Mapping[int, Mapping[str, Any]]
) -> None:
    """Attach compact trajectory metrics to the existing recorded training rows."""

    matrix_fields = {
        "same_task_trajectory_cls_debiased_cka_12x12",
        "same_task_trajectory_patch_debiased_cka_12x12",
        "same_task_trajectory_cls_rsa_spearman_12x12",
        "same_task_trajectory_patch_rsa_spearman_12x12",
    }
    for row in curves:
        metrics = trajectory.get(int(row.get("epoch", -1)))
        if not metrics:
            continue
        for key, value in metrics.items():
            row[key] = (
                json.dumps(value, separators=(",", ":"))
                if key in matrix_fields
                else value
            )


def fresh_dir_model(
    role: dict[str, Any], source_record: Any, source_dictionary: dict[str, Any], *, seed: int
) -> nn.Module:
    return build_model(
        role,
        model_family=source_record.model_family,
        seed=int(seed),
        basis_type=source_record.basis_type,
        dictionary_config_override=source_dictionary,
    )


def full_state_target_from_source(
    source_model: nn.Module,
    *,
    role: dict[str, Any],
    source_record: Any,
    source_dictionary: dict[str, Any],
    head_seed: int,
) -> nn.Module:
    """Copy the complete Source backbone and replace only the classification head."""

    target = deepcopy(source_model).cpu()
    fresh = fresh_dir_model(
        role, source_record, source_dictionary, seed=int(head_seed)
    )
    fresh_requires_grad = {
        name: bool(parameter.requires_grad) for name, parameter in fresh.named_parameters()
    }
    target.classification_head = deepcopy(fresh.classification_head).cpu()
    for name, parameter in target.named_parameters():
        if name not in fresh_requires_grad:
            raise RuntimeError(f"DiR full-state trainability template missing parameter: {name}")
        parameter.requires_grad_(fresh_requires_grad[name])
    del fresh
    return target


def backbone_exact_copy_audit(source_model: nn.Module, target_model: nn.Module) -> dict[str, Any]:
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    source_keys = {key for key in source_state if not key.startswith("classification_head.")}
    target_keys = {key for key in target_state if not key.startswith("classification_head.")}
    missing = sorted(source_keys - target_keys)
    extra = sorted(target_keys - source_keys)
    mismatches: list[str] = []
    for key in sorted(source_keys & target_keys):
        left = source_state[key].detach().cpu()
        right = target_state[key].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            mismatches.append(key)
    return {
        "passed": not missing and not extra and not mismatches,
        "contract": "exact_full_backbone_state_copy_excluding_classification_head",
        "checked_state_tensor_count": len(source_keys & target_keys),
        "missing_keys": missing,
        "extra_keys": extra,
        "mismatched_keys": mismatches,
        "classification_head_transferred": False,
    }


def targets_exactly_equal_audit(left: nn.Module, right: nn.Module) -> dict[str, Any]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if set(left_state) != set(right_state):
        return {
            "passed": False,
            "contract": "dictionary_fixed_and_dictionary_trainable_identical_initial_state",
            "reason": "state_dict_key_mismatch",
        }
    mismatches = [
        key
        for key in left_state
        if left_state[key].shape != right_state[key].shape
        or left_state[key].dtype != right_state[key].dtype
        or not torch.equal(left_state[key].detach().cpu(), right_state[key].detach().cpu())
    ]
    return {
        "passed": not mismatches,
        "contract": "dictionary_fixed_and_dictionary_trainable_identical_initial_state",
        "checked_state_tensor_count": len(left_state),
        "mismatched_keys": mismatches,
    }


def backbone_output_equivalence_audit(
    source_model: nn.Module,
    targets: Mapping[str, nn.Module],
    *,
    role: dict[str, Any],
    device: Any,
    sample_count: int = 8,
) -> dict[str, Any]:
    """Cheap e0 audit of head-independent full-state equivalence."""

    eval_loader = build_eval_loader(role, task_key="task2")
    images, _labels = next(iter(eval_loader))
    images = images[: max(1, int(sample_count))].to(device)

    def capture(model: nn.Module) -> torch.Tensor:
        was_training = bool(model.training)
        model.to(device).eval()
        try:
            with torch.no_grad():
                _logits, taps = model.forward_with_measurement_tensors(
                    input_images=images,
                    include_internal_block_taps=False,
                    requested_tap_names={"pre_classifier"},
                )
                return taps["pre_classifier"].detach().cpu()
        finally:
            model.train(was_training)
            model.cpu()

    source_output = capture(source_model)
    comparisons: dict[str, Any] = {}
    for condition, target in targets.items():
        target_output = capture(target)
        difference = (target_output - source_output).abs()
        comparisons[condition] = {
            "passed": bool(torch.equal(target_output, source_output)),
            "sample_count": int(target_output.shape[0]),
            "max_abs_difference": float(difference.max().item()) if difference.numel() else 0.0,
            "mean_abs_difference": float(difference.mean().item()) if difference.numel() else 0.0,
        }
    return {
        "passed": all(bool(item.get("passed", False)) for item in comparisons.values()),
        "contract": "e0_head_independent_pre_classifier_output_exact_equivalence",
        "task": "task2",
        "comparisons": comparisons,
    }


def target_support_reopen_observer() -> tuple[Callable[..., None], dict[str, Any]]:
    report: dict[str, Any] = {
        "applied": False,
        "epoch": None,
        "dictionary_layers_reopened": 0,
        "dictionary_layers_target_route_state_reset": 0,
    }

    def observer(**payload: Any) -> None:
        if report["applied"]:
            return
        reopened = 0
        route_state_reset = 0
        for module in payload["model"].modules():
            # Keep copied dictionary parameters, but remove Source-only routing
            # history/EMA/fixed-support state before the first Target minibatch.
            reset_target_state = getattr(
                module, "reset_forward_routed_target_adaptation_state", None
            )
            if callable(reset_target_state):
                initialized = getattr(
                    module, "_forward_routed_fixed_support_is_initialized", None
                )
                was_fixed = bool(initialized()) if callable(initialized) else False
                reset_target_state()
                route_state_reset += 1
                reopened += int(was_fixed)
        report.update(
            {
                "applied": True,
                "epoch": int(payload.get("epoch", 0)),
                "dictionary_layers_reopened": int(reopened),
                "dictionary_layers_target_route_state_reset": int(route_state_reset),
                "contract": (
                    "copy_Source_endpoint_at_e0_then_clear_Source_route_history_usage_EMA_"
                    "and_fixed_support_before_first_Target_minibatch"
                ),
            }
        )

    return observer, report


@dataclass
class TrainingMatrixRunner:
    """Train/load the seven endpoint conditions while preserving one shared contract."""

    role: dict[str, Any]
    source_record: Any
    source_dictionary: dict[str, Any]
    training: dict[str, Any]
    device: Any
    checkpoint_specs: Mapping[str, Mapping[str, Any]]
    endpoint_paths: Mapping[str, Path]
    checkpoint_metadata_base: Mapping[str, Any]
    training_contract_sha256: str
    resume_from_checkpoints: bool
    completed_conditions: set[str]
    snapshot_epoch: int
    snapshot_epochs: set[int]
    write_progress: Callable[[Mapping[str, Any]], None]
    persist_training_progress: Callable[[str], None]
    record_warning: Callable[..., None]
    training_rows: list[dict[str, Any]] = field(default_factory=list)
    ownership: dict[str, Any] = field(default_factory=dict)
    initialization_audit: dict[str, Any] = field(default_factory=dict)

    def run(self) -> tuple[nn.Module, dict[str, nn.Module], nn.Module, nn.Module, nn.Module]:
        dir_source = self._dir_source()
        dir_same = self._dir_same_task(dir_source)
        dir_targets = self._dir_different_task(dir_source, dir_same)
        dense_source = self._dense_source()
        dense_same = self._dense_same_task(dense_source)
        dense_different = self._dense_different_task(dense_source)
        return dir_source, dir_targets, dense_source, dense_same, dense_different

    def _dir_source(self) -> nn.Module:
        condition = "dir_source_a"
        self.write_progress({"stage": "training", "condition": condition})
        model = fresh_dir_model(
            self.role,
            self.source_record,
            self.source_dictionary,
            seed=int(self.training["dir_source_seed"]),
        )
        if self._should_load(condition):
            self._load("dir_source", model)
        else:
            epochs = int(self.training["dir_source_a_epochs"])
            model, curves, _ = _train_one(
                model=model,
                role=self.role,
                task_key="task1",
                run_id=condition,
                model_family=self.source_record.model_family,
                basis_type=self.source_record.basis_type,
                profile_name=self.source_record.profile,
                epochs=epochs,
                record_epochs=record_epochs(epochs),
                data_order_seed=int(self.training["dir_source_data_order_seed"]),
                device=self.device,
                natural_profile=self.source_record.natural_sparsity_profile,
                phase_profile=self.source_record.phase_schedule_profile,
                gradient_profile=self.source_record.gradient_clip_profile,
            )
            self._replace_rows(condition, curves)
            self._save("dir_source", model)
            self.persist_training_progress("dir_source_training_progress")
        return _offload_trained_model(model, execution_device=self.device)

    def _dir_same_task(self, dir_source: nn.Module) -> nn.Module:
        condition = "dir_same_task"
        self.write_progress({"stage": "training", "condition": condition})
        model = fresh_dir_model(
            self.role,
            self.source_record,
            self.source_dictionary,
            seed=int(self.training["dir_same_task_seed"]),
        )
        if self._should_load(condition):
            self._load(condition, model)
            verification = verify_active_dictionary_ownership(
                dir_source, model, include_classification_head=True
            )
            if not verification.get("passed", False):
                raise RuntimeError("DiR same-task endpoint failed Source-active D/scale verification")
        else:
            epochs = int(self.training["dir_target_epochs"])
            trajectory_epochs = record_epochs(epochs)
            trajectory_observer, trajectory_rows = same_task_trajectory_observer(
                dir_source,
                role=self.role,
                device=self.device,
                epochs=trajectory_epochs,
            )
            model = model.to(self.device)
            controller = ActiveDictionaryFreezeController(
                model,
                dir_source,
                include_classification_head=True,
                copy_active_coefficients=False,
            )
            trajectory_observer(model=model, epoch=0)

            def post_epoch_observer(**payload: Any) -> None:
                controller.post_epoch_observer(**payload)
                trajectory_observer(**payload)

            model, curves, snapshots = _train_one(
                model=model,
                role=self.role,
                task_key="task1",
                run_id=condition,
                model_family=self.source_record.model_family,
                basis_type=self.source_record.basis_type,
                profile_name=self.source_record.profile,
                epochs=epochs,
                record_epochs=trajectory_epochs,
                data_order_seed=int(self.training["dir_same_task_data_order_seed"]),
                device=self.device,
                natural_profile=self.source_record.natural_sparsity_profile,
                phase_profile=self.source_record.phase_schedule_profile,
                gradient_profile=self.source_record.gradient_clip_profile,
                step_observer=controller.step_observer,
                post_epoch_observer=post_epoch_observer,
                support_commit_post_observer=controller.post_epoch_observer,
                preserve_relative_coordinate_corrections_at_commit=True,
                snapshot_epochs=set(self.snapshot_epochs),
                skip_initial_dictionary_normalization=True,
            )
            verification = controller.finalize(model)
            if not verification.get("passed", False):
                self.record_warning(
                    stage="dir_same_task_active_dictionary_final_verification",
                    message="same-task active D/scale verification failed after final restore",
                )
            attach_same_task_trajectory(curves, trajectory_rows)
            self._replace_rows(condition, curves)
            snapshot_model = fresh_dir_model(
                self.role,
                self.source_record,
                self.source_dictionary,
                seed=int(self.training["dir_same_task_seed"]),
            )
            self._save_snapshot(condition, snapshot_model, snapshots)
            self._save(condition, model)
            self.persist_training_progress("dir_same_task_training_progress")
            del snapshots, snapshot_model
        self.ownership[condition] = {
            **verification,
            "condition_contract": (
                "fresh_Target_C_route_support_with_Source_active_D_and_D_owned_scales_fixed"
            ),
            "coefficient_initialization": "fresh_target",
        }
        return _offload_trained_model(model, execution_device=self.device)

    def _dir_different_task(
        self, dir_source: nn.Module, dir_same: nn.Module
    ) -> dict[str, nn.Module]:
        conditions = ("dir_dictionary_fixed", "dir_dictionary_trainable")
        targets = {
            name: full_state_target_from_source(
                dir_source,
                role=self.role,
                source_record=self.source_record,
                source_dictionary=self.source_dictionary,
                head_seed=int(self.training["dir_different_task_head_seed"]),
            )
            for name in conditions
        }
        self._audit_different_task_initialization(dir_source, targets)
        output: dict[str, nn.Module] = {"dir_same_task": dir_same}
        for condition in conditions:
            output[condition] = self._dir_different_condition(
                condition, dir_source, targets[condition]
            )
        return output

    def _dir_different_condition(
        self, condition: str, dir_source: nn.Module, target: nn.Module
    ) -> nn.Module:
        self.write_progress({"stage": "training", "condition": condition})
        is_fixed = condition == "dir_dictionary_fixed"
        # "Dictionary-Fixed" anchors only Source-active D slices plus D-owned
        # scales. Inactive D atom slices intentionally remain trainable so Target
        # adaptation may recruit previously inactive dictionary capacity.
        if self._should_load(condition):
            self._load(condition, target)
            verification = self._different_task_ownership(
                condition, dir_source, target, loaded=True
            )
        else:
            target = target.to(self.device)
            reopen_support, reopen_report = target_support_reopen_observer()
            controller = (
                ActiveDictionaryFreezeController(
                    target,
                    dir_source,
                    include_classification_head=False,
                    copy_active_coefficients=True,
                )
                if is_fixed
                else None
            )
            epochs = int(self.training["dir_target_epochs"])
            target, curves, snapshots = _train_one(
                model=target,
                role=self.role,
                task_key="task2",
                run_id=condition,
                model_family=self.source_record.model_family,
                basis_type=self.source_record.basis_type,
                profile_name=self.source_record.profile,
                epochs=epochs,
                record_epochs=record_epochs(epochs),
                data_order_seed=int(self.training["different_task_data_order_seed"]),
                device=self.device,
                natural_profile=self.source_record.natural_sparsity_profile,
                phase_profile=self.source_record.phase_schedule_profile,
                gradient_profile=self.source_record.gradient_clip_profile,
                step_observer=(controller.step_observer if controller is not None else None),
                post_epoch_observer=(
                    controller.post_epoch_observer if controller is not None else None
                ),
                support_commit_post_observer=(
                    controller.post_epoch_observer if controller is not None else None
                ),
                epoch_start_observer=reopen_support,
                preserve_relative_coordinate_corrections_at_commit=True,
                snapshot_epochs=set(self.snapshot_epochs),
                skip_initial_dictionary_normalization=True,
            )
            if not reopen_report.get("applied", False):
                raise RuntimeError(
                    f"{condition} failed to unlock copied Source support before Target training"
                )
            self.initialization_audit[condition]["support_reopen"] = dict(reopen_report)
            verification = self._different_task_ownership(
                condition, dir_source, target, loaded=False, controller=controller
            )
            self._replace_rows(condition, curves)
            snapshot_model = full_state_target_from_source(
                dir_source,
                role=self.role,
                source_record=self.source_record,
                source_dictionary=self.source_dictionary,
                head_seed=int(self.training["dir_different_task_head_seed"]),
            )
            self._save_snapshot(condition, snapshot_model, snapshots)
            self._save(condition, target)
            self.persist_training_progress(f"{condition}_training_progress")
            del snapshots, snapshot_model
        self.ownership[condition] = verification
        return _offload_trained_model(target, execution_device=self.device)

    def _dense_source(self) -> nn.Module:
        condition = "dense_source_a"
        self.write_progress({"stage": "training", "condition": condition})
        model = build_model(
            self.role, model_family="dense_vit", seed=int(self.training["dense_source_seed"])
        )
        if self._should_load(condition):
            self._load("dense_source", model)
        else:
            epochs = int(self.training["dense_source_a_epochs"])
            model, curves, _ = _train_one(
                model=model,
                role=self.role,
                task_key="task1",
                run_id=condition,
                model_family="dense_vit",
                basis_type="dense",
                profile_name=str(self.training["dense_learning_rate_profile"]),
                epochs=epochs,
                record_epochs=record_epochs(epochs),
                data_order_seed=int(self.training["dense_source_data_order_seed"]),
                device=self.device,
                gradient_profile=str(self.training["dense_gradient_clip_profile"]),
            )
            self._replace_rows(condition, curves)
            self._save("dense_source", model)
            self.persist_training_progress("dense_source_training_progress")
        return _offload_trained_model(model, execution_device=self.device)

    def _dense_same_task(self, dense_source: nn.Module) -> nn.Module:
        condition = "dense_same_task"
        self.write_progress({"stage": "training", "condition": condition})
        model = build_model(
            self.role, model_family="dense_vit", seed=int(self.training["dense_same_task_seed"])
        )
        if self._should_load(condition):
            self._load(condition, model)
        else:
            epochs = int(self.training["dense_target_epochs"])
            trajectory_epochs = record_epochs(epochs)
            trajectory_observer, trajectory_rows = same_task_trajectory_observer(
                dense_source,
                role=self.role,
                device=self.device,
                epochs=trajectory_epochs,
            )
            trajectory_observer(model=model, epoch=0)
            model, curves, snapshots = _train_one(
                model=model,
                role=self.role,
                task_key="task1",
                run_id=condition,
                model_family="dense_vit",
                basis_type="dense",
                profile_name=str(self.training["dense_learning_rate_profile"]),
                epochs=epochs,
                record_epochs=trajectory_epochs,
                data_order_seed=int(self.training["dense_same_task_data_order_seed"]),
                device=self.device,
                gradient_profile=str(self.training["dense_gradient_clip_profile"]),
                post_epoch_observer=trajectory_observer,
                snapshot_epochs=set(self.snapshot_epochs),
            )
            attach_same_task_trajectory(curves, trajectory_rows)
            self._replace_rows(condition, curves)
            snapshot_model = build_model(
                self.role,
                model_family="dense_vit",
                seed=int(self.training["dense_same_task_seed"]),
            )
            self._save_snapshot(condition, snapshot_model, snapshots)
            self._save(condition, model)
            self.persist_training_progress("dense_same_task_training_progress")
            del snapshots, snapshot_model
        return _offload_trained_model(model, execution_device=self.device)

    def _dense_different_task(self, dense_source: nn.Module) -> nn.Module:
        condition = "dense_different_task"
        self.write_progress({"stage": "training", "condition": condition})
        model = deepcopy(dense_source).cpu()
        _reset_dense_head(model, seed=int(self.training["dense_different_task_head_seed"]))
        if self._should_load(condition):
            self._load(condition, model)
        else:
            epochs = int(self.training["dense_target_epochs"])
            model, curves, snapshots = _train_one(
                model=model,
                role=self.role,
                task_key="task2",
                run_id=condition,
                model_family="dense_vit",
                basis_type="dense",
                profile_name=str(self.training["dense_learning_rate_profile"]),
                epochs=epochs,
                record_epochs=record_epochs(epochs),
                data_order_seed=int(self.training["different_task_data_order_seed"]),
                device=self.device,
                gradient_profile=str(self.training["dense_gradient_clip_profile"]),
                snapshot_epochs=set(self.snapshot_epochs),
            )
            self._replace_rows(condition, curves)
            snapshot_model = deepcopy(dense_source).cpu()
            _reset_dense_head(
                snapshot_model, seed=int(self.training["dense_different_task_head_seed"])
            )
            self._save_snapshot(condition, snapshot_model, snapshots)
            self._save(condition, model)
            self.persist_training_progress("dense_different_task_training_progress")
            del snapshots, snapshot_model
        return _offload_trained_model(model, execution_device=self.device)

    def _audit_different_task_initialization(
        self, dir_source: nn.Module, targets: Mapping[str, nn.Module]
    ) -> None:
        self.initialization_audit.update(
            {
                name: backbone_exact_copy_audit(dir_source, target)
                for name, target in targets.items()
            }
        )
        self.initialization_audit["dictionary_fixed_vs_trainable_initial_equality"] = (
            targets_exactly_equal_audit(
                targets["dir_dictionary_fixed"], targets["dir_dictionary_trainable"]
            )
        )
        self.initialization_audit["backbone_output_equivalence"] = (
            backbone_output_equivalence_audit(
                dir_source, targets, role=self.role, device=self.device, sample_count=8
            )
        )
        if not all(
            bool(record.get("passed", False))
            for record in self.initialization_audit.values()
            if isinstance(record, Mapping)
        ):
            raise RuntimeError(
                "DiR Dictionary-Fixed/Dictionary-Trainable initialization audit failed before long Target training"
            )

    def _different_task_ownership(
        self,
        condition: str,
        dir_source: nn.Module,
        target: nn.Module,
        *,
        loaded: bool,
        controller: ActiveDictionaryFreezeController | None = None,
    ) -> dict[str, Any]:
        if condition == "dir_dictionary_trainable":
            return {
                "passed": True,
                "condition_contract": "Source_full_backbone_start_phase_allowed_internal_facing_block_D_plus_included_head_D_trainable",
                "verification_mode": "dictionary_trainable_control_no_endpoint_equality_expected",
            }
        verification = (
            verify_active_dictionary_ownership(dir_source, target)
            if loaded
            else controller.finalize(target)  # type: ignore[union-attr]
        )
        if not verification.get("passed", False):
            if loaded:
                raise RuntimeError(
                    "DiR Dictionary-Fixed endpoint failed active D/scale verification"
                )
            self.record_warning(
                stage=f"{condition}_active_dictionary_final_verification",
                message="Dictionary-Fixed D/scale verification failed after final restore",
            )
        return {
            **verification,
            "condition_contract": "Source_full_backbone_start_active_D_and_D_owned_scales_fixed_Source_inactive_D_trainable_on_phase_allowed_internal_facing_block_D_plus_included_head_D",
        }

    def _should_load(self, condition: str) -> bool:
        return self.resume_from_checkpoints or condition in self.completed_conditions

    def _load(self, checkpoint_key: str, model: nn.Module) -> None:
        _load_model_checkpoint(
            self.endpoint_paths[checkpoint_key],
            model,
            expected_training_contract_sha256=self.training_contract_sha256,
            expected_identity=self.checkpoint_specs[checkpoint_key],
        )

    def _save(self, checkpoint_key: str, model: nn.Module) -> None:
        _save_model(
            self.endpoint_paths[checkpoint_key],
            model,
            metadata=_checkpoint_metadata_for_spec(
                self.checkpoint_metadata_base, self.checkpoint_specs[checkpoint_key]
            ),
        )

    def _save_snapshot(
        self,
        checkpoint_key: str,
        model: nn.Module,
        snapshots: Mapping[int, Mapping[str, torch.Tensor]],
    ) -> None:
        if self.snapshot_epoch not in snapshots:
            return
        model.load_state_dict(snapshots[self.snapshot_epoch], strict=True)
        snapshot_key = f"{checkpoint_key}_e{self.snapshot_epoch}"
        _save_model(
            Path(self.checkpoint_specs[snapshot_key]["path"]),
            model,
            metadata=_checkpoint_metadata_for_spec(
                self.checkpoint_metadata_base, self.checkpoint_specs[snapshot_key]
            ),
        )

    def _replace_rows(self, run_id: str, curves: Any) -> None:
        self.training_rows[:] = _replace_training_rows_for_run(
            self.training_rows, run_id, curves
        )


# Endpoint checkpoint stage
from .common import (
    ARTIFACT_COMPLETION_FILE,
    CHECKPOINT_PROVENANCE_FILE,
    CHECKPOINT_TRAINING_FILE,
    EXPECTED_CHECKPOINT_COUNT,
    IMPLEMENTATION_REVISION,
    OWNERSHIP_FILE,
    TRAINING_FILE,
    _canonical_json_sha256,
    _read_csv_rows,
    _sha256_file,
    _write_csv,
)
from .reporting import _support_commit_output_parity_summary
from .runtime import (
    _checkpoint_group_complete,
    _checkpoint_provenance_payload,
    _checkpoint_specs,
    _load_checkpoint_payload,
    _load_or_rebuild_checkpoint_provenance,
    _persist_training_csv_for_resume,
    _restore_training_csv_from_checkpoint,
    _training_rows_have_run,
    _validate_checkpoint_identity,
    _validate_checkpoint_inventory,
    _validate_existing_checkpoint_files,
    _write_checkpoint_provenance,
)


@dataclass
class TrainingStageResult:
    dir_source: nn.Module
    dir_targets: dict[str, nn.Module]
    models: dict[str, nn.Module]
    endpoint_paths: dict[str, Path]
    checkpoint_inventory_status: str
    resume_from_checkpoints: bool
    checkpoint_provenance: dict[str, Any]
    checkpoint_provenance_sha256: str
    checkpoint_provenance_rebuilt: bool
    checkpoint_training_csv_path: Path
    checkpoint_provenance_path: Path
    checkpoint_files: list[Path]
    training_csv_sha256: str
    ownership: dict[str, Any]
    support_commit_output_parity: dict[str, Any]
    initialization_audit: dict[str, Any]


@dataclass
class _TrainingRuntime:
    checkpoint_specs: dict[str, dict[str, Any]]
    endpoint_paths: dict[str, Path]
    checkpoint_inventory_status: str
    resume_from_checkpoints: bool
    completed_conditions: set[str]
    training_rows: list[dict[str, Any]]
    training_csv_path: Path
    checkpoint_training_csv_path: Path
    checkpoint_provenance_path: Path
    snapshot_epoch: int
    snapshot_epochs: set[int]


def run_training_stage(
    *,
    artifact_run_id: str,
    checkpoint_dir: Path,
    device: Any,
    effective_config_sha256: str,
    output_dir: Path,
    plan: Mapping[str, Any],
    post_training_warnings: list[dict[str, Any]],
    role: dict[str, Any],
    source_dictionary: dict[str, Any],
    source_record: Any,
    training: dict[str, Any],
    training_contract_sha256: str,
    write_progress: Callable[[Mapping[str, Any]], None],
    write_post_training_json: Callable[..., bool],
    write_post_training_progress: Callable[..., bool],
    record_post_training_warning: Callable[..., None],
) -> TrainingStageResult:
    """Train/load all seven endpoints, then freeze checkpoint provenance for measurement."""

    runtime = _prepare_training_runtime(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        plan=plan,
        source_record=source_record,
        training=training,
        training_contract_sha256=training_contract_sha256,
    )
    checkpoint_metadata_base = {
        "effective_config_sha256": effective_config_sha256,
        "training_contract_sha256": training_contract_sha256,
        "implementation_revision": IMPLEMENTATION_REVISION,
    }

    def persist_training_progress(stage: str) -> None:
        try:
            _persist_training_csv_for_resume(
                runtime.training_csv_path,
                runtime.checkpoint_training_csv_path,
                runtime.training_rows,
            )
        except Exception as error:
            record_post_training_warning(
                stage=stage,
                message=(
                    "condition-level training progress persistence failed; current run continues "
                    "but interruption may require retraining this condition"
                ),
                error=error,
            )

    matrix_runner = TrainingMatrixRunner(
        role=role,
        source_record=source_record,
        source_dictionary=source_dictionary,
        training=training,
        device=device,
        checkpoint_specs=runtime.checkpoint_specs,
        endpoint_paths=runtime.endpoint_paths,
        checkpoint_metadata_base=checkpoint_metadata_base,
        training_contract_sha256=training_contract_sha256,
        resume_from_checkpoints=runtime.resume_from_checkpoints,
        completed_conditions=runtime.completed_conditions,
        snapshot_epoch=runtime.snapshot_epoch,
        snapshot_epochs=runtime.snapshot_epochs,
        write_progress=write_progress,
        persist_training_progress=persist_training_progress,
        record_warning=record_post_training_warning,
        training_rows=runtime.training_rows,
    )
    (
        dir_source,
        dir_targets,
        dense_source,
        dense_same,
        dense_different,
    ) = matrix_runner.run()

    _validate_loaded_checkpoint_metadata(
        runtime.checkpoint_specs,
        training_contract_sha256=training_contract_sha256,
    )
    training_csv_sha256 = _persist_final_training_csv(
        runtime=runtime,
        record_warning=record_post_training_warning,
    )
    support_commit_output_parity = _support_commit_parity(
        runtime.training_rows,
        commit_epoch=int(training["dir_target_epochs"]),
        record_warning=record_post_training_warning,
    )
    (
        checkpoint_provenance,
        checkpoint_provenance_sha256,
        checkpoint_provenance_rebuilt,
    ) = _finalize_checkpoint_provenance(
        artifact_run_id=artifact_run_id,
        runtime=runtime,
        training_contract_sha256=training_contract_sha256,
        training_csv_sha256=training_csv_sha256,
        support_commit_output_parity=support_commit_output_parity,
        record_warning=record_post_training_warning,
    )

    ownership = matrix_runner.ownership
    initialization_audit = matrix_runner.initialization_audit
    _persist_ownership_report(
        output_dir=output_dir,
        ownership=ownership,
        initialization_audit=initialization_audit,
        write_json=write_post_training_json,
    )
    checkpoint_files, checkpoint_inventory_status = _final_checkpoint_inventory(
        checkpoint_dir=checkpoint_dir,
        checkpoint_specs=runtime.checkpoint_specs,
        record_warning=record_post_training_warning,
    )

    models = {
        "dir_source": dir_source,
        "dir_same_task": dir_targets["dir_same_task"],
        "dir_dictionary_fixed": dir_targets["dir_dictionary_fixed"],
        "dir_dictionary_trainable": dir_targets["dir_dictionary_trainable"],
        "dense_source": dense_source,
        "dense_same_task": dense_same,
        "dense_different_task": dense_different,
    }
    write_post_training_progress(
        {
            "stage": "post_training_measurement",
            "status": "running",
            "checkpoint_inventory_status": checkpoint_inventory_status,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_count": sum(path.is_file() for path in checkpoint_files),
            "post_training_warning_count": len(post_training_warnings),
        },
        stage="post_training_measurement_progress",
    )

    return TrainingStageResult(
        dir_source=dir_source,
        dir_targets=dir_targets,
        models=models,
        endpoint_paths=runtime.endpoint_paths,
        checkpoint_inventory_status=checkpoint_inventory_status,
        resume_from_checkpoints=runtime.resume_from_checkpoints,
        checkpoint_provenance=checkpoint_provenance,
        checkpoint_provenance_sha256=checkpoint_provenance_sha256,
        checkpoint_provenance_rebuilt=checkpoint_provenance_rebuilt,
        checkpoint_training_csv_path=runtime.checkpoint_training_csv_path,
        checkpoint_provenance_path=runtime.checkpoint_provenance_path,
        checkpoint_files=checkpoint_files,
        training_csv_sha256=training_csv_sha256,
        ownership=ownership,
        support_commit_output_parity=support_commit_output_parity,
        initialization_audit=initialization_audit,
    )


def _prepare_training_runtime(
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    plan: Mapping[str, Any],
    source_record: Any,
    training: Mapping[str, Any],
    training_contract_sha256: str,
) -> _TrainingRuntime:
    dir_target_epoch = int(training["dir_target_epochs"])
    dense_target_epoch = int(training["dense_target_epochs"])
    checkpoint_epochs = {int(value) for value in training["checkpoint_epochs"]}
    snapshot_epochs = checkpoint_epochs - {dir_target_epoch}
    if snapshot_epochs != {20} or dense_target_epoch != dir_target_epoch:
        raise ValueError(
            "Final DiR matrix requires one e20 snapshot and common e52 Target endpoints"
        )

    checkpoint_specs = _checkpoint_specs(
        checkpoint_dir, training, dir_model_family=source_record.model_family
    )
    endpoint_keys = (
        "dir_source",
        "dir_same_task",
        "dir_dictionary_fixed",
        "dir_dictionary_trainable",
        "dense_source",
        "dense_same_task",
        "dense_different_task",
    )
    endpoint_paths = {key: Path(checkpoint_specs[key]["path"]) for key in endpoint_keys}
    inventory_status = _validate_checkpoint_inventory(checkpoint_dir, checkpoint_specs)
    resume_enabled = bool(
        plan.get("execution", {}).get("resume_measurements_from_checkpoints", True)
    )
    if inventory_status in {"complete", "partial"} and not resume_enabled:
        raise RuntimeError("DiR checkpoints exist but resume is disabled; refusing overwrite")
    if inventory_status != "fresh":
        _validate_existing_checkpoint_files(
            checkpoint_specs, training_contract_sha256=training_contract_sha256
        )
    resume_from_checkpoints = bool(resume_enabled and inventory_status == "complete")

    training_csv_path = output_dir / TRAINING_FILE
    checkpoint_training_csv_path = checkpoint_dir / CHECKPOINT_TRAINING_FILE
    checkpoint_provenance_path = checkpoint_dir / CHECKPOINT_PROVENANCE_FILE
    if (checkpoint_dir / ARTIFACT_COMPLETION_FILE).is_file():
        raise RuntimeError("Completed checkpoint inventory cannot be resumed as an active run")

    checkpoint_rows = (
        _read_csv_rows(checkpoint_training_csv_path)
        if checkpoint_training_csv_path.is_file()
        else []
    )
    if inventory_status == "complete" and not checkpoint_rows:
        raise RuntimeError("Complete checkpoint inventory requires training_metrics.csv")

    training_rows: list[dict[str, Any]] = []
    if resume_from_checkpoints:
        _restore_training_csv_from_checkpoint(
            checkpoint_training_csv_path, training_csv_path
        )
        training_rows = _read_csv_rows(training_csv_path)

    completed_conditions: set[str] = set()
    run_ids = (
        "dir_source_a",
        "dir_same_task",
        "dir_dictionary_fixed",
        "dir_dictionary_trainable",
        "dense_source_a",
        "dense_same_task",
        "dense_different_task",
    )
    if inventory_status == "fresh":
        checkpoint_training_csv_path.unlink(missing_ok=True)
        checkpoint_provenance_path.unlink(missing_ok=True)
    elif inventory_status == "partial":
        checkpoint_provenance_path.unlink(missing_ok=True)
        if checkpoint_training_csv_path.is_file():
            _restore_training_csv_from_checkpoint(
                checkpoint_training_csv_path, training_csv_path
            )
            training_rows = _read_csv_rows(training_csv_path)
        for condition in run_ids:
            if _checkpoint_group_complete(
                checkpoint_specs, condition
            ) and _training_rows_have_run(training_rows, condition):
                completed_conditions.add(condition)

    return _TrainingRuntime(
        checkpoint_specs=checkpoint_specs,
        endpoint_paths=endpoint_paths,
        checkpoint_inventory_status=inventory_status,
        resume_from_checkpoints=resume_from_checkpoints,
        completed_conditions=completed_conditions,
        training_rows=training_rows,
        training_csv_path=training_csv_path,
        checkpoint_training_csv_path=checkpoint_training_csv_path,
        checkpoint_provenance_path=checkpoint_provenance_path,
        snapshot_epoch=20,
        snapshot_epochs=snapshot_epochs,
    )


def _validate_loaded_checkpoint_metadata(
    checkpoint_specs: Mapping[str, Mapping[str, Any]], *, training_contract_sha256: str
) -> None:
    for spec in checkpoint_specs.values():
        path = Path(spec["path"])
        if not path.is_file():
            continue
        payload = _load_checkpoint_payload(path)
        metadata = dict(payload.get("metadata", {}) or {})
        _validate_checkpoint_identity(metadata, spec, path=path)
        if str(metadata.get("training_contract_sha256", "")) != training_contract_sha256:
            raise ValueError(f"DiR checkpoint training contract hash mismatch: {path.name}")


def _persist_final_training_csv(
    *, runtime: _TrainingRuntime, record_warning: Callable[..., None]
) -> str:
    try:
        return _persist_training_csv_for_resume(
            runtime.training_csv_path,
            runtime.checkpoint_training_csv_path,
            runtime.training_rows,
        )
    except Exception as error:
        record_warning(
            stage="checkpoint_training_csv_persistence",
            message="checkpoint-side training CSV persistence failed after training",
            error=error,
        )
        if not runtime.training_csv_path.is_file():
            try:
                _write_csv(runtime.training_csv_path, runtime.training_rows)
            except Exception as csv_error:
                record_warning(
                    stage="training_csv_fallback_persistence",
                    message="fallback training CSV persistence failed after training",
                    error=csv_error,
                )
        return (
            _sha256_file(runtime.training_csv_path)
            if runtime.training_csv_path.is_file()
            else ""
        )


def _support_commit_parity(
    training_rows: list[dict[str, Any]],
    *,
    commit_epoch: int,
    record_warning: Callable[..., None],
) -> dict[str, Any]:
    try:
        result = _support_commit_output_parity_summary(
            training_rows, commit_epoch=commit_epoch, expected_sample_count=128
        )
    except Exception as error:
        result = {
            "status": "warning_validation_failed",
            "passed": False,
            "invalid_reasons": [
                f"support_commit_output_parity_exception:{type(error).__name__}"
            ],
            "error": str(error),
        }
        record_warning(
            stage="support_commit_output_parity",
            message="support-commit parity validation raised after completed training",
            error=error,
        )
    if not result.get("passed", False):
        record_warning(
            stage="support_commit_output_parity",
            message=(
                "support-commit parity requires review after completed training; measurement continues"
            ),
        )
    return result


def _finalize_checkpoint_provenance(
    *,
    artifact_run_id: str,
    runtime: _TrainingRuntime,
    training_contract_sha256: str,
    training_csv_sha256: str,
    support_commit_output_parity: Mapping[str, Any],
    record_warning: Callable[..., None],
) -> tuple[dict[str, Any], str, bool]:
    parity_sha256 = _canonical_json_sha256(support_commit_output_parity)
    if runtime.resume_from_checkpoints:
        return _load_or_rebuild_checkpoint_provenance(
            runtime.checkpoint_provenance_path,
            checkpoint_specs=runtime.checkpoint_specs,
            current_artifact_run_id=artifact_run_id,
            training_contract_sha256=training_contract_sha256,
            training_csv_sha256=training_csv_sha256,
            support_commit_output_parity=support_commit_output_parity,
            support_commit_output_parity_sha256=parity_sha256,
        )

    provenance = _checkpoint_provenance_payload(
        checkpoint_specs=runtime.checkpoint_specs,
        training_run_id=artifact_run_id,
        provenance_origin="created_at_training",
        training_contract_sha256=training_contract_sha256,
        training_csv_sha256=training_csv_sha256,
        support_commit_output_parity=support_commit_output_parity,
        support_commit_output_parity_sha256=parity_sha256,
    )
    try:
        sha256 = _write_checkpoint_provenance(
            runtime.checkpoint_provenance_path, provenance
        )
    except Exception as error:
        sha256 = ""
        record_warning(
            stage="checkpoint_provenance_persistence",
            message="checkpoint provenance persistence failed after training",
            error=error,
        )
    return provenance, sha256, False


def _persist_ownership_report(
    *,
    output_dir: Path,
    ownership: Mapping[str, Any],
    initialization_audit: Mapping[str, Any],
    write_json: Callable[..., bool],
) -> None:
    payload = {
        **ownership,
        "initialization_audit": initialization_audit,
        "final_matrix": {
            "same_task": ["dir_same_task", "dense_same_task"],
            "different_task": [
                "dir_dictionary_fixed",
                "dir_dictionary_trainable",
                "dense_different_task",
            ],
            "different_task_dictionary_fixed_vs_trainable_shared_initialization": True,
            "classification_head_transfer": "none_for_different_task",
            "optimizer_state_transfer": "none",
        },
    }
    write_json(
        output_dir / OWNERSHIP_FILE,
        payload,
        stage="parameter_ownership_persistence",
        message="parameter-ownership report persistence failed after training",
    )


def _final_checkpoint_inventory(
    *,
    checkpoint_dir: Path,
    checkpoint_specs: Mapping[str, Mapping[str, Any]],
    record_warning: Callable[..., None],
) -> tuple[list[Path], str]:
    checkpoint_files = [Path(spec["path"]) for spec in checkpoint_specs.values()]
    try:
        status = _validate_checkpoint_inventory(checkpoint_dir, checkpoint_specs)
    except Exception as error:
        status = "warning_validation_failed"
        record_warning(
            stage="checkpoint_inventory_validation",
            message="checkpoint inventory validation failed after completed training",
            error=error,
        )
    if not (
        status == "complete"
        and len(checkpoint_files) == EXPECTED_CHECKPOINT_COUNT
        and all(path.is_file() for path in checkpoint_files)
    ):
        record_warning(
            stage="checkpoint_inventory",
            message="checkpoint inventory is incomplete after completed training",
        )
    return checkpoint_files, status
