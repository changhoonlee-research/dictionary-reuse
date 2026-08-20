"""Dictionary reuse and hard-freeze enforcement for final DiR experiments.

Same-task reuse may include classification-head dictionary D/scale while keeping
Target C/route/support fresh. Different-task reuse excludes the classification
head so DiR and Dense both receive task-specific fresh heads. Source-active D
coordinates and D-owned scales can be copied/frozen without freezing C. In the
``Dictionary-Fixed`` condition, "fixed" therefore means Source-active D slices
(and the corresponding D-owned scales) are anchored exactly; inactive D atom
slices remain eligible for Target training only where the active phase permits D
updates. In the release ``internal_only`` phase this means the internal-facing D
coordinates (and classification-head D when that head is included), not every
residual-facing or endpoint D coordinate.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import torch
from torch import nn

from .model.dictionary_operator import iter_dictionary_layers

_BLOCK_PATTERN = re.compile(r"^transformer_blocks\.(\d+)(\..*)$")
_CLASSIFICATION_HEAD_PREFIX = "classification_head"


def _is_classification_head_name(name: str) -> bool:
    value = str(name)
    return value == _CLASSIFICATION_HEAD_PREFIX or value.startswith(
        _CLASSIFICATION_HEAD_PREFIX + "."
    )


def _backbone_dictionary_layers(model: nn.Module) -> dict[str, Any]:
    return {
        name: layer
        for name, layer in iter_dictionary_layers(model)
        if not _is_classification_head_name(name)
    }


def _dictionary_layers(
    model: nn.Module, *, include_classification_head: bool = False
) -> dict[str, Any]:
    if include_classification_head:
        return dict(iter_dictionary_layers(model))
    return _backbone_dictionary_layers(model)


def endpoint_active_masks(
    model: nn.Module, *, include_classification_head: bool = False
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for layer_name, layer in iter_dictionary_layers(model):
        if not include_classification_head and _is_classification_head_name(layer_name):
            continue
        method = getattr(layer, "_actual_forward_support_mask_for_metrics", None)
        if callable(method):
            mask = method(device=layer.coefficient_magnitude.device).detach().bool()
        else:
            fixed = getattr(layer, "forward_routed_fixed_support_mask", None)
            if isinstance(fixed, torch.Tensor) and int(fixed.numel()) == int(layer.atom_count):
                mask = fixed.detach().bool()
            else:
                mask = layer.coefficient_commit_mask.detach().bool()
        if int(mask.numel()) != int(layer.atom_count) or not bool(mask.any()):
            raise RuntimeError(f"Invalid DiR source active support for {layer_name}")
        masks[layer_name] = mask.cpu().clone()
    if not masks:
        raise RuntimeError("DiR backbone transplant requires dictionary layers.")
    return masks


def _unique_dictionary_scales(
    model: nn.Module, *, include_classification_head: bool = False
) -> list[tuple[str, torch.Tensor]]:
    """Return learned D-owned functional scales, excluding Target bookkeeping."""

    seen: set[int] = set()
    values: list[tuple[str, torch.Tensor]] = []
    for layer_name, layer in iter_dictionary_layers(model):
        if not include_classification_head and _is_classification_head_name(layer_name):
            continue
        tensor = getattr(layer, "dictionary_log_scale", None)
        if isinstance(tensor, torch.Tensor) and id(tensor) not in seen:
            seen.add(id(tensor))
            values.append((f"{layer_name}.dictionary_log_scale", tensor))
    for module_name, module in model.named_modules():
        if not include_classification_head and _is_classification_head_name(module_name):
            continue
        for attr in ("dictionary_qk_log_scale", "dictionary_vo_log_scale"):
            tensor = getattr(module, attr, None)
            if isinstance(tensor, torch.Tensor) and id(tensor) not in seen:
                seen.add(id(tensor))
                values.append((f"{module_name}.{attr}", tensor))
    return values


def _exact_tensor_match(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    if tuple(reference.shape) != tuple(candidate.shape) or reference.dtype != candidate.dtype:
        return False
    candidate_on_reference = candidate.detach().to(
        device=reference.device, dtype=reference.dtype
    )
    return bool(torch.equal(reference.detach(), candidate_on_reference))


def _validate_block_mapping(mapping: Mapping[int, int] | None) -> dict[int, int]:
    if mapping is None:
        return {}
    normalized = {int(target): int(source) for target, source in mapping.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("DiR block mapping must be one-to-one")
    return normalized


def _source_name_for_target(name: str, block_mapping: Mapping[int, int]) -> str:
    match = _BLOCK_PATTERN.match(str(name))
    if match is None:
        return str(name)
    target_index = int(match.group(1))
    source_index = int(block_mapping.get(target_index, target_index))
    return f"transformer_blocks.{source_index}{match.group(2)}"


def _mapped_active_masks(
    source_model: nn.Module,
    target_model: nn.Module,
    *,
    block_mapping: Mapping[int, int] | None = None,
    include_classification_head: bool = False,
) -> dict[str, torch.Tensor]:
    mapping = _validate_block_mapping(block_mapping)
    source_masks = endpoint_active_masks(
        source_model, include_classification_head=include_classification_head
    )
    source_layers = _dictionary_layers(
        source_model, include_classification_head=include_classification_head
    )
    target_layers = _dictionary_layers(
        target_model, include_classification_head=include_classification_head
    )
    if set(source_layers) != set(target_layers):
        raise ValueError("DiR backbone dictionary layer names differ between Source and Target")
    output: dict[str, torch.Tensor] = {}
    for target_name in target_layers:
        source_name = _source_name_for_target(target_name, mapping)
        if source_name not in source_masks:
            raise ValueError(f"DiR mapped Source layer missing: {source_name}")
        output[target_name] = source_masks[source_name].clone()
    return output


def mapped_endpoint_active_masks(
    source_model: nn.Module,
    target_model: nn.Module,
    *,
    block_mapping: Mapping[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Return Source endpoint masks keyed in Target layer coordinates.

    Correct transfer uses identity coordinates. Shuffled transfer remaps each
    Target block to the Source block that supplied its D+C+scale bundle.
    """

    return _mapped_active_masks(
        source_model, target_model, block_mapping=block_mapping, include_classification_head=False
    )


def verify_active_dictionary_ownership(
    source_model: nn.Module,
    target_model: nn.Module,
    *,
    block_mapping: Mapping[int, int] | None = None,
    include_classification_head: bool = False,
) -> dict[str, Any]:
    """Verify the D/scales required to remain frozen at a condition boundary."""

    mapping = _validate_block_mapping(block_mapping)
    source_layers = _dictionary_layers(
        source_model, include_classification_head=include_classification_head
    )
    target_layers = _dictionary_layers(
        target_model, include_classification_head=include_classification_head
    )
    if set(source_layers) != set(target_layers):
        return {
            "passed": False,
            "reason": "backbone_dictionary_layer_names_mismatch",
            "source_layer_count": len(source_layers),
            "target_layer_count": len(target_layers),
        }

    mapped_masks = _mapped_active_masks(
        source_model,
        target_model,
        block_mapping=mapping,
        include_classification_head=include_classification_head,
    )
    mismatches: list[str] = []
    for target_name, active_cpu in mapped_masks.items():
        source_name = _source_name_for_target(target_name, mapping)
        source_layer = source_layers[source_name]
        target_layer = target_layers[target_name]
        if tuple(source_layer.row_atoms.shape) != tuple(target_layer.row_atoms.shape):
            mismatches.append(f"{target_name}.row_shape")
            continue
        if tuple(source_layer.col_atoms.shape) != tuple(target_layer.col_atoms.shape):
            mismatches.append(f"{target_name}.col_shape")
            continue
        source_active = active_cpu.to(source_layer.row_atoms.device)
        target_active = active_cpu.to(target_layer.row_atoms.device)
        if not _exact_tensor_match(
            source_layer.row_atoms.detach()[..., source_active],
            target_layer.row_atoms.detach()[..., target_active],
        ):
            mismatches.append(f"{target_name}.row_active")
        if not _exact_tensor_match(
            source_layer.col_atoms.detach()[..., source_active],
            target_layer.col_atoms.detach()[..., target_active],
        ):
            mismatches.append(f"{target_name}.col_active")

    source_scales = dict(_unique_dictionary_scales(source_model, include_classification_head=include_classification_head))
    target_scales = dict(_unique_dictionary_scales(target_model, include_classification_head=include_classification_head))
    for target_name, target_scale in target_scales.items():
        source_name = _source_name_for_target(target_name, mapping)
        if source_name not in source_scales:
            mismatches.append(f"{target_name}.source_scale_missing")
            continue
        if not _exact_tensor_match(source_scales[source_name], target_scale):
            mismatches.append(target_name)

    return {
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "active_atom_count_total": int(sum(int(mask.sum()) for mask in mapped_masks.values())),
        "dictionary_layer_count": len(mapped_masks),
        "classification_head_transferred": False,
        "classification_head_dictionary_D_scale_reused": bool(include_classification_head),
        "block_mapping_target_to_source": {
            str(key): int(value) for key, value in sorted(mapping.items())
        },
        "verification_mode": (
            "exact_frozen_dictionary_D_and_D_owned_scale_comparison_including_head_dictionary"
            if include_classification_head
            else "exact_frozen_backbone_D_and_D_owned_scale_comparison"
        ),
    }


class ActiveDictionaryFreezeController:
    """Keep selected Source-active D and D-owned scales immutable.

    The controller always reasserts exact Source values for Source-active D slices
    and D-owned scales. It deliberately does *not* freeze Source-inactive D where
    the active training phase permits D updates. Under the release ``internal_only``
    phase, trainability is limited to internal-facing D coordinates (plus head D when
    included); residual-facing block sides and other residual endpoints remain fixed
    by phase selection. Active C
    is copied only when ``copy_active_coefficients=True`` and is never frozen. This
    supports both final-paper contracts: same-task reuses only Source-active D/scale
    with fresh C, while different-task Dictionary-Fixed starts from the exact Source
    full backbone including C. Route/support dynamics remain under the training
    stage. ``block_mapping`` is retained for mapped follow-up controls.
    """

    def __init__(
        self,
        target_model: nn.Module,
        source_model: nn.Module,
        *,
        block_mapping: Mapping[int, int] | None = None,
        include_classification_head: bool = False,
        copy_active_coefficients: bool = True,
    ) -> None:
        self.block_mapping = _validate_block_mapping(block_mapping)
        self.include_classification_head = bool(include_classification_head)
        self.copy_active_coefficients = bool(copy_active_coefficients)
        source_layers = _dictionary_layers(
            source_model, include_classification_head=self.include_classification_head
        )
        target_layers = _dictionary_layers(
            target_model, include_classification_head=self.include_classification_head
        )
        if set(source_layers) != set(target_layers):
            raise ValueError("DiR transplant requires matching backbone dictionary layer names.")
        self.active_masks = _mapped_active_masks(
            source_model,
            target_model,
            block_mapping=self.block_mapping,
            include_classification_head=self.include_classification_head,
        )
        self.snapshots: dict[str, dict[str, torch.Tensor]] = {}
        self.scale_snapshots: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        self._gradient_hook_handles: list[Any] = []
        coefficient_mismatches: list[str] = []

        with torch.no_grad():
            for target_name, target_layer in target_layers.items():
                source_name = _source_name_for_target(target_name, self.block_mapping)
                source_layer = source_layers[source_name]
                active = self.active_masks[target_name].to(
                    device=target_layer.row_atoms.device
                )
                source_active = active.to(source_layer.row_atoms.device)
                if tuple(source_layer.row_atoms.shape) != tuple(target_layer.row_atoms.shape):
                    raise ValueError(f"DiR row atom shape mismatch for {target_name}")
                if tuple(source_layer.col_atoms.shape) != tuple(target_layer.col_atoms.shape):
                    raise ValueError(f"DiR col atom shape mismatch for {target_name}")
                if tuple(source_layer.coefficient_magnitude.shape) != tuple(
                    target_layer.coefficient_magnitude.shape
                ):
                    raise ValueError(f"DiR C shape mismatch for {target_name}")

                row = source_layer.row_atoms.detach()[..., source_active].to(
                    device=target_layer.row_atoms.device,
                    dtype=target_layer.row_atoms.dtype,
                ).clone()
                col = source_layer.col_atoms.detach()[..., source_active].to(
                    device=target_layer.col_atoms.device,
                    dtype=target_layer.col_atoms.dtype,
                ).clone()
                target_layer.row_atoms[..., active] = row
                target_layer.col_atoms[..., active] = col
                if self.copy_active_coefficients:
                    coefficient = source_layer.coefficient_magnitude.detach()[source_active].to(
                        device=target_layer.coefficient_magnitude.device,
                        dtype=target_layer.coefficient_magnitude.dtype,
                    ).clone()
                    target_layer.coefficient_magnitude[active] = coefficient
                    if not torch.equal(
                        target_layer.coefficient_magnitude.detach()[active], coefficient
                    ):
                        coefficient_mismatches.append(target_name)
                self.snapshots[target_name] = {
                    "active": active.detach().clone(),
                    "row": row,
                    "col": col,
                    "source_layer_name": source_name,
                }

                for parameter in (target_layer.row_atoms, target_layer.col_atoms):
                    def mask_gradient(
                        gradient: torch.Tensor,
                        *,
                        frozen_mask: torch.Tensor = active,
                    ) -> torch.Tensor:
                        if frozen_mask.device != gradient.device:
                            raise RuntimeError(
                                "DiR freeze controller was created before final model device placement"
                            )
                        broadcast_shape = [1] * gradient.ndim
                        broadcast_shape[-1] = int(frozen_mask.numel())
                        return gradient.masked_fill(
                            frozen_mask.reshape(broadcast_shape), 0
                        )

                    self._gradient_hook_handles.append(
                        parameter.register_hook(mask_gradient)
                    )

            source_scales = dict(_unique_dictionary_scales(source_model, include_classification_head=self.include_classification_head))
            for target_scale_name, target_scale in _unique_dictionary_scales(target_model, include_classification_head=self.include_classification_head):
                source_scale_name = _source_name_for_target(
                    target_scale_name, self.block_mapping
                )
                if source_scale_name not in source_scales:
                    raise ValueError(
                        f"DiR mapped Source D-owned scale missing: {source_scale_name}"
                    )
                source_scale = source_scales[source_scale_name]
                if tuple(source_scale.shape) != tuple(target_scale.shape):
                    raise ValueError(
                        f"DiR D-owned scale shape mismatch: {target_scale_name}"
                    )
                snapshot = source_scale.detach().to(
                    device=target_scale.device, dtype=target_scale.dtype
                ).clone()
                target_scale.copy_(snapshot)
                self.scale_snapshots.append(
                    (target_scale_name, target_scale, snapshot)
                )
                if isinstance(target_scale, nn.Parameter):
                    target_scale._transplanted_dictionary_scale_hard_frozen = True
                    target_scale.requires_grad_(False)

        self.initial_coefficient_copy_passed = not coefficient_mismatches
        self.initial_coefficient_copy_mismatches = coefficient_mismatches
        if not self.initial_coefficient_copy_passed:
            self._release_gradient_hooks()
            raise RuntimeError("DiR active-C exact copy failed before training.")
        if not self._frozen_state_matches_snapshots(target_model):
            self._release_gradient_hooks()
            raise RuntimeError("DiR active-D exact copy failed before training.")

    def _frozen_state_matches_snapshots(self, target_model: nn.Module) -> bool:
        target_layers = _dictionary_layers(target_model, include_classification_head=self.include_classification_head)
        for layer_name, snapshot in self.snapshots.items():
            layer = target_layers[layer_name]
            active = snapshot["active"]
            if active.device != layer.row_atoms.device:
                return False
            if not torch.equal(layer.row_atoms.detach()[..., active], snapshot["row"]):
                return False
            if not torch.equal(layer.col_atoms.detach()[..., active], snapshot["col"]):
                return False
        for _name, tensor, snapshot in self.scale_snapshots:
            if tensor.device != snapshot.device or tensor.dtype != snapshot.dtype:
                return False
            if not torch.equal(tensor.detach(), snapshot):
                return False
        return True

    @torch.no_grad()
    def _zero_optimizer_state(
        self, optimizer: torch.optim.Optimizer, target_model: nn.Module
    ) -> None:
        target_layers = _dictionary_layers(target_model, include_classification_head=self.include_classification_head)
        for layer_name, snapshot in self.snapshots.items():
            active = snapshot["active"]
            layer = target_layers[layer_name]
            for parameter in (layer.row_atoms, layer.col_atoms):
                state = optimizer.state.get(parameter, {})
                for value in state.values():
                    if not torch.is_tensor(value) or tuple(value.shape) != tuple(parameter.shape):
                        continue
                    if active.device != value.device:
                        raise RuntimeError(
                            "DiR optimizer state device differs from freeze snapshot device"
                        )
                    value[..., active] = 0
        for _name, parameter, _snapshot in self.scale_snapshots:
            if not isinstance(parameter, nn.Parameter):
                continue
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if torch.is_tensor(value) and tuple(value.shape) == tuple(parameter.shape):
                    value.zero_()

    @torch.no_grad()
    def restore(
        self,
        target_model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        target_layers = _dictionary_layers(target_model, include_classification_head=self.include_classification_head)
        for layer_name, snapshot in self.snapshots.items():
            layer = target_layers[layer_name]
            active = snapshot["active"]
            row_snapshot = snapshot["row"]
            col_snapshot = snapshot["col"]
            if (
                active.device != layer.row_atoms.device
                or row_snapshot.device != layer.row_atoms.device
                or col_snapshot.device != layer.col_atoms.device
            ):
                raise RuntimeError(
                    "DiR freeze controller must be created after the target reaches its final device"
                )
            if row_snapshot.dtype != layer.row_atoms.dtype or col_snapshot.dtype != layer.col_atoms.dtype:
                raise RuntimeError("DiR freeze snapshot dtype differs from target dictionary dtype")
            layer.row_atoms[..., active] = row_snapshot
            layer.col_atoms[..., active] = col_snapshot
        for _name, tensor, snapshot in self.scale_snapshots:
            if snapshot.device != tensor.device or snapshot.dtype != tensor.dtype:
                raise RuntimeError(
                    "DiR scale freeze controller must be created after final model device placement"
                )
            tensor.copy_(snapshot)
        if optimizer is not None:
            self._zero_optimizer_state(optimizer, target_model)

    def step_observer(self, **payload: Any) -> None:
        self.restore(payload["model"], payload["optimizer"])

    def post_epoch_observer(self, **payload: Any) -> None:
        self.restore(payload["model"], payload["optimizer"])

    def _release_gradient_hooks(self) -> None:
        handles = self._gradient_hook_handles
        self._gradient_hook_handles = []
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    def finalize(self, target_model: nn.Module) -> dict[str, Any]:
        """Reassert the freeze and verify only tensors required to stay frozen."""

        self.restore(target_model)
        passed = self._frozen_state_matches_snapshots(target_model)
        result = {
            "passed": passed,
            "final_idempotent_restore_applied": True,
            "initial_coefficient_copy_passed": self.initial_coefficient_copy_passed,
            "initial_coefficient_copy_mismatches": list(
                self.initial_coefficient_copy_mismatches
            ),
            "active_atom_count_total": int(
                sum(int(mask.sum()) for mask in self.active_masks.values())
            ),
            "dictionary_layer_count": len(self.active_masks),
            "classification_head_transferred": False,
            "classification_head_dictionary_D_scale_reused": bool(self.include_classification_head),
            "block_mapping_target_to_source": {
                str(key): int(value)
                for key, value in sorted(self.block_mapping.items())
            },
            "verification_mode": (
                "exact_frozen_dictionary_D_and_D_owned_scale_comparison_including_head_dictionary"
                if self.include_classification_head
                else "exact_frozen_backbone_D_and_D_owned_scale_comparison"
            ),
            "copied_fixed_tensor_contract": [
                "source_endpoint_active_dictionary_row_atoms",
                "source_endpoint_active_dictionary_col_atoms",
                "dictionary_D_owned_scales",
            ],
            "copied_trainable_tensor_contract": (
                ["source_endpoint_C_for_the_corresponding_source_active_atoms"]
                if self.copy_active_coefficients
                else []
            ),
            "coefficient_initialization_contract": (
                "source_active_C_copied_trainable"
                if self.copy_active_coefficients
                else "target_fresh_C_preserved"
            ),
            "explicitly_not_copied": [
                "entire_classification_head",
                "route_state",
                "support_state",
                "optimizer_state",
                "relative_coordinate_log_scale",
                "dictionary_qk_coordinate_log_scale",
                "dictionary_vo_coordinate_log_scale",
            ],
            "classification_head_contract": (
                "target_head_structure_and_C_remain_target_initialized_while_head_dictionary_D_scale_are_reused"
                if self.include_classification_head
                else "fully_target_initialized_different_task_head"
            ),
            "inactive_dictionary_slices": "target_initialized_and_trainable_not_hard_frozen",
            "target_coordinate_correction_buffers": "target_owned_bookkeeping",
        }
        self._release_gradient_hooks()
        return result
