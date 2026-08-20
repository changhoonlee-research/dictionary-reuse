"""DiR capture and intervention engine.

The normal model forward is untouched. This module executes an explicit eval-only
forward when DiR requests tensors or causal interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch import nn


INTERVENTION_POINTS = {
    "block_update",
    "post_o_attention_output",
    "post_w2_mlp_output",
    "block_output",
}


@dataclass(frozen=True)
class Intervention:
    block_index: int
    point: str
    replacement: torch.Tensor | None = None
    zero: bool = False

    def __post_init__(self) -> None:
        if int(self.block_index) < 0:
            raise ValueError("block_index must be non-negative")
        if str(self.point) not in INTERVENTION_POINTS:
            raise ValueError(f"Unsupported DiR intervention point: {self.point}")
        if bool(self.zero) == (self.replacement is not None):
            raise ValueError("Specify exactly one of zero=True or replacement=<tensor>.")


def _replacement_for(
    interventions: Mapping[tuple[int, str], Intervention],
    *,
    block_index: int,
    point: str,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    item = interventions.get((int(block_index), str(point)))
    if item is None:
        return None
    if item.zero:
        return torch.zeros_like(reference)
    assert item.replacement is not None
    replacement = item.replacement.to(device=reference.device, dtype=reference.dtype)
    if tuple(replacement.shape) != tuple(reference.shape):
        raise ValueError(
            f"DiR intervention shape mismatch at block {block_index} {point}: "
            f"expected {tuple(reference.shape)}, got {tuple(replacement.shape)}"
        )
    return replacement


def _patch_embedding_sequence(
    model: nn.Module,
    input_images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    patch_tokens = model.patch_embedding(input_images)
    batch_size = int(patch_tokens.shape[0])
    class_token_parameter = model.class_token() if isinstance(model.class_token, nn.Module) else model.class_token
    position_embedding_parameter = (
        model.position_embedding() if isinstance(model.position_embedding, nn.Module) else model.position_embedding
    )
    class_token = class_token_parameter.expand(batch_size, -1, -1)
    token_sequence = torch.cat([class_token, patch_tokens], dim=1)
    token_sequence = token_sequence + position_embedding_parameter
    return token_sequence, {
        "patch_embedding_out": patch_tokens,
        "embedding_sequence_out": token_sequence,
    }


def _index_interventions(
    interventions: Sequence[Intervention] | None,
) -> dict[tuple[int, str], Intervention]:
    indexed: dict[tuple[int, str], Intervention] = {}
    for item in interventions or ():
        key = (int(item.block_index), str(item.point))
        if key in indexed:
            raise ValueError(f"Duplicate DiR intervention: {key}")
        indexed[key] = item
    return indexed


def _forward_from_token_sequence(
    model: nn.Module,
    token_sequence: torch.Tensor,
    *,
    start_block_index: int,
    requested: set[str] | None,
    indexed_interventions: Mapping[tuple[int, str], Intervention],
    initial_captures: MutableMapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    captures: dict[str, torch.Tensor] = dict(initial_captures or {})

    def save(name: str, value: torch.Tensor) -> None:
        if requested is None or name in requested:
            captures[name] = value

    depth = len(model.transformer_blocks)
    start = int(start_block_index)
    if start < 0 or start > depth:
        raise ValueError(f"Invalid DiR suffix start block: {start}")

    for block_index in range(start, depth):
        block = model.transformer_blocks[block_index]
        block_input = token_sequence
        save(f"block_{block_index:02d}_input", block_input)

        pre_attention_norm = block.first_layer_normalization(block_input)
        attention_output, attention_taps = block.multi_head_self_attention.forward_with_measurement_tensors(
            pre_attention_norm
        )
        patched_attention = _replacement_for(
            indexed_interventions,
            block_index=block_index,
            point="post_o_attention_output",
            reference=attention_output,
        )
        if patched_attention is not None:
            attention_output = patched_attention
        attention_contribution = attention_output
        residual_after_attention = block_input + attention_contribution

        pre_mlp_norm = block.second_layer_normalization(residual_after_attention)
        mlp_output, mlp_taps = block.feed_forward_network.forward_with_measurement_tensors(pre_mlp_norm)
        patched_mlp = _replacement_for(
            indexed_interventions,
            block_index=block_index,
            point="post_w2_mlp_output",
            reference=mlp_output,
        )
        if patched_mlp is not None:
            mlp_output = patched_mlp
        mlp_contribution = mlp_output
        block_output = residual_after_attention + mlp_contribution

        native_update = block_output - block_input
        patched_update = _replacement_for(
            indexed_interventions,
            block_index=block_index,
            point="block_update",
            reference=native_update,
        )
        if patched_update is not None:
            block_output = block_input + patched_update

        patched_block_output = _replacement_for(
            indexed_interventions,
            block_index=block_index,
            point="block_output",
            reference=block_output,
        )
        if patched_block_output is not None:
            block_output = patched_block_output

        token_sequence = block_output
        final_update = block_output - block_input

        save(f"block_{block_index:02d}_pre_attention_norm", pre_attention_norm)
        for tap_name, tap_value in attention_taps.items():
            if tap_name == "post_o_attention_output":
                tap_value = attention_output
            save(f"block_{block_index:02d}_{tap_name}", tap_value)
        save(f"block_{block_index:02d}_attention_residual_contribution", attention_contribution)
        save(f"block_{block_index:02d}_post_attention_residual", residual_after_attention)
        save(f"block_{block_index:02d}_pre_mlp_norm", pre_mlp_norm)
        for tap_name, tap_value in mlp_taps.items():
            if tap_name == "post_w2_mlp_output":
                tap_value = mlp_output
            save(f"block_{block_index:02d}_{tap_name}", tap_value)
        save(f"block_{block_index:02d}_mlp_residual_contribution", mlp_contribution)
        save(f"block_{block_index:02d}_output", block_output)
        save(f"block_{block_index:02d}_update", final_update)

    normalized = model.pre_classifier_normalization(token_sequence)
    class_token = normalized[:, 0]
    logits = model.classification_head(class_token)
    save("pre_classifier", normalized)
    save("final_cls", class_token)
    save("logits", logits)
    return logits, captures


def forward_with_capture_and_interventions(
    model: nn.Module,
    input_images: torch.Tensor,
    *,
    capture_points: Sequence[str] | None = None,
    interventions: Sequence[Intervention] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one deterministic DiR measurement forward without changing model state."""

    requested = None if capture_points is None else {str(value) for value in capture_points}
    indexed = _index_interventions(interventions)
    token_sequence, initial_taps = _patch_embedding_sequence(model, input_images)
    initial_captures = {
        name: value
        for name, value in initial_taps.items()
        if requested is None or name in requested
    }
    return _forward_from_token_sequence(
        model,
        token_sequence,
        start_block_index=0,
        requested=requested,
        indexed_interventions=indexed,
        initial_captures=initial_captures,
    )


def forward_from_block_input_with_interventions(
    model: nn.Module,
    block_input: torch.Tensor,
    *,
    start_block_index: int,
    capture_points: Sequence[str] | None = None,
    interventions: Sequence[Intervention] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the exact transformer suffix from a previously captured native block input.

    This is mathematically identical to a full forward for interventions at or after
    ``start_block_index`` but avoids recomputing the unchanged embedding and prefix.
    """

    requested = None if capture_points is None else {str(value) for value in capture_points}
    indexed = _index_interventions(interventions)
    if any(int(index) < int(start_block_index) for index, _point in indexed):
        raise ValueError("DiR suffix forward cannot apply an intervention before its start block")
    return _forward_from_token_sequence(
        model,
        block_input,
        start_block_index=int(start_block_index),
        requested=requested,
        indexed_interventions=indexed,
    )

def _measurement_transient_attributes(model: nn.Module) -> list[str]:
    transient_names: list[str] = []
    for module_name, module in model.named_modules():
        for attr in vars(module):
            if attr.startswith("_dir_") or attr == "_intervention_output_atom_mask":
                transient_names.append(f"{module_name}.{attr}")
    return sorted(transient_names)


def capture_model_runtime_signature(model: nn.Module) -> dict[str, Any]:
    """Cheap per-module mutation detector without copying model tensors to CPU.

    Tensor ``_version`` changes on in-place mutation. Shape/dtype/device and
    transient-attribute checks catch accidental replacement or leaked
    intervention state without copying full model tensors to CPU.
    """

    tensors: list[tuple[str, tuple[int, ...], str, str, int]] = []
    for name, tensor in sorted(model.state_dict(keep_vars=True).items()):
        tensors.append(
            (
                name,
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                int(getattr(tensor, "_version", -1)),
            )
        )
    return {
        "tensors": tensors,
        "training": bool(model.training),
        "measurement_transient_attributes": _measurement_transient_attributes(model),
    }


