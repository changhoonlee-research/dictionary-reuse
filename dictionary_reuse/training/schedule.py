"""Training phase policy, dictionary installation, support commit, and trainability control."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .schema import _DICTIONARY_PARAMETER_SUFFIXES, _is_dense_model_family
from ..model.basis import _basis_primitive_spec_from_config
from ..model.routing import _entmax_alpha_1d
from ..model.dictionary_operator import (
    SeparableDictionaryLinear,
    _block_index_from_layer_name,
    _resolve_block_basis_type,
    iter_dictionary_layers,
)


# --- Dictionary installation ------------------------------------------------
def apply_dictionary_to_ffn_layers(
    model: nn.Module,
    *,
    dictionary_mode: str,
    atom_count: int,
    low_atom_count: int,
    basis_type: str,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]] | None = None,
    basis_bank_seed: int | None = None,
    shared_basis_bank: bool = False,
    bias_policy: str = "zero_frozen",
    dictionary_config: dict[str, Any] | None = None,
) -> None:
    dictionary_config = dictionary_config or {}
    base_basis_seed = int(seed if basis_bank_seed is None else basis_bank_seed)
    coefficient_quantization_config = (
        dictionary_config.get("coefficient_quantization", {})
        if isinstance(dictionary_config.get("coefficient_quantization", {}), dict)
        else {}
    )

    def _coefficient_quantization_config_for_layer(
        *,
        block_index: int,
        layer_seed_offset: int,
    ) -> dict[str, Any]:
        return _coefficient_quantization_config_for_dictionary_layer(
            coefficient_quantization_config,
            seed=int(seed) + int(block_index) * 1000,
            layer_seed_offset=int(layer_seed_offset),
        )

    for block_index, transformer_block in enumerate(model.transformer_blocks):
        ffn = transformer_block.feed_forward_network
        block_seed_offset = 0 if shared_basis_bank else block_index * 1000
        block_basis_type = _resolve_block_basis_type(dictionary_config, basis_type, int(block_index))
        block_primitive_spec = (
            list(primitive_spec)
            if primitive_spec is not None and block_basis_type == str(basis_type)
            else _basis_primitive_spec_from_config(dictionary_config, block_basis_type, atom_count, low_atom_count)
        )
        first_linear = ffn.first_linear_layer
        ffn.first_linear_layer = _make_dictionary_linear(
            original_linear=first_linear,
            in_features=int(first_linear.in_features),
            out_features=int(first_linear.out_features),
            bias=first_linear.bias is not None,
            atom_count=atom_count,
            low_atom_count=low_atom_count,
            basis_type=block_basis_type,
            dictionary_mode=dictionary_mode,
            seed=base_basis_seed + block_seed_offset + 11,
            bias_policy=bias_policy,
            primitive_spec=block_primitive_spec,
            dictionary_config=dictionary_config,
            coefficient_quantization_config=_coefficient_quantization_config_for_layer(block_index=block_index, layer_seed_offset=101),
            role="mlp_w1",
        )
        second_linear = ffn.second_linear_layer
        ffn.second_linear_layer = _make_dictionary_linear(
            original_linear=second_linear,
            in_features=int(second_linear.in_features),
            out_features=int(second_linear.out_features),
            bias=second_linear.bias is not None,
            atom_count=atom_count,
            low_atom_count=low_atom_count,
            basis_type=block_basis_type,
            dictionary_mode=dictionary_mode,
            seed=base_basis_seed + block_seed_offset + 29,
            bias_policy=bias_policy,
            primitive_spec=block_primitive_spec,
            dictionary_config=dictionary_config,
            coefficient_quantization_config=_coefficient_quantization_config_for_layer(block_index=block_index, layer_seed_offset=201),
            role="mlp_w2",
        )

def _apply_block_dictionary_to_projection(
    *,
    block_index: int,
    layer_seed_offset: int,
    original_linear: nn.Linear | None,
    in_features: int,
    out_features: int,
    role: str,
    dictionary_mode: str,
    atom_count: int,
    low_atom_count: int,
    basis_type: str,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]] | None,
    basis_bank_seed: int | None,
    shared_basis_bank: bool,
    bias_policy: str,
    dictionary_config: Mapping[str, Any],
    coefficient_quantization_config: Mapping[str, Any],
) -> SeparableDictionaryLinear:
    base_basis_seed = int(seed if basis_bank_seed is None else basis_bank_seed)
    block_seed_offset = 0 if shared_basis_bank else int(block_index) * 1000
    block_basis_type = _resolve_block_basis_type(dict(dictionary_config), basis_type, int(block_index))
    block_primitive_spec = (
        list(primitive_spec)
        if primitive_spec is not None and block_basis_type == str(basis_type)
        else _basis_primitive_spec_from_config(dict(dictionary_config), block_basis_type, atom_count, low_atom_count)
    )
    return _make_dictionary_linear(
        original_linear=original_linear,
        in_features=int(in_features),
        out_features=int(out_features),
        bias=bool(original_linear.bias is not None) if original_linear is not None else False,
        atom_count=atom_count,
        low_atom_count=low_atom_count,
        basis_type=block_basis_type,
        dictionary_mode=dictionary_mode,
        seed=base_basis_seed + block_seed_offset + int(layer_seed_offset),
        bias_policy=bias_policy,
        primitive_spec=block_primitive_spec,
        dictionary_config=dictionary_config,
        coefficient_quantization_config=_coefficient_quantization_config_for_dictionary_layer(coefficient_quantization_config, seed=int(seed) + int(block_index) * 1000, layer_seed_offset=int(layer_seed_offset)),
        role=role,
    )

def apply_dictionary_to_attention_layers(
    model: nn.Module,
    *,
    dictionary_mode: str,
    atom_count: int,
    low_atom_count: int,
    basis_type: str,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]] | None = None,
    basis_bank_seed: int | None = None,
    shared_basis_bank: bool = False,
    bias_policy: str = "zero_frozen",
    dictionary_config: dict[str, Any] | None = None,
) -> None:
    dictionary_config = dictionary_config or {}
    coefficient_quantization_payload = dictionary_config.get(
        "coefficient_quantization", {}
    )
    coefficient_quantization_config = (
        coefficient_quantization_payload
        if isinstance(coefficient_quantization_payload, dict)
        else {}
    )
    common_projection_kwargs = {
        "dictionary_mode": dictionary_mode,
        "atom_count": atom_count,
        "low_atom_count": low_atom_count,
        "basis_type": basis_type,
        "seed": seed,
        "primitive_spec": primitive_spec,
        "basis_bank_seed": basis_bank_seed,
        "shared_basis_bank": shared_basis_bank,
        "bias_policy": bias_policy,
        "dictionary_config": dictionary_config,
        "coefficient_quantization_config": coefficient_quantization_config,
    }
    for block_index, transformer_block in enumerate(model.transformer_blocks):
        attention = transformer_block.multi_head_self_attention
        dim = int(attention.embedding_dimension)
        attention.query_projection = _apply_block_dictionary_to_projection(
            block_index=block_index,
            layer_seed_offset=301,
            original_linear=None,
            in_features=dim,
            out_features=dim,
            role="attention_q",
            **common_projection_kwargs,
        )
        attention.key_projection = _apply_block_dictionary_to_projection(
            block_index=block_index,
            layer_seed_offset=311,
            original_linear=None,
            in_features=dim,
            out_features=dim,
            role="attention_k",
            **common_projection_kwargs,
        )
        attention.value_projection = _apply_block_dictionary_to_projection(
            block_index=block_index,
            layer_seed_offset=321,
            original_linear=None,
            in_features=dim,
            out_features=dim,
            role="attention_v",
            **common_projection_kwargs,
        )
        output_linear = attention.output_projection
        attention.output_projection = _apply_block_dictionary_to_projection(
            block_index=block_index,
            layer_seed_offset=331,
            original_linear=output_linear,
            in_features=int(output_linear.in_features),
            out_features=int(output_linear.out_features),
            role="attention_o",
            **common_projection_kwargs,
        )
        if hasattr(attention, "dictionary_qk_log_scale") or hasattr(attention, "dictionary_vo_log_scale"):
            raise RuntimeError("attention dictionary scale groups were already installed")
        attention.register_parameter("dictionary_qk_log_scale", nn.Parameter(torch.zeros((), dtype=torch.float32)))
        attention.register_parameter("dictionary_vo_log_scale", nn.Parameter(torch.zeros((), dtype=torch.float32)))
        with torch.no_grad():
            qk_initial_coordinate = (
                attention.query_projection.relative_coordinate_log_scale.detach()
                + attention.key_projection.relative_coordinate_log_scale.detach()
            )
            vo_initial_coordinate = (
                attention.value_projection.relative_coordinate_log_scale.detach()
                + attention.output_projection.relative_coordinate_log_scale.detach()
            )
            for projection in (
                attention.query_projection,
                attention.key_projection,
                attention.value_projection,
                attention.output_projection,
            ):
                projection.relative_coordinate_log_scale.zero_()
        attention.register_buffer(
            "dictionary_qk_coordinate_log_scale", qk_initial_coordinate.clone(), persistent=True
        )
        attention.register_buffer(
            "dictionary_vo_coordinate_log_scale", vo_initial_coordinate.clone(), persistent=True
        )
        attention.query_key_value_projection = None

def apply_dictionary_to_patch_embedding(
    model: nn.Module,
    *,
    dictionary_mode: str,
    atom_count: int,
    low_atom_count: int,
    basis_type: str,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]] | None = None,
    basis_bank_seed: int | None = None,
    bias_policy: str = "zero_frozen",
    dictionary_config: dict[str, Any] | None = None,
) -> None:
    dictionary_config = dictionary_config or {}
    patch = model.patch_embedding
    base_basis_seed = int(seed if basis_bank_seed is None else basis_bank_seed)
    patch_basis_type = str(dictionary_config.get("patch_basis_type", dictionary_config.get("endpoint_basis_type", basis_type)))
    patch_primitive_spec = _basis_primitive_spec_from_config(dictionary_config, patch_basis_type, atom_count, low_atom_count) if primitive_spec is None or patch_basis_type != str(basis_type) else list(primitive_spec)
    in_features = int(patch.number_of_input_channels) * int(patch.patch_embedding_kernel_size) * int(patch.patch_embedding_kernel_size)
    coefficient_quantization_payload = dictionary_config.get(
        "coefficient_quantization", {}
    )
    coefficient_quantization_config = (
        coefficient_quantization_payload
        if isinstance(coefficient_quantization_payload, dict)
        else {}
    )
    projection = _make_dictionary_linear(
        original_linear=None,
        in_features=in_features,
        out_features=int(patch.embedding_dimension),
        bias=False,
        atom_count=atom_count,
        low_atom_count=low_atom_count,
        basis_type=patch_basis_type,
        dictionary_mode=dictionary_mode,
        seed=base_basis_seed + 7001,
        bias_policy=bias_policy,
        primitive_spec=patch_primitive_spec,
        dictionary_config=dictionary_config,
        coefficient_quantization_config=(
            _coefficient_quantization_config_for_dictionary_layer(
                coefficient_quantization_config,
                seed=int(seed),
                layer_seed_offset=7001,
            )
        ),
        role="patch",
    )
    model.patch_embedding = DictionaryPatchEmbedding(
        image_size=int(patch.image_size),
        patch_size=int(patch.patch_size),
        patch_embedding_kernel_size=int(patch.patch_embedding_kernel_size),
        patch_embedding_stride=int(patch.patch_embedding_stride),
        patch_embedding_padding=int(patch.patch_embedding_padding),
        number_of_input_channels=int(patch.number_of_input_channels),
        embedding_dimension=int(patch.embedding_dimension),
        projection=projection,
    )

def apply_dictionary_to_token_embeddings(
    model: nn.Module,
    *,
    dictionary_mode: str,
    atom_count: int,
    low_atom_count: int,
    basis_type: str,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]] | None = None,
    basis_bank_seed: int | None = None,
    bias_policy: str = "zero_frozen",
    dictionary_config: dict[str, Any] | None = None,
) -> None:
    dictionary_config = dictionary_config or {}
    base_basis_seed = int(seed if basis_bank_seed is None else basis_bank_seed)
    token_basis_type = str(dictionary_config.get("token_basis_type", dictionary_config.get("endpoint_basis_type", basis_type)))
    token_primitive_spec = _basis_primitive_spec_from_config(dictionary_config, token_basis_type, atom_count, low_atom_count) if primitive_spec is None or token_basis_type != str(basis_type) else list(primitive_spec)
    coefficient_quantization_config = dictionary_config.get("coefficient_quantization", {}) if isinstance(dictionary_config.get("coefficient_quantization", {}), dict) else {}
    class_coefficient_quantization_config = _coefficient_quantization_config_for_dictionary_layer(
        coefficient_quantization_config,
        seed=int(seed),
        layer_seed_offset=7101,
    )
    position_coefficient_quantization_config = _coefficient_quantization_config_for_dictionary_layer(
        coefficient_quantization_config,
        seed=int(seed),
        layer_seed_offset=7201,
    )
    class_shape = tuple(int(dim) for dim in model.class_token.shape)
    class_dict = _make_dictionary_linear(
        original_linear=None,
        in_features=1,
        out_features=int(math.prod(class_shape)),
        bias=False,
        atom_count=atom_count,
        low_atom_count=low_atom_count,
        basis_type=token_basis_type,
        dictionary_mode=dictionary_mode,
        seed=base_basis_seed + 7101,
        bias_policy=bias_policy,
        primitive_spec=token_primitive_spec,
        dictionary_config=dictionary_config,
        coefficient_quantization_config=class_coefficient_quantization_config,
        role="class_token",
    )
    position_shape = tuple(int(dim) for dim in model.position_embedding.shape)
    if len(position_shape) != 3 or int(position_shape[0]) != 1:
        raise ValueError(f"expected position_embedding shape [1, T, D], got {position_shape!r}")
    position_dict = _make_dictionary_linear(
        original_linear=None,
        in_features=1,
        out_features=int(math.prod(position_shape)),
        bias=False,
        atom_count=atom_count,
        low_atom_count=low_atom_count,
        basis_type=token_basis_type,
        dictionary_mode=dictionary_mode,
        seed=base_basis_seed + 7201,
        bias_policy=bias_policy,
        primitive_spec=token_primitive_spec,
        dictionary_config=dictionary_config,
        coefficient_quantization_config=position_coefficient_quantization_config,
        role="position_embedding",
    )
    position_module = TensorDictionaryParameter(tensor_shape=position_shape, dictionary=position_dict)
    delattr(model, "class_token")
    delattr(model, "position_embedding")
    model.class_token = TensorDictionaryParameter(tensor_shape=class_shape, dictionary=class_dict)
    model.position_embedding = PositionDictionaryParameter(position_module=position_module)

def apply_dictionary_to_classification_head(
    model: nn.Module,
    *,
    dictionary_mode: str,
    atom_count: int,
    low_atom_count: int,
    basis_type: str,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]] | None = None,
    basis_bank_seed: int | None = None,
    bias_policy: str = "zero_frozen",
    dictionary_config: dict[str, Any] | None = None,
) -> None:
    dictionary_config = dictionary_config or {}
    head = model.classification_head
    if not isinstance(head, nn.Linear):
        raise TypeError("classification_head must be nn.Linear before DiR replacement")
    base_basis_seed = int(seed if basis_bank_seed is None else basis_bank_seed)
    head_basis_type = str(dictionary_config.get("head_basis_type", dictionary_config.get("endpoint_basis_type", "dct_only_256")))
    head_primitive_spec = _basis_primitive_spec_from_config(dictionary_config, head_basis_type, atom_count, low_atom_count) if primitive_spec is None or head_basis_type != str(basis_type) else list(primitive_spec)
    coefficient_quantization_payload = dictionary_config.get(
        "coefficient_quantization", {}
    )
    coefficient_quantization_config = (
        coefficient_quantization_payload
        if isinstance(coefficient_quantization_payload, dict)
        else {}
    )
    model.classification_head = _make_dictionary_linear(
        original_linear=head,
        in_features=int(head.in_features),
        out_features=int(head.out_features),
        bias=head.bias is not None,
        atom_count=atom_count,
        low_atom_count=low_atom_count,
        basis_type=head_basis_type,
        dictionary_mode=dictionary_mode,
        seed=base_basis_seed + 7301,
        bias_policy=bias_policy,
        primitive_spec=head_primitive_spec,
        dictionary_config=dictionary_config,
        coefficient_quantization_config=(
            _coefficient_quantization_config_for_dictionary_layer(
                coefficient_quantization_config,
                seed=int(seed),
                layer_seed_offset=7301,
            )
        ),
        role="head",
    )


# --- Phase ownership and trainability ---------------------------------------
def _coefficient_parameter_id_set(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for _name, layer in iter_dictionary_layers(model):
        ids.add(id(layer.coefficient_magnitude))
    return ids

def _dictionary_scale_parameter_id_set(model: nn.Module) -> set[int]:
    return {id(tensor) for _name, tensor in _named_dictionary_scale_tensors(model) if isinstance(tensor, nn.Parameter)}

def _hard_frozen_dictionary_scale_parameter_id_set(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for _name, tensor in _named_dictionary_scale_tensors(model):
        if isinstance(tensor, nn.Parameter) and bool(getattr(tensor, "_transplanted_dictionary_scale_hard_frozen", False)):
            ids.add(id(tensor))
    return ids

def _learned_dictionary_atom_parameter_id_set(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for _name, layer in iter_dictionary_layers(model):
        ids.update({id(layer.row_atoms), id(layer.col_atoms)})
    return ids

def _hard_frozen_dictionary_atom_parameter_id_set(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for _name, layer in iter_dictionary_layers(model):
        if not bool(getattr(layer, "_transplanted_dictionary_atoms_hard_frozen", False)):
            continue
        ids.update({id(layer.row_atoms), id(layer.col_atoms)})
    return ids

def _is_light_backbone_parameter_name(name: str) -> bool:
    lowered = name.lower()
    return "normalization" in lowered or ".norm" in lowered or "layernorm" in lowered

def _phase_cycle_from_config(phase_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    config = phase_config or {}
    if not bool(config.get("enabled", False)):
        return []
    cycle = config.get("cycle", [])
    if not isinstance(cycle, list) or not cycle:
        raise ValueError("enabled phase schedule requires a non-empty cycle list")
    result: list[dict[str, Any]] = []
    for item in cycle:
        if not isinstance(item, dict):
            raise ValueError("phase schedule cycle entries must be JSON objects")
        steps = int(item.get("steps", 0))
        if steps <= 0:
            raise ValueError("phase schedule cycle steps must be positive")
        groups = [str(group) for group in item.get("groups", [])]
        if not groups:
            raise ValueError("phase schedule cycle entry requires at least one trainable group")
        unknown = [group for group in groups if group not in {"C", "D", "B"}]
        if unknown:
            raise ValueError(f"unknown phase schedule groups {unknown!r}")
        entry: dict[str, Any] = {"name": str(item.get("name", "+".join(groups))), "steps": steps, "groups": tuple(groups)}
        # Preserve the current per-phase parameter scope used by DiR training.
        scope_payload = item.get("parameter_scope")
        if isinstance(scope_payload, dict):
            entry["parameter_scope"] = deepcopy(scope_payload)
        result.append(entry)
    return result

def _phase_for_step(
    phase_cycle: Sequence[dict[str, Any]],
    step_index: int,
) -> tuple[dict[str, Any] | None, int]:
    if not phase_cycle:
        return None, 0
    cycle_length = sum(int(item["steps"]) for item in phase_cycle)
    if cycle_length <= 0:
        return None, 0
    position = int(step_index) % cycle_length
    cursor = 0
    for item in phase_cycle:
        cursor += int(item["steps"])
        if position < cursor:
            return item, position
    return phase_cycle[-1], position

def _phase_backbone_parameter_allowed(parameter_name: str, phase_config: dict[str, Any] | None) -> bool:
    """Return whether a non-DiR backbone parameter belongs to B for phase scheduling.

    Full DiR runs set ``backbone_scope=none`` so LayerNorm-style leftovers are
    frozen outside C/D and do not appear as a B bypass. Other profiles retain
    the configured light/full backbone behavior.
    """

    scope = str((phase_config or {}).get("backbone_scope", "light")).strip().lower()
    if scope in {"", "none", "off", "disabled", "zero"}:
        return False
    if scope in {"full", "all"}:
        return True
    if scope in {"light", "layernorm", "norm", "normalization"}:
        return _is_light_backbone_parameter_name(parameter_name)
    raise ValueError(f"Unknown phase_schedule backbone_scope={scope!r}")

def _phase_config_with_active_parameter_scope(
    phase_config: dict[str, Any] | None,
    phase: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach the current cycle-entry scope to a phase config copy.

    This keeps existing phase profiles unchanged while allowing scoped C/D
    updates over one full-DiR part at a time.
    """

    result = dict(phase_config or {})
    scope_payload = None
    if isinstance(phase, dict):
        if isinstance(phase.get("parameter_scope"), dict):
            scope_payload = phase.get("parameter_scope")
        elif isinstance(phase.get("dictionary_scope"), dict):
            scope_payload = phase.get("dictionary_scope")
    if isinstance(scope_payload, dict):
        result["active_parameter_scope"] = deepcopy(scope_payload)
    else:
        result.pop("active_parameter_scope", None)
    return result

def _active_phase_parameter_scope(phase_config: dict[str, Any] | None) -> dict[str, Any] | None:
    scope = (phase_config or {}).get("active_parameter_scope")
    return scope if isinstance(scope, dict) and scope else None

def _dictionary_layer_name_from_parameter_name(parameter_name: str) -> str:
    name = str(parameter_name)
    for suffix in _DICTIONARY_PARAMETER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name.rsplit(".", 1)[0] if "." in name else name

def _dictionary_atom_side_from_parameter_name(parameter_name: str) -> str | None:
    name = str(parameter_name)
    if name.endswith(".row_atoms"):
        return "row"
    if name.endswith(".col_atoms"):
        return "col"
    return None

def _phase_scope_atom_policy(scope: dict[str, Any] | None) -> str:
    if not isinstance(scope, dict):
        return "all"
    raw_value = scope.get("dictionary_atom_scope", scope.get("atom_scope", scope.get("dictionary_atom_side", "all")))
    value = str(raw_value).strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"", "all", "full", "none", "unrestricted", "any"}:
        return "all"
    if value in {"internal", "internal_only", "commonspace_internal_only", "common_space_internal_only", "residual_interface_frozen", "interface_frozen"}:
        return "internal_only"
    if value in {"residual", "residual_only", "residual_facing", "residual_facing_only", "interface_only"}:
        return "residual_facing_only"
    raise ValueError(f"unknown phase dictionary_atom_scope={raw_value!r}")

def _dictionary_layer_residual_interface(layer_name: str) -> str | None:
    name = str(layer_name)
    if ".multi_head_self_attention." in name:
        if any(token in name for token in ("query_projection", "key_projection", "value_projection")):
            return "residual_to_internal"
        if "output_projection" in name:
            return "internal_to_residual"
        return None
    if ".feed_forward_network." in name:
        if "first_linear_layer" in name:
            return "residual_to_internal"
        if "second_linear_layer" in name:
            return "internal_to_residual"
        return None
    if (
        name.startswith("patch_embedding.")
        or name.startswith("class_token.")
        or name.startswith("position_embedding.")
        or name == "classification_head"
        or name.startswith("classification_head.")
    ):
        return "endpoint_residual"
    return None

def _phase_scope_allows_dictionary_atom_parameter_name(
    parameter_name: str,
    phase_config: dict[str, Any] | None,
) -> bool:
    """Return whether a row/col Dictionary atom side is allowed by the active atom-side policy.

    ``dictionary_atom_scope=internal_only`` keeps residual-facing read/write bases
    fixed while allowing only the opposite, block-internal side of Q/K/V/O/W1/W2
    to train. Coefficients deliberately do not use this filter.
    """

    scope = _active_phase_parameter_scope(phase_config)
    policy = _phase_scope_atom_policy(scope)
    if policy == "all":
        return True

    side = _dictionary_atom_side_from_parameter_name(parameter_name)
    if side not in {"row", "col"}:
        return False

    layer_name = _dictionary_layer_name_from_parameter_name(parameter_name)
    interface = _dictionary_layer_residual_interface(layer_name)
    if interface == "residual_to_internal":
        residual_facing_side = "col"
    elif interface == "internal_to_residual":
        residual_facing_side = "row"
    else:
        # The classification head remains a normal D-owned Dictionary layer even
        # under the block-internal atom policy. Other residual endpoints have no
        # internal-facing atom side and therefore stay fixed here.
        if layer_name == "classification_head" or layer_name.startswith("classification_head."):
            return True
        return False

    if policy == "internal_only":
        return side != residual_facing_side
    if policy == "residual_facing_only":
        return side == residual_facing_side
    raise ValueError(f"unknown normalized dictionary atom policy={policy!r}")

def _scope_string_set(scope: dict[str, Any], key: str) -> set[str]:
    value = scope.get(key)
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    return {str(value).strip().lower()} if str(value).strip() else set()

def _scope_int_set(scope: dict[str, Any], key: str) -> set[int] | None:
    value = scope.get(key)
    if value is None:
        return None
    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else [value]
    result: set[int] = set()
    for item in values:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            raise ValueError(f"phase parameter scope {key!r} entries must be integers: {value!r}")
    return result

def _phase_scope_allows_layer_kind(layer_name: str, layer_kinds: set[str]) -> bool:
    if not layer_kinds:
        return True
    name = str(layer_name)
    is_attention = ".multi_head_self_attention." in name or name.endswith(".multi_head_self_attention")
    is_mlp = ".feed_forward_network." in name or name.endswith(".feed_forward_network")
    if is_attention:
        if {"attention", "attention_qkvo", "qkvo"} & layer_kinds:
            return True
        if "query_projection" in name:
            return bool({"query", "q", "attention_query", "attention_q"} & layer_kinds)
        if "key_projection" in name:
            return bool({"key", "k", "attention_key", "attention_k"} & layer_kinds)
        if "value_projection" in name:
            return bool({"value", "v", "attention_value", "attention_v"} & layer_kinds)
        if "output_projection" in name:
            return bool({"output", "o", "attention_output", "attention_o"} & layer_kinds)
        return False
    if is_mlp:
        if {"mlp", "ffn", "feed_forward", "feed_forward_network"} & layer_kinds:
            return True
        if "first_linear_layer" in name:
            return bool({"w1", "mlp_w1", "ffn_w1", "first_linear", "first_linear_layer"} & layer_kinds)
        if "second_linear_layer" in name:
            return bool({"w2", "mlp_w2", "ffn_w2", "second_linear", "second_linear_layer"} & layer_kinds)
        return False
    return False

def _phase_scope_allows_parameter_name(parameter_name: str, phase_config: dict[str, Any] | None) -> bool:
    """Return whether a C/D tensor is inside the current partwise phase scope."""

    scope = _active_phase_parameter_scope(phase_config)
    if scope is None:
        return True

    layer_name = _dictionary_layer_name_from_parameter_name(parameter_name)
    parts = _scope_string_set(scope, "parts") | _scope_string_set(scope, "part")
    layer_kinds = _scope_string_set(scope, "layer_kinds") | _scope_string_set(scope, "layer_kind")

    if layer_name.startswith("patch_embedding."):
        return (not parts and scope.get("blocks") is None) or "patch" in parts or bool(scope.get("include_patch", False))

    if layer_name.startswith("class_token."):
        return (
            (not parts and scope.get("blocks") is None)
            or "class_token" in parts
            or "token" in parts
            or "token_position" in parts
            or bool(scope.get("include_class_token", False))
        )

    if layer_name.startswith("position_embedding."):
        return (
            (not parts and scope.get("blocks") is None)
            or "position_embedding" in parts
            or "position" in parts
            or "token_position" in parts
            or bool(scope.get("include_position_embedding", False))
        )

    if layer_name == "classification_head" or layer_name.startswith("classification_head."):
        return (not parts and scope.get("blocks") is None) or "head" in parts or "classification_head" in parts or bool(scope.get("include_head", False))

    block_index = _block_index_from_layer_name(layer_name)
    if not isinstance(block_index, int):
        return False

    if parts and not ({"block", "blocks", "transformer_block", "transformer_blocks"} & parts):
        return False

    allowed_blocks = _scope_int_set(scope, "blocks")
    if allowed_blocks is not None and int(block_index) not in allowed_blocks:
        return False

    return _phase_scope_allows_layer_kind(layer_name, layer_kinds)

def _set_phase_trainability(
    model: nn.Module,
    *,
    active_groups: Sequence[str],
    model_family: str,
    profile: LearningRateProfile,
    phase_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Activate current C/D/B groups while preserving zero-lr hard freezes.

    C owns all coefficient tensors, including the DiR classification head.
    D owns all learned Dictionary atoms/scales, including the DiR head. B is the
    non-Dictionary backbone and, for dense models, the ordinary classification head.
    """

    active = {str(group) for group in active_groups}
    coefficient_ids = _coefficient_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    dictionary_scale_ids = _dictionary_scale_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    hard_frozen_dictionary_scale_ids = _hard_frozen_dictionary_scale_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    dictionary_atom_ids = _learned_dictionary_atom_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    hard_frozen_dictionary_atom_ids = _hard_frozen_dictionary_atom_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    dictionary_prefixes = _dictionary_parameter_prefixes(model) if not _is_dense_model_family(model_family) else tuple()
    head_ids = _classification_head_parameter_id_set(model)
    head_lr = profile.non_dictionary_lr if profile.head_lr is None else float(profile.head_lr)
    trainable_counts = {"C": 0, "D": 0, "B": 0}
    for name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        trainable = False
        if parameter_id in coefficient_ids:
            trainable = (
                "C" in active
                and _phase_scope_allows_parameter_name(name, phase_config)
                and float(profile.coefficient_lr) > 0.0
            )
            if trainable:
                trainable_counts["C"] += int(parameter.numel())
        elif parameter_id in dictionary_scale_ids:
            trainable = (
                parameter_id not in hard_frozen_dictionary_scale_ids
                and "D" in active
                and _phase_scope_allows_parameter_name(name, phase_config)
                and float(profile.dictionary_lr) > 0.0
            )
            if trainable:
                trainable_counts["D"] += int(parameter.numel())
        elif parameter_id in dictionary_atom_ids:
            trainable = (
                parameter_id not in hard_frozen_dictionary_atom_ids
                and "D" in active
                and _phase_scope_allows_parameter_name(name, phase_config)
                and _phase_scope_allows_dictionary_atom_parameter_name(name, phase_config)
                and float(profile.dictionary_lr) > 0.0
            )
            if trainable:
                trainable_counts["D"] += int(parameter.numel())
        elif parameter_id in head_ids and _is_dense_model_family(model_family):
            trainable = (
                "B" in active
                and _phase_scope_allows_parameter_name(name, phase_config)
                and float(head_lr) > 0.0
            )
            if trainable:
                trainable_counts["B"] += int(parameter.numel())
        elif any(name.startswith(prefix) for prefix in dictionary_prefixes):
            trainable = False
        else:
            allow_backbone_name = _phase_backbone_parameter_allowed(name, phase_config)
            trainable = "B" in active and float(profile.non_dictionary_lr) > 0.0 and allow_backbone_name
            if trainable:
                trainable_counts["B"] += int(parameter.numel())
        parameter.requires_grad_(bool(trainable))
    return {
        "phase_trainable_groups": "+".join(sorted(active)),
        "phase_trainable_C_param_count": trainable_counts["C"],
        "phase_trainable_D_param_count": trainable_counts["D"],
        "phase_trainable_B_param_count": trainable_counts["B"],
    }

def _phase_parameter_groups(
    model: nn.Module,
    *,
    model_family: str,
    phase_config: dict[str, Any] | None,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Classify parameters into C/D/B groups for phase scheduling."""

    coefficient_ids = _coefficient_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    dictionary_scale_ids = _dictionary_scale_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    dictionary_atom_ids = _learned_dictionary_atom_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    dictionary_prefixes = _dictionary_parameter_prefixes(model) if not _is_dense_model_family(model_family) else tuple()
    head_ids = _classification_head_parameter_id_set(model)
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {"C": [], "D": [], "B": []}
    for name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        placed = False
        if parameter_id in coefficient_ids:
            if _phase_scope_allows_parameter_name(name, phase_config):
                groups["C"].append((name, parameter))
            placed = True
        elif parameter_id in dictionary_scale_ids:
            if _phase_scope_allows_parameter_name(name, phase_config):
                groups["D"].append((name, parameter))
            placed = True
        elif parameter_id in dictionary_atom_ids:
            if _phase_scope_allows_parameter_name(name, phase_config) and _phase_scope_allows_dictionary_atom_parameter_name(name, phase_config):
                groups["D"].append((name, parameter))
            placed = True
        elif parameter_id in head_ids and _is_dense_model_family(model_family):
            if _phase_scope_allows_parameter_name(name, phase_config):
                groups["B"].append((name, parameter))
            placed = True
        if placed:
            continue
        if any(name.startswith(prefix) for prefix in dictionary_prefixes):
            continue
        if _phase_backbone_parameter_allowed(name, phase_config):
            groups["B"].append((name, parameter))
    return groups

def _phase_optimizer_param_counts(phase_groups: dict[str, list[tuple[str, nn.Parameter]]]) -> dict[str, int]:
    return {group: sum(int(parameter.numel()) for _name, parameter in parameters if parameter.requires_grad) for group, parameters in phase_groups.items()}


# --- Full-DiR integrity checks ----------------------------------------------
def full_dictionary_integrity_report(
    model: nn.Module,
    phase_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit that a ViT has no trainable dense carrier outside DiR modules.

    This is intentionally a structural report rather than a new loss. It checks
    that full DiR builds route patch/token/attention/MLP/head through existing
    SeparableDictionaryLinear modules and that the configured phase schedule leaves
    no B bypass.
    """

    layer_names = [name for name, _layer in iter_dictionary_layers(model)]
    phase_groups = _phase_parameter_groups(model, model_family="direct_normalized_dictionary", phase_config=phase_config or {})
    b_param_count = sum(int(parameter.numel()) for _name, parameter in phase_groups.get("B", []))
    b_trainable_count = sum(int(parameter.numel()) for _name, parameter in phase_groups.get("B", []) if parameter.requires_grad)
    dense_qkv_present = 0
    dense_qkv_trainable = 0
    attention_split_ready = 0
    for transformer_block in getattr(model, "transformer_blocks", []):
        attention = transformer_block.multi_head_self_attention
        if getattr(attention, "query_key_value_projection", None) is not None:
            dense_qkv_present += 1
            dense_qkv_trainable += sum(int(parameter.numel()) for parameter in attention.query_key_value_projection.parameters() if parameter.requires_grad)
        if (
            isinstance(getattr(attention, "query_projection", None), SeparableDictionaryLinear)
            and isinstance(getattr(attention, "key_projection", None), SeparableDictionaryLinear)
            and isinstance(getattr(attention, "value_projection", None), SeparableDictionaryLinear)
            and isinstance(getattr(attention, "output_projection", None), SeparableDictionaryLinear)
        ):
            attention_split_ready += 1
    dense_rms_layer_names = [
        name
        for name, layer in iter_dictionary_layers(model)
        if bool(getattr(layer, "_dictionary_uses_dense_rms", False))
    ]
    return {
        "full_dictionary_dictionary_layer_count": int(len(layer_names)),
        "full_dictionary_patch_is_dictionary": isinstance(getattr(model, "patch_embedding", None), DictionaryPatchEmbedding),
        "full_dictionary_class_token_is_dictionary": isinstance(getattr(model, "class_token", None), TensorDictionaryParameter),
        "full_dictionary_position_embedding_is_dictionary": isinstance(getattr(model, "position_embedding", None), PositionDictionaryParameter),
        "full_dictionary_head_is_dictionary": isinstance(getattr(model, "classification_head", None), SeparableDictionaryLinear),
        "full_dictionary_attention_split_block_count": int(attention_split_ready),
        "full_dictionary_dense_qkv_present_count": int(dense_qkv_present),
        "full_dictionary_dense_qkv_trainable_param_count": int(dense_qkv_trainable),
        "full_dictionary_phase_B_param_count": int(b_param_count),
        "full_dictionary_phase_B_trainable_param_count": int(b_trainable_count),
        "full_dictionary_dense_rms_layer_count": int(len(dense_rms_layer_names)),
        "full_dictionary_dense_rms_layer_names": ";".join(dense_rms_layer_names[:20]),
    }

def assert_full_dictionary_integrity(
    model: nn.Module,
    phase_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = full_dictionary_integrity_report(model, phase_config=phase_config or {})
    failures: list[str] = []
    for key in (
        "full_dictionary_patch_is_dictionary",
        "full_dictionary_class_token_is_dictionary",
        "full_dictionary_position_embedding_is_dictionary",
        "full_dictionary_head_is_dictionary",
    ):
        if not bool(report.get(key, False)):
            failures.append(key)
    if int(report.get("full_dictionary_attention_split_block_count", 0)) != int(getattr(model, "transformer_depth", 0)):
        failures.append("attention_qkvo_not_split_for_all_blocks")
    if int(report.get("full_dictionary_dense_qkv_present_count", 0)) != 0:
        failures.append("dense_qkv_path_present")
    if int(report.get("full_dictionary_phase_B_param_count", 0)) != 0:
        failures.append("phase_B_not_empty")
    if int(report.get("full_dictionary_dense_rms_layer_count", 0)) != 0:
        failures.append("dense_rms_leakage_detected")
    if failures:
        raise ValueError("full DiR integrity check failed: " + ",".join(failures))
    return report

def _routed_gate_eval_enabled_for_epoch(config: dict[str, Any] | None, *, epoch: int | None) -> bool:
    payload = config or {}
    if not bool(payload.get("forward_routed_gate_eval_pair_enabled", False)):
        return False
    epochs = payload.get("forward_routed_gate_eval_pair_epochs")
    if epochs is None or epochs == "":
        return True
    if epoch is None:
        return False
    try:
        epoch_set = {int(value) for value in epochs}
    except TypeError:
        return True
    return int(epoch) in epoch_set


# --- Dictionary layer construction ------------------------------------------
def _dictionary_bias_policy(config: dict[str, Any]) -> str:
    dictionary_config = config.get("dictionary", {})
    return str(dictionary_config.get("ffn_bias_policy", "zero_frozen"))

def _deterministic_dictionary_target_rms(
    dictionary_config: Mapping[str, Any],
    *,
    role: str,
    in_features: int,
    out_features: int,
) -> float:
    by_role = dictionary_config.get("deterministic_target_weight_rms_by_role", {})
    if isinstance(by_role, Mapping) and str(role) in by_role:
        return float(by_role[str(role)])
    if "deterministic_target_weight_rms" in dictionary_config:
        return float(dictionary_config["deterministic_target_weight_rms"])
    if str(role).startswith("patch"):
        return math.sqrt(2.0 / float(max(1, int(in_features))))
    return 0.02

def _coefficient_quantization_config_for_dictionary_layer(
    coefficient_quantization_config: Mapping[str, Any],
    *,
    seed: int,
    layer_seed_offset: int,
) -> dict[str, Any]:
    """Return a per-layer C config whose magnitude seed follows the run seed.

    D may intentionally share ``basis_bank_seed`` across Source and Target. C
    must remain fresh, so both supported magnitude-seed locations are derived
    from the run seed plus the layer identity.
    """

    layer_config = deepcopy(dict(coefficient_quantization_config))
    relative_payload = layer_config.get("relative_coefficient", {})
    if isinstance(relative_payload, Mapping) and bool(relative_payload.get("enabled", False)):
        magnitude_seed = int(seed) + int(layer_seed_offset)
        layer_config["magnitude_init_seed"] = magnitude_seed
        magnitude_init = layer_config.get("magnitude_init", {})
        magnitude_init = dict(magnitude_init) if isinstance(magnitude_init, Mapping) else {}
        magnitude_init["seed"] = magnitude_seed
        layer_config["magnitude_init"] = magnitude_init
    return layer_config

def _make_dictionary_linear(
    *,
    original_linear: nn.Linear | None,
    in_features: int,
    out_features: int,
    bias: bool,
    dictionary_mode: str,
    atom_count: int,
    low_atom_count: int,
    basis_type: str,
    seed: int,
    bias_policy: str,
    primitive_spec: Sequence[dict[str, int | str]] | None,
    dictionary_config: Mapping[str, Any],
    coefficient_quantization_config: Mapping[str, Any] | None,
    role: str,
) -> SeparableDictionaryLinear:
    avoid_dense_rms_leakage = bool(dictionary_config.get("avoid_dense_rms_leakage", False))
    layer_coefficient_quantization_config = dict(coefficient_quantization_config or {})
    layer_coefficient_quantization_config["_dictionary_role"] = str(role)
    if original_linear is not None and not avoid_dense_rms_leakage:
        return SeparableDictionaryLinear.from_linear(
            original_linear,
            atom_count=atom_count,
            low_atom_count=low_atom_count,
            basis_type=basis_type,
            dictionary_mode=dictionary_mode,
            seed=seed,
            bias_policy=bias_policy,
            primitive_spec=primitive_spec,
            coefficient_quantization_config=layer_coefficient_quantization_config,
        )
    return SeparableDictionaryLinear.from_dimensions(
        in_features=int(in_features),
        out_features=int(out_features),
        bias=bool(bias),
        atom_count=atom_count,
        low_atom_count=low_atom_count,
        basis_type=basis_type,
        dictionary_mode=dictionary_mode,
        seed=seed,
        target_weight_rms=_deterministic_dictionary_target_rms(dictionary_config, role=role, in_features=in_features, out_features=out_features),
        bias_policy=bias_policy,
        primitive_spec=primitive_spec,
        coefficient_quantization_config=layer_coefficient_quantization_config,
    )

class DictionaryPatchEmbedding(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        patch_embedding_kernel_size: int,
        patch_embedding_stride: int,
        patch_embedding_padding: int,
        number_of_input_channels: int,
        embedding_dimension: int,
        projection: SeparableDictionaryLinear,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.patch_embedding_kernel_size = int(patch_embedding_kernel_size)
        self.patch_embedding_stride = int(patch_embedding_stride)
        self.patch_embedding_padding = int(patch_embedding_padding)
        self.number_of_input_channels = int(number_of_input_channels)
        self.embedding_dimension = int(embedding_dimension)
        self.projection = projection
        projected_size = (self.image_size + 2 * self.patch_embedding_padding - self.patch_embedding_kernel_size) // self.patch_embedding_stride + 1
        self.number_of_patches_per_side = int(projected_size)
        self.number_of_patches = int(projected_size * projected_size)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(input_images, kernel_size=self.patch_embedding_kernel_size, padding=self.patch_embedding_padding, stride=self.patch_embedding_stride)
        return self.projection(patches.transpose(1, 2))

class TensorDictionaryParameter(nn.Module):
    def __init__(self, *, tensor_shape: Sequence[int], dictionary: SeparableDictionaryLinear) -> None:
        super().__init__()
        self.tensor_shape = tuple(int(dim) for dim in tensor_shape)
        self.dictionary = dictionary

    def forward(self) -> torch.Tensor:
        reference = self.dictionary.coefficient_magnitude
        unit = torch.ones((1, 1, 1), device=reference.device, dtype=reference.dtype)
        return self.dictionary(unit).reshape(self.tensor_shape)

class PositionDictionaryParameter(nn.Module):
    def __init__(self, *, position_module: TensorDictionaryParameter) -> None:
        super().__init__()
        self.position_module = position_module

    def forward(self) -> torch.Tensor:
        return self.position_module()


# --- Support commit and coordinate preservation -----------------------------
def _named_dictionary_scale_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Return all D-owned functional scales and non-trainable coordinate state."""

    items: list[tuple[str, torch.Tensor]] = []
    for layer_name, layer in iter_dictionary_layers(model):
        items.append((f"{layer_name}.dictionary_log_scale", layer.dictionary_log_scale))
        if layer.dictionary_role not in {"attention_q", "attention_k", "attention_v", "attention_o"}:
            # Attention projection-local coordinate buffers are nonfunctional
            # storage only. Their initialization and support-anchor shifts are
            # aggregated into the one shared QK or VO coordinate correction, so
            # including the inert local buffers in transplant/hash audits would
            # falsely advertise four extra functional scales.
            items.append((f"{layer_name}.relative_coordinate_log_scale", layer.relative_coordinate_log_scale))
    for module_name, module in model.named_modules():
        qk = getattr(module, "dictionary_qk_log_scale", None)
        vo = getattr(module, "dictionary_vo_log_scale", None)
        qk_coordinate = getattr(module, "dictionary_qk_coordinate_log_scale", None)
        vo_coordinate = getattr(module, "dictionary_vo_coordinate_log_scale", None)
        if isinstance(qk, torch.Tensor):
            items.append((f"{module_name}.dictionary_qk_log_scale", qk))
        if isinstance(vo, torch.Tensor):
            items.append((f"{module_name}.dictionary_vo_log_scale", vo))
        if isinstance(qk_coordinate, torch.Tensor):
            items.append((f"{module_name}.dictionary_qk_coordinate_log_scale", qk_coordinate))
        if isinstance(vo_coordinate, torch.Tensor):
            items.append((f"{module_name}.dictionary_vo_coordinate_log_scale", vo_coordinate))
    return items

def _forward_routed_hard_mask_from_mass(
    mass: torch.Tensor,
    *,
    alpha: float,
    tolerance: float,
    score_eps: float,
    entmax_iterations: int = 16,
) -> torch.Tensor:
    """Convert the current model's stored route-mass vector into a 0/1 support mask.

    This helper is used when that model commits its own hard support. In the
    current DiR release, Source support is not transplanted to Targets: each
    Target keeps its own route/support state and commits its own support.
    """

    mass_f = mass.detach().float()
    if int(mass_f.numel()) <= 0:
        return torch.zeros_like(mass_f, dtype=torch.bool)
    mass_distribution = mass_f / mass_f.sum().clamp_min(float(score_eps))
    p = _entmax_alpha_1d(
        mass_distribution.clamp_min(float(score_eps)).log(),
        float(alpha),
        iterations=max(1, int(entmax_iterations) or 1),
    )
    hard = p > float(tolerance)
    if int(hard.sum().item()) <= 0:
        hard[int(torch.argmax(p).item())] = True
    return hard.to(dtype=torch.bool)

def _relative_support_commit_targets(
    model: nn.Module,
) -> tuple[tuple[nn.Module, ...], tuple[nn.Module, ...]]:
    """Return cached attention-group and local relative-support commit targets."""

    cached = getattr(model, "_relative_support_commit_targets_cache", None)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached
    attention_groups: list[nn.Module] = []
    for module in model.modules():
        if callable(getattr(module, "flush_dictionary_coordinate_corrections_", None)):
            attention_groups.append(module)
    local_layers: list[nn.Module] = []
    for _layer_name, layer in iter_dictionary_layers(model):
        if getattr(layer, "dictionary_role", "") in {
            "attention_q", "attention_k", "attention_v", "attention_o"
        }:
            continue
        if callable(getattr(layer, "commit_pending_relative_support_transition_", None)):
            local_layers.append(layer)
    cached = (tuple(attention_groups), tuple(local_layers))
    setattr(model, "_relative_support_commit_targets_cache", cached)
    return cached

@torch.no_grad()
def commit_pending_relative_support_transitions_(model: nn.Module) -> None:
    """Commit every dynamic support/q transition once after optimizer.step()."""

    attention_groups, local_layers = _relative_support_commit_targets(model)
    # Attention owns one shared functional scale for QK and one for VO.
    for module in attention_groups:
        module.flush_dictionary_coordinate_corrections_()
    # All remaining dictionary operators own one local coordinate scale.
    for layer in local_layers:
        layer.commit_pending_relative_support_transition_(apply_local_scale=True)

@torch.no_grad()
def fold_relative_coordinate_scales_into_dictionary_scales_(model: nn.Module) -> dict[str, Any]:
    """Fold non-learned support-coordinate state into learned log-scales.

    The operation is an exact log-coordinate translation performed only after
    the persistent route support is committed. It adds no model freedom and
    leaves every represented operator unchanged.
    """

    folded_non_attention = 0
    folded_attention_groups = 0
    max_abs_log_correction = 0.0
    # First collect any queued Q/K/V/O corrections into their shared buffers.
    for module in model.modules():
        flush = getattr(module, "flush_dictionary_coordinate_corrections_", None)
        if callable(flush):
            flush()
    # Non-attention operators own one local learned dictionary log-scale.
    for _layer_name, layer in iter_dictionary_layers(model):
        if getattr(layer, "dictionary_role", "") in {"attention_q", "attention_k", "attention_v", "attention_o"}:
            continue
        coordinate = getattr(layer, "relative_coordinate_log_scale", None)
        learned = getattr(layer, "dictionary_log_scale", None)
        if not isinstance(coordinate, torch.Tensor) or not isinstance(learned, torch.Tensor):
            continue
        value = float(coordinate.detach().abs().cpu())
        max_abs_log_correction = max(max_abs_log_correction, value)
        learned.add_(coordinate.to(device=learned.device, dtype=learned.dtype))
        coordinate.zero_()
        folded_non_attention += 1
    # Attention uses one functional QK scale and one functional VO scale.
    for module in model.modules():
        for learned_name, coordinate_name in (
            ("dictionary_qk_log_scale", "dictionary_qk_coordinate_log_scale"),
            ("dictionary_vo_log_scale", "dictionary_vo_coordinate_log_scale"),
        ):
            learned = getattr(module, learned_name, None)
            coordinate = getattr(module, coordinate_name, None)
            if not isinstance(learned, torch.Tensor) or not isinstance(coordinate, torch.Tensor):
                continue
            value = float(coordinate.detach().abs().cpu())
            max_abs_log_correction = max(max_abs_log_correction, value)
            learned.add_(coordinate.to(device=learned.device, dtype=learned.dtype))
            coordinate.zero_()
            folded_attention_groups += 1
    return {
        "relative_coordinate_fold_applied": bool(folded_non_attention or folded_attention_groups),
        "relative_coordinate_fold_non_attention_layer_count": int(folded_non_attention),
        "relative_coordinate_fold_attention_group_count": int(folded_attention_groups),
        "relative_coordinate_fold_max_abs_log_correction": float(max_abs_log_correction),
        "relative_coordinate_preserved_target_owned": False,
    }

@torch.no_grad()
def preserve_relative_coordinate_scales_after_support_commit_(model: nn.Module) -> dict[str, Any]:
    """Flush support corrections into Target-owned coordinate buffers without folding.

    DiR Target models freeze the copied Source D-owned learned scales. Folding a
    support-coordinate correction into those scales and then restoring the frozen
    Source values would erase the Target's exact coordinate compensation. Attention
    pending corrections are therefore flushed into the shared QK/VO coordinate
    buffers, while all coordinate buffers remain explicit checkpoint state.
    """

    non_attention_layers = 0
    attention_groups = 0
    max_abs_log_correction = 0.0
    for module in model.modules():
        flush = getattr(module, "flush_dictionary_coordinate_corrections_", None)
        if callable(flush):
            flush()
    for _layer_name, layer in iter_dictionary_layers(model):
        if getattr(layer, "dictionary_role", "") in {
            "attention_q", "attention_k", "attention_v", "attention_o"
        }:
            continue
        coordinate = getattr(layer, "relative_coordinate_log_scale", None)
        if not isinstance(coordinate, torch.Tensor):
            continue
        max_abs_log_correction = max(
            max_abs_log_correction, float(coordinate.detach().abs().cpu())
        )
        non_attention_layers += 1
    for module in model.modules():
        found = False
        for coordinate_name in (
            "dictionary_qk_coordinate_log_scale",
            "dictionary_vo_coordinate_log_scale",
        ):
            coordinate = getattr(module, coordinate_name, None)
            if not isinstance(coordinate, torch.Tensor):
                continue
            found = True
            max_abs_log_correction = max(
                max_abs_log_correction, float(coordinate.detach().abs().cpu())
            )
        if found:
            attention_groups += 1
    return {
        "relative_coordinate_fold_applied": False,
        "relative_coordinate_fold_non_attention_layer_count": 0,
        "relative_coordinate_fold_attention_group_count": 0,
        "relative_coordinate_fold_max_abs_log_correction": float(
            max_abs_log_correction
        ),
        "relative_coordinate_preserved_target_owned": True,
        "relative_coordinate_preserved_non_attention_layer_count": int(
            non_attention_layers
        ),
        "relative_coordinate_preserved_attention_module_count": int(attention_groups),
    }

def commit_forward_routed_hard_support_masks_from_ema(
    model: nn.Module,
    *,
    alpha: float | None = None,
    tolerance: float | None = None,
    score_eps: float | None = None,
    entmax_iterations: int | None = None,
    prefix: str = "forward_routed_hard_support_commit",
    fold_relative_coordinate_scales: bool = True,
) -> dict[str, Any]:
    """Commit EMA route state into persistent hard-support masks.

    Normal runs cut the full EMA vector. coefficient-contract Condition C instead cuts only
    the inactive residual pool, appends a virtual null candidate, permits an
    empty residual result, and unions that result with the protected A-active
    support before coordinate correction and persistence.
    """

    counts: list[int] = []
    residual_counts: list[int] = []
    alphas: list[float] = []
    initialized_all = True
    committed_layers = 0
    contract_aware_layers = 0
    for _layer_name, layer in iter_dictionary_layers(model):
        if not hasattr(layer, "global_solution_usage_ema") or not hasattr(layer, "forward_routed_fixed_support_mask"):
            continue
        ema_initialized = bool(getattr(layer, "_global_solution_usage_ema_is_initialized", lambda: False)())
        initialized_all = initialized_all and ema_initialized
        if not ema_initialized:
            continue
        layer_alpha = float(alpha if alpha is not None else getattr(layer, "forward_routed_gate_alpha", 1.0))
        layer_tolerance = float(tolerance if tolerance is not None else getattr(layer, "forward_routed_gate_support_tolerance", 1e-8))
        layer_score_eps = float(score_eps if score_eps is not None else getattr(layer, "forward_routed_gate_score_eps", 1e-6))
        layer_iterations = max(1, int(entmax_iterations if entmax_iterations is not None else getattr(layer, "forward_routed_gate_entmax_iterations", 16) or 16))

        residual_enabled = bool(getattr(layer, "_residual_route_enabled_cached", False))
        force_on = getattr(layer, "_route_force_on_mask", None)
        trainable = getattr(layer, "_route_trainable_mask", None)
        if residual_enabled:
            if not isinstance(force_on, torch.Tensor) or not isinstance(trainable, torch.Tensor):
                raise RuntimeError("coefficient-contract residual support commit is missing persistent contract masks")
            force_on = force_on.to(device=layer.global_solution_usage_ema.device, dtype=torch.bool).reshape(-1)
            trainable = trainable.to(device=layer.global_solution_usage_ema.device, dtype=torch.bool).reshape(-1)
            if int(force_on.numel()) != int(layer.atom_count) or int(trainable.numel()) != int(layer.atom_count):
                raise RuntimeError("coefficient-contract residual support commit mask length mismatch")
            candidate_indices = torch.nonzero(trainable, as_tuple=False).reshape(-1)
            if int(candidate_indices.numel()) > 0:
                residual_mass = layer.global_solution_usage_ema.detach().float().index_select(0, candidate_indices)
                logits = torch.cat(
                    (residual_mass.clamp_min(layer_score_eps).log(), residual_mass.new_zeros((1,))),
                    dim=0,
                )
                probability = _entmax_alpha_1d(
                    logits,
                    layer_alpha,
                    iterations=layer_iterations,
                )
                residual_candidate_mask = probability[:-1] > layer_tolerance
                residual_mask = torch.zeros_like(force_on)
                residual_mask.index_copy_(0, candidate_indices, residual_candidate_mask)
            else:
                residual_mask = torch.zeros_like(force_on)
            mask = force_on | residual_mask
            residual_counts.append(int(residual_mask.sum().item()))
            contract_aware_layers += 1
        else:
            mask = _forward_routed_hard_mask_from_mass(
                layer.global_solution_usage_ema,
                alpha=layer_alpha,
                tolerance=layer_tolerance,
                score_eps=layer_score_eps,
                entmax_iterations=layer_iterations,
            ).to(device=layer.forward_routed_fixed_support_mask.device)
            residual_counts.append(0)

        mask = mask.to(device=layer.forward_routed_fixed_support_mask.device, dtype=torch.bool)
        if bool(getattr(layer, "relative_coefficient_enabled", False)):
            previous = (
                layer.forward_routed_previous_support.to(device=mask.device, dtype=torch.bool)
                if bool(getattr(layer, "_forward_routed_support_seen", False))
                else layer._static_relative_support_mask(device=mask.device)
            )
            layer._relative_support_coordinate_correction_tensor_(previous, mask)
        layer.forward_routed_fixed_support_mask.copy_(mask)
        layer._set_forward_routed_fixed_support_initialized(True)
        layer.forward_routed_previous_support.copy_(mask.to(device=layer.forward_routed_previous_support.device))
        layer.forward_routed_support_union.logical_or_(mask.to(device=layer.forward_routed_support_union.device))
        layer._forward_routed_support_seen = True
        counts.append(int(mask.detach().float().sum().item()))
        alphas.append(float(layer_alpha))
        committed_layers += 1
    fold_report = (
        fold_relative_coordinate_scales_into_dictionary_scales_(model)
        if bool(fold_relative_coordinate_scales)
        else preserve_relative_coordinate_scales_after_support_commit_(model)
    )
    return {
        f"{prefix}_applied": bool(committed_layers > 0),
        f"{prefix}_layer_count": int(committed_layers),
        f"{prefix}_ema_initialized_all_layers": bool(initialized_all and committed_layers > 0),
        f"{prefix}_active_count_min": min(counts) if counts else 0,
        f"{prefix}_active_count_max": max(counts) if counts else 0,
        f"{prefix}_active_count_mean": (sum(counts) / len(counts)) if counts else 0.0,
        f"{prefix}_active_count_per_layer": ";".join(str(count) for count in counts),
        f"{prefix}_residual_active_count_min": min(residual_counts) if residual_counts else 0,
        f"{prefix}_residual_active_count_max": max(residual_counts) if residual_counts else 0,
        f"{prefix}_residual_active_count_mean": (sum(residual_counts) / len(residual_counts)) if residual_counts else 0.0,
        f"{prefix}_contract_aware_layer_count": int(contract_aware_layers),
        f"{prefix}_alpha_min": min(alphas) if alphas else 0.0,
        f"{prefix}_alpha_max": max(alphas) if alphas else 0.0,
        **fold_report,
    }


# --- Gradient norm helpers --------------------------------------------------
def _classification_head_parameter_id_set(model: nn.Module) -> set[int]:
    if not hasattr(model, "classification_head"):
        return set()
    return {id(parameter) for parameter in model.classification_head.parameters()}

def _dictionary_parameter_prefixes(model: nn.Module) -> tuple[str, ...]:
    return tuple(f"{name}." for name, _layer in iter_dictionary_layers(model))

def _add_grad_sq(total: torch.Tensor | None, term: torch.Tensor) -> torch.Tensor:
    term = term.detach().float()
    if total is None:
        return term
    if total.device != term.device:
        term = term.to(total.device)
    return total + term

def _sqrt_tensor_sum(total: torch.Tensor | None) -> float:
    if total is None:
        return 0.0
    return math.sqrt(max(0.0, float(total.detach().item())))

def _parameter_grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    total_sq: torch.Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total_sq = _add_grad_sq(total_sq, parameter.grad.detach().float().pow(2).sum())
    return _sqrt_tensor_sum(total_sq)
