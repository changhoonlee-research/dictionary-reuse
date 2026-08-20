"""Model construction, datasets, optimizers, evaluation, gradients, and training diagnostics."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ..model.vit import create_vision_transformer_small_patch4_for_cifar100

from .schema import (
    _DATASET_CACHE,
    _TASK_SUBSET_CACHE,
    _is_dense_model_family,
    _min_mean_max_metrics,
)
from ..model.basis import (
    _basis_primitive_spec_from_config,
    _set_seed,
    _stable_json_hash,
    _topk_indices_1d,
)
from ..model.routing import (
    _rank_correlation_for_abs_values,
    _scalar_report_float,
    _spearman_correlation,
)
from ..model.dictionary_operator import (
    _basis_type_is_block_profile,
    _block_index_from_layer_name,
    _ffn_layer_kind_from_layer_name,
    iter_dictionary_layers,
)
from .sparsity import _contribution_support_counts_from_mass
from .schedule import (
    _add_grad_sq,
    _classification_head_parameter_id_set,
    _coefficient_parameter_id_set,
    _dictionary_bias_policy,
    _dictionary_parameter_prefixes,
    _dictionary_scale_parameter_id_set,
    _named_dictionary_scale_tensors,
    _parameter_grad_norm,
    _sqrt_tensor_sum,
    apply_dictionary_to_attention_layers,
    apply_dictionary_to_classification_head,
    apply_dictionary_to_ffn_layers,
    apply_dictionary_to_patch_embedding,
    apply_dictionary_to_token_embeddings,
    full_dictionary_integrity_report,
)

def model_state_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def optional_max_batches(value: Any, default: int | None = None) -> int | None:
    if value in {None, "", "none", "None", "full", "all"}:
        return default
    parsed = int(value)
    return parsed if parsed > 0 else None


def build_model(
    config: dict[str, Any],
    *,
    model_family: str,
    seed: int,
    basis_type: str | None = None,
    dictionary_config_override: dict[str, Any] | None = None,
) -> nn.Module:
    _set_seed(seed)
    dictionary = dictionary_config_override or config["dictionary"]
    apply_scope_for_init = str(dictionary.get("apply_scope", "")).lower()
    skip_dense_carrier_initialization = (
        model_family == "direct_normalized_dictionary"
        and apply_scope_for_init in {"full_dictionary", "full_carrier_dictionary"}
        and bool(dictionary.get("avoid_dense_rms_leakage", False))
    )
    model = create_vision_transformer_small_patch4_for_cifar100(
        number_of_classes=int(config["dataset"]["number_of_classes"]),
        initialize_parameters=not skip_dense_carrier_initialization,
    )
    if bool(skip_dense_carrier_initialization):
        model._dense_carrier_initialization_skipped = True
    if model_family == "direct_normalized_dictionary":
        atom_count = int(dictionary["atom_count"])
        low_atom_count = int(dictionary["low_atom_count"])
        resolved_basis_type = str(basis_type or dictionary["basis_type"])
        primitive_spec = None
        if not _basis_type_is_block_profile(dictionary, resolved_basis_type):
            primitive_spec = _basis_primitive_spec_from_config(dictionary, resolved_basis_type, atom_count, low_atom_count)
        basis_bank_seed = int(dictionary.get("basis_bank_seed", seed))
        shared_basis_bank = bool(dictionary.get("shared_basis_bank", False))
        bias_policy = _dictionary_bias_policy(config)
        apply_scope = str(dictionary.get("apply_scope", "")).lower()
        if bool(dictionary.get("dictionary_patch_enabled", False)) or apply_scope in {"full_dictionary", "full_carrier_dictionary"}:
            apply_dictionary_to_patch_embedding(
                model,
                dictionary_mode=model_family,
                atom_count=atom_count,
                low_atom_count=low_atom_count,
                basis_type=resolved_basis_type,
                seed=seed,
                primitive_spec=primitive_spec,
                basis_bank_seed=basis_bank_seed,
                bias_policy=bias_policy,
                dictionary_config=dictionary,
            )
        if bool(dictionary.get("dictionary_token_embedding_enabled", False)) or apply_scope in {"full_dictionary", "full_carrier_dictionary"}:
            apply_dictionary_to_token_embeddings(
                model,
                dictionary_mode=model_family,
                atom_count=atom_count,
                low_atom_count=low_atom_count,
                basis_type=resolved_basis_type,
                seed=seed,
                primitive_spec=primitive_spec,
                basis_bank_seed=basis_bank_seed,
                bias_policy=bias_policy,
                dictionary_config=dictionary,
            )
        if bool(dictionary.get("dictionary_attention_enabled", False)) or apply_scope in {"full_dictionary", "full_carrier_dictionary", "attention_mlp"}:
            apply_dictionary_to_attention_layers(
                model,
                dictionary_mode=model_family,
                atom_count=atom_count,
                low_atom_count=low_atom_count,
                basis_type=resolved_basis_type,
                seed=seed,
                primitive_spec=primitive_spec,
                basis_bank_seed=basis_bank_seed,
                shared_basis_bank=shared_basis_bank,
                bias_policy=bias_policy,
                dictionary_config=dictionary,
            )
        apply_dictionary_to_ffn_layers(
            model,
            dictionary_mode=model_family,
            atom_count=atom_count,
            low_atom_count=low_atom_count,
            basis_type=resolved_basis_type,
            seed=seed,
            primitive_spec=primitive_spec,
            basis_bank_seed=basis_bank_seed,
            shared_basis_bank=shared_basis_bank,
            bias_policy=bias_policy,
            dictionary_config=dictionary,
        )
        if bool(dictionary.get("dictionary_head_enabled", False)) or apply_scope in {"full_dictionary", "full_carrier_dictionary"}:
            apply_dictionary_to_classification_head(
                model,
                dictionary_mode=model_family,
                atom_count=atom_count,
                low_atom_count=low_atom_count,
                basis_type=resolved_basis_type,
                seed=seed,
                primitive_spec=primitive_spec,
                basis_bank_seed=basis_bank_seed,
                bias_policy=bias_policy,
                dictionary_config=dictionary,
            )
        if apply_scope in {"full_dictionary", "full_carrier_dictionary"} and bool(dictionary.get("require_full_dictionary_integrity", False)):
            # Build-time check excludes phase-B because phase config is selected per run.
            build_report = full_dictionary_integrity_report(model, phase_config={"backbone_scope": "none"})
            if (
                not bool(build_report.get("full_dictionary_patch_is_dictionary", False))
                or not bool(build_report.get("full_dictionary_class_token_is_dictionary", False))
                or not bool(build_report.get("full_dictionary_position_embedding_is_dictionary", False))
                or not bool(build_report.get("full_dictionary_head_is_dictionary", False))
                or int(build_report.get("full_dictionary_attention_split_block_count", 0)) != int(getattr(model, "transformer_depth", 0))
                or int(build_report.get("full_dictionary_dense_qkv_present_count", 0)) != 0
                or int(build_report.get("full_dictionary_dense_rms_layer_count", 0)) != 0
            ):
                raise ValueError(f"full DiR build integrity check failed: {build_report}")
    elif not _is_dense_model_family(model_family):
        raise ValueError(f"Unknown model_family={model_family!r}")
    return model


# --- Dataset and loader construction ----------------------------------------
def _select_class_indices(dataset: Any, classes: Sequence[int], per_class_limit: int) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise ValueError("CIFAR-100 dataset must expose targets or labels")
    classes_set = {int(item) for item in classes}
    per_class_counts = {int(item): 0 for item in classes}
    indices: list[int] = []
    for index, target in enumerate(targets):
        target_int = int(target)
        if target_int not in classes_set:
            continue
        if per_class_counts[target_int] >= int(per_class_limit):
            continue
        indices.append(index)
        per_class_counts[target_int] += 1
        if all(count >= int(per_class_limit) for count in per_class_counts.values()):
            break
    return indices

def _normalize_cifar100_split(split_name: str) -> str:
    normalized = str(split_name).strip().lower()
    aliases = {
        "training": "train",
        "cifar100_train": "train",
        "cifar-100_train": "train",
        "eval": "test",
        "evaluation": "test",
        "official_test": "test",
        "official_cifar100_test": "test",
        "cifar100_test": "test",
        "cifar-100_test": "test",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"train", "test"}:
        raise ValueError(f"Unsupported CIFAR-100 split {split_name!r}; expected 'train' or 'test'.")
    return normalized

def atom_usage_console_metrics(model: nn.Module) -> dict[str, float]:
    """Return a compact gate-active atom summary for sparse console logs.

    This is reporting-only. It does not alter gate schedule state and uses the
    current deterministic gate value with a 0.5 active threshold, matching the
    A1Q "how many atoms are currently usable" interpretation.
    """

    active_total = 0.0
    atom_total = 0.0
    gate_sum = 0.0
    alpha_sum = 0.0
    mass95_sum = 0.0
    mass95_count = 0
    layer_count = 0
    with torch.no_grad():
        for _name, layer in iter_dictionary_layers(model):
            atom_count = int(getattr(layer, "atom_count", 0) or 0)
            if atom_count <= 0:
                continue
            layer_count += 1
            fixed_route_mask = getattr(layer, "forward_routed_fixed_support_mask", None)
            if (
                bool(getattr(layer, "_forward_routed_fixed_support_is_initialized", lambda: False)())
                and isinstance(fixed_route_mask, torch.Tensor)
                and bool(getattr(layer, "forward_routed_gate_enabled", False))
            ):
                gate = fixed_route_mask.detach().float().cpu()
            else:
                route_gate = getattr(layer, "_last_forward_routed_hard_gate", None)
                if bool(getattr(layer, "_last_forward_routed_gate_enabled", False)) and isinstance(route_gate, torch.Tensor):
                    gate = route_gate.detach().float().cpu()
                else:
                    gate = torch.ones(atom_count, dtype=torch.float32)
            active_total += float((gate >= 0.5).sum().item())
            atom_total += float(gate.numel())
            gate_sum += float(gate.sum().item())
            alpha_sum += float(getattr(layer, "forward_routed_gate_alpha", 0.0) or 0.0)
            route_mass95 = getattr(layer, "_last_forward_routed_pre_route_mass95_atoms", None)
            if torch.is_tensor(route_mass95) and int(route_mass95.numel()) > 0:
                mass95_sum += float(route_mass95.detach().float().mean().cpu())
                mass95_count += 1
    if layer_count <= 0 or atom_total <= 0.0:
        return {
            "atom_usage_layer_count": 0.0,
            "atom_usage_active_total": 0.0,
            "atom_usage_atom_total": 0.0,
            "atom_usage_active_per_layer": 0.0,
            "atom_usage_atoms_per_layer": 0.0,
            "atom_usage_active_ratio": 0.0,
            "atom_usage_gate_mean": 0.0,
            "route_alpha_mean": 0.0,
            "route_mass95_mean": 0.0,
        }
    return {
        "atom_usage_layer_count": float(layer_count),
        "atom_usage_active_total": float(active_total),
        "atom_usage_atom_total": float(atom_total),
        "atom_usage_active_per_layer": float(active_total) / float(layer_count),
        "atom_usage_atoms_per_layer": float(atom_total) / float(layer_count),
        "atom_usage_active_ratio": float(active_total) / float(atom_total),
        "atom_usage_gate_mean": float(gate_sum) / float(atom_total),
        "route_alpha_mean": float(alpha_sum) / float(layer_count),
        "route_mass95_mean": float(mass95_sum) / float(mass95_count) if mass95_count > 0 else 0.0,
    }

def _format_atom_usage_console_fields(
    atom_usage: dict[str, float],
    sparsity_metrics: dict[str, Any] | None = None,
) -> str:
    """Return the one-line atom usage suffix appended to epoch_eval logs only."""

    _ = sparsity_metrics
    if float(atom_usage.get("atom_usage_layer_count", 0.0) or 0.0) <= 0.0:
        return ""
    active_per_layer = float(atom_usage.get("atom_usage_active_per_layer", 0.0) or 0.0)
    atoms_per_layer = float(atom_usage.get("atom_usage_atoms_per_layer", 0.0) or 0.0)
    alpha = float(atom_usage.get("route_alpha_mean", 0.0) or 0.0)
    mass95 = float(atom_usage.get("route_mass95_mean", 0.0) or 0.0)
    fields = [f"atoms={active_per_layer:.1f}/{atoms_per_layer:.0f}"]
    if alpha > 0.0:
        fields.append(f"alpha={alpha:.3f}")
    if mass95 > 0.0:
        fields.append(f"mass95={mass95:.1f}")
    return " ".join(fields)

def _dataset_source_split(dataset_config: dict[str, Any], *, loader_split_name: str) -> str:
    loader_split = str(loader_split_name).strip().lower()
    if loader_split == "train":
        configured = dataset_config.get("train_split", dataset_config.get("dataset_split", "train"))
    elif loader_split in {"eval", "test", "validation"}:
        configured = dataset_config.get("eval_split", dataset_config.get("test_split", "test"))
    else:
        raise ValueError(f"Unsupported loader split {loader_split_name!r}")
    return _normalize_cifar100_split(str(configured))

def _images_per_class_for_loader_split(dataset_config: dict[str, Any], *, loader_split_name: str) -> int:
    loader_split = str(loader_split_name).strip().lower()
    if loader_split == "train":
        value = dataset_config.get("train_images_per_class", dataset_config.get("images_per_class"))
    elif loader_split in {"eval", "test", "validation"}:
        value = dataset_config.get("eval_images_per_class", dataset_config.get("images_per_class"))
    else:
        raise ValueError(f"Unsupported loader split {loader_split_name!r}")
    if value is None:
        raise ValueError("dataset images_per_class must be configured")
    return int(value)

def _validate_train_eval_sources(config: dict[str, Any], *, task_key: str) -> None:
    dataset_config = config["dataset"]
    train_split = _dataset_source_split(dataset_config, loader_split_name="train")
    eval_split = _dataset_source_split(dataset_config, loader_split_name="eval")
    if train_split == eval_split:
        raise ValueError(
            f"Train and eval loaders both use CIFAR-100 split {train_split!r} for {task_key}; "
            "the release requires a separate eval split."
        )

def _dataset_cache_key(dataset_config: dict[str, Any], *, dataset_split: str | None = None) -> str:
    split = _normalize_cifar100_split(dataset_split or dataset_config.get("dataset_split", "train"))
    return _stable_json_hash(
        {
            "dataset_root": str(Path(str(dataset_config["dataset_root"])).expanduser()),
            "dataset_split": split,
            "normalization_mean": dataset_config.get("normalization_mean"),
            "normalization_standard_deviation": dataset_config.get("normalization_standard_deviation"),
            "cifar100_download": dataset_config.get("cifar100_download"),
        }
    )

def _load_cifar100_dataset(
    dataset_config: dict[str, Any],
    *,
    dataset_split: str,
    cache_enabled: bool = True,
) -> Any:
    """Load a CIFAR-100 split once while keeping per-run DataLoader RNG isolated."""

    from ..cifar100_measurement_dataset import build_cifar100_evaluation_dataset, build_cifar100_training_dataset

    split = _normalize_cifar100_split(dataset_split)
    cache_key = _dataset_cache_key(dataset_config, dataset_split=split)
    if cache_enabled and cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]
    dataset_root = Path(str(dataset_config["dataset_root"])).expanduser()
    builder = build_cifar100_training_dataset if split == "train" else build_cifar100_evaluation_dataset
    dataset = builder(
        dataset_root,
        normalization_mean=dataset_config["normalization_mean"],
        normalization_standard_deviation=dataset_config["normalization_standard_deviation"],
        cifar100_download_options=dataset_config.get("cifar100_download"),
    )
    if cache_enabled:
        _DATASET_CACHE[cache_key] = dataset
    return dataset

def _task_subset_cache_key(
    dataset_config: dict[str, Any],
    task_key: str,
    loader_split_name: str = "train",
) -> str:
    task_config = dataset_config[task_key]
    dataset_split = _dataset_source_split(dataset_config, loader_split_name=loader_split_name)
    images_per_class = _images_per_class_for_loader_split(dataset_config, loader_split_name=loader_split_name)
    return _stable_json_hash(
        {
            "dataset": _dataset_cache_key(dataset_config, dataset_split=dataset_split),
            "loader_split_name": str(loader_split_name),
            "dataset_split": dataset_split,
            "task_key": task_key,
            "classes": list(task_config["classes"]),
            "images_per_class": int(images_per_class),
        }
    )

def _build_task_subset(config: dict[str, Any], *, task_key: str, loader_split_name: str = "train") -> Subset:
    _validate_train_eval_sources(config, task_key=task_key)
    dataset_config = config["dataset"]
    runtime_config = config.get("runtime", {})
    dataset_cache_enabled = bool(runtime_config.get("cache_dataset_object", True))
    subset_cache_enabled = bool(runtime_config.get("cache_task_subsets", dataset_cache_enabled))
    dataset_split = _dataset_source_split(dataset_config, loader_split_name=loader_split_name)
    images_per_class = _images_per_class_for_loader_split(dataset_config, loader_split_name=loader_split_name)
    cache_key = _task_subset_cache_key(dataset_config, task_key, loader_split_name=loader_split_name)
    if subset_cache_enabled and cache_key in _TASK_SUBSET_CACHE:
        return _TASK_SUBSET_CACHE[cache_key]
    dataset = _load_cifar100_dataset(dataset_config, dataset_split=dataset_split, cache_enabled=dataset_cache_enabled)
    task_config = dataset_config[task_key]
    indices = _select_class_indices(dataset, task_config["classes"], int(images_per_class))
    if not indices:
        raise ValueError(f"No indices selected for {task_key} {loader_split_name} split")
    subset = Subset(dataset, indices)
    if subset_cache_enabled:
        _TASK_SUBSET_CACHE[cache_key] = subset
    return subset

def _worker_init_for_seed(data_order_seed: int):
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = int(data_order_seed) + int(worker_id)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn

def build_train_loader(
    config: dict[str, Any],
    *,
    task_key: str,
    data_order_seed: int | None = None,
) -> DataLoader:
    """Build a fresh train DataLoader so shuffle RNG state is isolated per sub-run."""

    runtime_config = config.get("runtime", {})
    subset = _build_task_subset(config, task_key=task_key, loader_split_name="train")
    data_order_seed = int(data_order_seed if data_order_seed is not None else runtime_config.get("data_order_seed", runtime_config.get("base_seed", 20260527)))
    generator = torch.Generator()
    generator.manual_seed(data_order_seed)
    return DataLoader(
        subset,
        batch_size=int(runtime_config["batch_size"]),
        shuffle=True,
        generator=generator,
        worker_init_fn=_worker_init_for_seed(data_order_seed),
        num_workers=int(runtime_config.get("num_workers", 2)),
        pin_memory=bool(torch.cuda.is_available()),
    )

def build_eval_loader(config: dict[str, Any], *, task_key: str) -> DataLoader:
    """Build a deterministic evaluation DataLoader once and reuse it across sub-runs."""

    runtime_config = config.get("runtime", {})
    return DataLoader(
        _build_task_subset(config, task_key=task_key, loader_split_name="eval"),
        batch_size=int(runtime_config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(runtime_config.get("num_workers", 2)),
        pin_memory=bool(torch.cuda.is_available()),
    )


# --- Optimizer and trainability ---------------------------------------------
def _dictionary_parameter_id_set(model: nn.Module) -> set[int]:
    ids: set[int] = set(_dictionary_scale_parameter_id_set(model))
    for _name, layer in iter_dictionary_layers(model):
        for parameter in (layer.coefficient_magnitude,):
            ids.add(id(parameter))
        ids.update({id(layer.row_atoms), id(layer.col_atoms)})
        if layer.bias is not None:
            ids.add(id(layer.bias))
    return ids

def apply_learning_rate_profile_trainability(
    model: nn.Module,
    profile: LearningRateProfile,
    *,
    model_family: str,
) -> None:
    """Freeze zero-lr parameter groups before autograd builds unnecessary graphs."""

    coefficient_trainable = float(profile.coefficient_lr) > 0.0
    dictionary_trainable = float(profile.dictionary_lr) > 0.0
    non_dictionary_trainable = float(profile.non_dictionary_lr) > 0.0
    head_lr = profile.non_dictionary_lr if profile.head_lr is None else float(profile.head_lr)
    head_trainable = float(head_lr) > 0.0
    dictionary_ids = _dictionary_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    head_ids = _classification_head_parameter_id_set(model)
    if not _is_dense_model_family(model_family):
        for _name, layer in iter_dictionary_layers(model):
            layer.coefficient_magnitude.requires_grad_(
                coefficient_trainable and bool(layer.coefficient_magnitude.requires_grad)
            )
            dictionary_scale = getattr(layer, "dictionary_log_scale", None)
            if isinstance(dictionary_scale, nn.Parameter):
                hard_frozen = bool(getattr(dictionary_scale, "_transplanted_dictionary_scale_hard_frozen", False))
                dictionary_scale.requires_grad_(dictionary_trainable and not hard_frozen)
            layer.row_atoms.requires_grad_(dictionary_trainable and bool(layer.row_atoms.requires_grad))
            layer.col_atoms.requires_grad_(dictionary_trainable and bool(layer.col_atoms.requires_grad))
            if layer.bias is not None and float(profile.non_dictionary_lr) <= 0.0:
                layer.bias.requires_grad_(False)
        for _module_name, module in model.named_modules():
            for attr in ("dictionary_qk_log_scale", "dictionary_vo_log_scale"):
                dictionary_scale = getattr(module, attr, None)
                if isinstance(dictionary_scale, nn.Parameter):
                    hard_frozen = bool(getattr(dictionary_scale, "_transplanted_dictionary_scale_hard_frozen", False))
                    dictionary_scale.requires_grad_(dictionary_trainable and not hard_frozen)
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in dictionary_ids:
            continue
        if parameter_id in head_ids:
            parameter.requires_grad_(head_trainable and bool(parameter.requires_grad))
        else:
            parameter.requires_grad_(non_dictionary_trainable and bool(parameter.requires_grad))

def build_optimizer(
    model: nn.Module,
    profile: LearningRateProfile,
    *,
    model_family: str,
) -> torch.optim.Optimizer:
    apply_learning_rate_profile_trainability(model, profile, model_family=model_family)
    head_lr = profile.non_dictionary_lr if profile.head_lr is None else float(profile.head_lr)
    coefficient_parameters: list[nn.Parameter] = []
    dictionary_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    backbone_parameters: list[nn.Parameter] = []
    dictionary_ids = _dictionary_parameter_id_set(model) if not _is_dense_model_family(model_family) else set()
    head_ids = _classification_head_parameter_id_set(model)
    if not _is_dense_model_family(model_family):
        for _name, layer in iter_dictionary_layers(model):
            for parameter in (layer.coefficient_magnitude,):
                if parameter.requires_grad:
                    coefficient_parameters.append(parameter)
            for parameter in (layer.row_atoms, layer.col_atoms):
                if parameter.requires_grad:
                    dictionary_parameters.append(parameter)
        seen_dictionary_parameter_ids = {id(parameter) for parameter in dictionary_parameters}
        for _scale_name, parameter in _named_dictionary_scale_tensors(model):
            if isinstance(parameter, nn.Parameter) and parameter.requires_grad and id(parameter) not in seen_dictionary_parameter_ids:
                dictionary_parameters.append(parameter)
                seen_dictionary_parameter_ids.add(id(parameter))
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in dictionary_ids or not parameter.requires_grad:
            continue
        if parameter_id in head_ids:
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    groups: list[dict[str, Any]] = []
    if coefficient_parameters and float(profile.coefficient_lr) > 0.0:
        groups.append({"params": coefficient_parameters, "lr": profile.coefficient_lr, "name": "coefficient"})
    if dictionary_parameters and float(profile.dictionary_lr) > 0.0:
        groups.append({"params": dictionary_parameters, "lr": profile.dictionary_lr, "name": "dictionary"})
    if backbone_parameters and float(profile.non_dictionary_lr) > 0.0:
        groups.append({"params": backbone_parameters, "lr": profile.non_dictionary_lr, "name": "non_dictionary_backbone"})
    if head_parameters and float(head_lr) > 0.0:
        groups.append({"params": head_parameters, "lr": head_lr, "name": "classification_head"})
    if not groups:
        raise ValueError("No trainable parameters remain after applying learning-rate profile")
    return torch.optim.AdamW(groups, weight_decay=0.0)


# --- Evaluation and route-parity diagnostics --------------------------------
def _empty_forward_support_commit_output_parity_metrics() -> dict[str, Any]:
    return {
        "forward_support_commit_output_parity_status": "not_checked",
        "forward_support_commit_output_parity_sample_count": 0,
        "forward_support_commit_output_parity_max_abs_logit_difference": float("nan"),
        "forward_support_commit_output_parity_mean_abs_logit_difference": float("nan"),
        "forward_support_commit_output_parity_relative_l2_difference": float("nan"),
        "forward_support_commit_output_parity_prediction_mismatch_count": 0,
        "forward_support_commit_output_parity_prediction_mismatch_fraction": float("nan"),
        "forward_support_commit_output_parity_accuracy_before": float("nan"),
        "forward_support_commit_output_parity_accuracy_after": float("nan"),
        "forward_support_commit_output_parity_accuracy_abs_difference": float("nan"),
        "forward_support_commit_output_parity_passed": False,
    }

def _forward_output_parity_metrics(
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
    prediction_mismatch_maximum: int,
    accuracy_difference_maximum: float,
) -> dict[str, Any]:
    """Measure whether hard-support commit changes the represented function."""

    if before_logits.shape != after_logits.shape:
        raise ValueError("Support-commit parity logits must have identical shapes")
    if before_logits.ndim != 2 or labels.ndim != 1 or before_logits.shape[0] != labels.shape[0]:
        raise ValueError("Support-commit parity expects [sample,class] logits and [sample] labels")
    before = before_logits.detach().float()
    after = after_logits.detach().float()
    labels_long = labels.detach().long().to(before.device)
    difference = after - before
    max_abs = float(difference.abs().max().cpu()) if difference.numel() else 0.0
    mean_abs = float(difference.abs().mean().cpu()) if difference.numel() else 0.0
    relative_l2 = float(
        (difference.norm() / before.norm().clamp_min(torch.finfo(before.dtype).eps)).cpu()
    )
    before_prediction = before.argmax(dim=1)
    after_prediction = after.argmax(dim=1)
    mismatch_count = int((before_prediction != after_prediction).sum().cpu())
    sample_count = int(before.shape[0])
    before_accuracy = float((before_prediction == labels_long).float().mean().cpu()) if sample_count else 0.0
    after_accuracy = float((after_prediction == labels_long).float().mean().cpu()) if sample_count else 0.0
    accuracy_difference = abs(after_accuracy - before_accuracy)
    passed = bool(
        max_abs <= float(max_abs_tolerance)
        and relative_l2 <= float(relative_l2_tolerance)
        and mismatch_count <= int(prediction_mismatch_maximum)
        and accuracy_difference <= float(accuracy_difference_maximum)
    )
    return {
        "forward_support_commit_output_parity_status": "passed" if passed else "failed",
        "forward_support_commit_output_parity_sample_count": sample_count,
        "forward_support_commit_output_parity_max_abs_logit_difference": max_abs,
        "forward_support_commit_output_parity_mean_abs_logit_difference": mean_abs,
        "forward_support_commit_output_parity_relative_l2_difference": relative_l2,
        "forward_support_commit_output_parity_prediction_mismatch_count": mismatch_count,
        "forward_support_commit_output_parity_prediction_mismatch_fraction": mismatch_count / max(1, sample_count),
        "forward_support_commit_output_parity_accuracy_before": before_accuracy,
        "forward_support_commit_output_parity_accuracy_after": after_accuracy,
        "forward_support_commit_output_parity_accuracy_abs_difference": accuracy_difference,
        "forward_support_commit_output_parity_passed": passed,
    }

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_correct_logit_margin = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images, labels = batch[0].to(device), batch[1].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            labels_long = labels.long()
            true_logits = logits.gather(1, labels_long.view(-1, 1)).squeeze(1)
            masked_logits = logits.clone()
            masked_logits.scatter_(1, labels_long.view(-1, 1), float("-inf"))
            strongest_other_logits = masked_logits.max(dim=1).values
            margins = true_logits - strongest_other_logits
            total_loss += float(loss.cpu()) * int(labels.numel())
            total_correct += int((logits.argmax(dim=1) == labels).sum().cpu())
            total_correct_logit_margin += float(margins.detach().float().sum().cpu())
            total_count += int(labels.numel())
    return {
        "loss": total_loss / max(1, total_count),
        "accuracy": total_correct / max(1, total_count),
        "correct_logit_margin": total_correct_logit_margin / max(1, total_count),
        "count": float(total_count),
    }

def _empty_routed_hard_gate_eval_metrics() -> dict[str, float | str]:
    return {
        "eval_loss_dynamic_hard_route": "",
        "eval_accuracy_dynamic_hard_route": "",
        "eval_count_dynamic_hard_route": "",
        "eval_loss_fixed_ema_hard_route": "",
        "eval_accuracy_fixed_ema_hard_route": "",
        "eval_count_fixed_ema_hard_route": "",
        "fixed_ema_minus_dynamic_hard_route_accuracy": "",
        "fixed_ema_minus_dynamic_hard_route_loss": "",
        "fixed_ema_support_available": "",
    }

@contextmanager
def temporary_forward_routed_eval_support(model: nn.Module, *, use_ema_support: bool) -> Iterator[None]:
    saved: list[tuple[SeparableDictionaryLinear, bool, bool]] = []
    for _name, layer in iter_dictionary_layers(model):
        if not bool(getattr(layer, "forward_routed_gate_enabled", False)):
            continue
        saved.append((
            layer,
            bool(getattr(layer, "forward_routed_gate_eval_use_ema_support", False)),
            bool(getattr(layer, "forward_routed_gate_straight_through", True)),
        ))
        layer.forward_routed_gate_eval_use_ema_support = bool(use_ema_support)
        layer.forward_routed_gate_straight_through = False
    try:
        yield
    finally:
        for layer, eval_use_ema, straight_through in saved:
            layer.forward_routed_gate_eval_use_ema_support = bool(eval_use_ema)
            layer.forward_routed_gate_straight_through = bool(straight_through)

def routed_hard_gate_eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
    enabled: bool = False,
    dynamic_enabled: bool = True,
    fixed_enabled: bool = True,
    base_eval_metrics: dict[str, float] | None = None,
) -> dict[str, float | str]:
    layers = [layer for _name, layer in iter_dictionary_layers(model) if bool(getattr(layer, "forward_routed_gate_enabled", False))]
    if not bool(enabled) or not layers:
        return _empty_routed_hard_gate_eval_metrics()
    was_training = bool(model.training)
    if bool(dynamic_enabled):
        with temporary_forward_routed_eval_support(model, use_ema_support=False):
            dynamic = evaluate_model(model, loader, device=device, max_batches=max_batches)
    else:
        dynamic = {"loss": "", "accuracy": "", "count": ""}
    fixed_available = bool(fixed_enabled) and any(
        bool(getattr(layer, "_forward_routed_fixed_support_is_initialized", lambda: False)())
        or bool(getattr(layer, "_global_solution_usage_ema_is_initialized", lambda: False)())
        for layer in layers
    )
    fixed_only_base_reusable = (
        fixed_available
        and not bool(dynamic_enabled)
        and base_eval_metrics is not None
        and all(bool(getattr(layer, "forward_routed_gate_eval_use_ema_support", False)) for layer in layers)
    )
    if fixed_only_base_reusable:
        fixed = {
            "loss": base_eval_metrics["loss"],
            "accuracy": base_eval_metrics["accuracy"],
            "count": base_eval_metrics["count"],
        }
    elif fixed_available:
        with temporary_forward_routed_eval_support(model, use_ema_support=True):
            fixed = evaluate_model(model, loader, device=device, max_batches=max_batches)
    else:
        fixed = {"loss": "", "accuracy": "", "count": ""}
    if was_training:
        model.train()
    else:
        model.eval()
    result: dict[str, float | str] = {
        "eval_loss_dynamic_hard_route": dynamic["loss"],
        "eval_accuracy_dynamic_hard_route": dynamic["accuracy"],
        "eval_count_dynamic_hard_route": dynamic["count"],
        "eval_loss_fixed_ema_hard_route": fixed["loss"],
        "eval_accuracy_fixed_ema_hard_route": fixed["accuracy"],
        "eval_count_fixed_ema_hard_route": fixed["count"],
        "fixed_ema_support_available": 1.0 if fixed_available else 0.0,
    }
    if fixed_available and bool(dynamic_enabled):
        result["fixed_ema_minus_dynamic_hard_route_accuracy"] = float(fixed["accuracy"]) - float(dynamic["accuracy"])
        result["fixed_ema_minus_dynamic_hard_route_loss"] = float(fixed["loss"]) - float(dynamic["loss"])
    else:
        result["fixed_ema_minus_dynamic_hard_route_accuracy"] = ""
        result["fixed_ema_minus_dynamic_hard_route_loss"] = ""
    return result


# --- Update and gradient diagnostics ----------------------------------------
def _snapshot_dictionary_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: layer.current_weight().detach().cpu().clone() for name, layer in iter_dictionary_layers(model)}

def _effective_update_ratio(model: nn.Module, previous: dict[str, torch.Tensor]) -> dict[str, float]:
    ratios: list[float] = []
    for name, layer in iter_dictionary_layers(model):
        old_weight = previous.get(name)
        if old_weight is None:
            continue
        current = layer.current_weight().detach().cpu()
        numerator = (current - old_weight).float().norm()
        denominator = old_weight.float().norm().clamp_min(1e-12)
        ratios.append(float((numerator / denominator).cpu()))
    if not ratios:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": sum(ratios) / len(ratios), "max": max(ratios)}

def _snapshot_update_state(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    prefixes = _dictionary_parameter_prefixes(model)
    non_dictionary_parameters: dict[str, torch.Tensor] = {}
    dictionary_bias_parameters: dict[str, torch.Tensor] = {}
    for layer_name, layer in iter_dictionary_layers(model):
        if layer.bias is not None:
            dictionary_bias_parameters[f"{layer_name}.bias"] = layer.bias.detach().cpu().clone()
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            continue
        non_dictionary_parameters[name] = parameter.detach().cpu().clone()
    return {
        "dictionary_weights": _snapshot_dictionary_weights(model),
        "dictionary_bias_parameters": dictionary_bias_parameters,
        "non_dictionary_parameters": non_dictionary_parameters,
    }

def _update_norms(model: nn.Module, snapshot: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    dictionary_weight_delta_sq = 0.0
    dictionary_weight_base_sq = 0.0
    for name, layer in iter_dictionary_layers(model):
        old_weight = snapshot.get("dictionary_weights", {}).get(name)
        if old_weight is None:
            continue
        current = layer.current_weight().detach().cpu().float()
        old_float = old_weight.float()
        dictionary_weight_delta_sq += float((current - old_float).pow(2).sum())
        dictionary_weight_base_sq += float(old_float.pow(2).sum())

    dictionary_bias_delta_sq = 0.0
    dictionary_bias_base_sq = 0.0
    for layer_name, layer in iter_dictionary_layers(model):
        if layer.bias is None:
            continue
        old_bias = snapshot.get("dictionary_bias_parameters", {}).get(f"{layer_name}.bias")
        if old_bias is None:
            continue
        current = layer.bias.detach().cpu().float()
        old_float = old_bias.float()
        dictionary_bias_delta_sq += float((current - old_float).pow(2).sum())
        dictionary_bias_base_sq += float(old_float.pow(2).sum())

    non_dictionary_delta_sq = 0.0
    non_dictionary_base_sq = 0.0
    for name, parameter in model.named_parameters():
        old_parameter = snapshot.get("non_dictionary_parameters", {}).get(name)
        if old_parameter is None:
            continue
        current = parameter.detach().cpu().float()
        old_float = old_parameter.float()
        non_dictionary_delta_sq += float((current - old_float).pow(2).sum())
        non_dictionary_base_sq += float(old_float.pow(2).sum())

    dictionary_weight_update_norm = math.sqrt(dictionary_weight_delta_sq)
    dictionary_bias_update_norm = math.sqrt(dictionary_bias_delta_sq)
    dictionary_update_norm = math.sqrt(dictionary_weight_delta_sq + dictionary_bias_delta_sq)
    dictionary_weight_base_norm = math.sqrt(dictionary_weight_base_sq)
    dictionary_bias_base_norm = math.sqrt(dictionary_bias_base_sq)
    dictionary_base_norm = math.sqrt(dictionary_weight_base_sq + dictionary_bias_base_sq)
    non_dictionary_update_norm = math.sqrt(non_dictionary_delta_sq)
    non_dictionary_base_norm = math.sqrt(non_dictionary_base_sq)
    return {
        "dictionary_weight_update_norm": dictionary_weight_update_norm,
        "dictionary_weight_update_ratio": dictionary_weight_update_norm / max(1e-12, dictionary_weight_base_norm),
        "dictionary_bias_update_norm": dictionary_bias_update_norm,
        "dictionary_bias_update_ratio": dictionary_bias_update_norm / max(1e-12, dictionary_bias_base_norm),
        "dictionary_update_norm": dictionary_update_norm,
        "non_dictionary_update_norm": non_dictionary_update_norm,
        "dictionary_update_ratio": dictionary_update_norm / max(1e-12, dictionary_base_norm),
        "non_dictionary_update_ratio": non_dictionary_update_norm / max(1e-12, non_dictionary_base_norm),
        "dictionary_to_non_dictionary_update_ratio": dictionary_update_norm / max(1e-12, non_dictionary_update_norm),
        "dictionary_weight_to_non_dictionary_update_ratio": dictionary_weight_update_norm / max(1e-12, non_dictionary_update_norm),
    }

def _coefficient_support_grad_norms(model: nn.Module) -> dict[str, float]:
    """Split coefficient gradients by the currently hard-committed support mask.

    When no coefficient support is committed, all coefficient gradient is treated
    as active-support gradient. A copied source-prior mask that has been cleared
    for target-side relearn is not treated as hard inactive support.
    """

    active_sq: torch.Tensor | None = None
    inactive_sq: torch.Tensor | None = None
    for _layer_name, layer in iter_dictionary_layers(model):
        if layer.coefficient_support_is_committed():
            keep = layer.coefficient_commit_mask.detach().bool()
        else:
            keep = None
        for coefficient in (layer.coefficient_magnitude,):
            if coefficient.grad is None:
                continue
            grad = coefficient.grad.detach().float()
            if keep is None:
                active_sq = _add_grad_sq(active_sq, grad.pow(2).sum())
                continue
            keep_on_grad = keep.to(device=grad.device)
            if tuple(keep_on_grad.shape) != tuple(grad.shape):
                active_sq = _add_grad_sq(active_sq, grad.pow(2).sum())
                continue
            active_sq = _add_grad_sq(active_sq, grad.masked_select(keep_on_grad).pow(2).sum())
            inactive_sq = _add_grad_sq(inactive_sq, grad.masked_select(~keep_on_grad).pow(2).sum())
    return {
        "coefficient_active_support_grad_norm": _sqrt_tensor_sum(active_sq),
        "coefficient_inactive_support_grad_norm": _sqrt_tensor_sum(inactive_sq),
    }

def _zero_inactive_coefficient_gradients(model: nn.Module) -> int:
    """Remove gradients that cannot survive a hard-committed support mask.

    This is only a fixed-support optimization. Before source/independent hard
    commit there is no committed mask, so inactive coefficients remain eligible
    candidates and their gradients are preserved.
    """

    zeroed = 0
    for _layer_name, layer in iter_dictionary_layers(model):
        if not layer.coefficient_support_is_committed():
            continue
        keep = layer.coefficient_commit_mask.detach().bool()
        for coefficient in (layer.coefficient_magnitude,):
            if coefficient.grad is None:
                continue
            keep_on_grad = keep.to(device=coefficient.grad.device)
            if tuple(keep_on_grad.shape) != tuple(coefficient.grad.shape):
                continue
            inactive = ~keep_on_grad
            coefficient.grad.masked_fill_(inactive, 0.0)
            zeroed += 1
    return zeroed

def _zero_clip_metrics() -> dict[str, float]:
    return {
        "coefficient_grad_norm_pre_clip": 0.0,
        "coefficient_active_support_grad_norm_pre_clip": 0.0,
        "coefficient_inactive_support_grad_norm_pre_clip": 0.0,
        "coefficient_grad_norm_post_clip": 0.0,
        "coefficient_active_support_grad_norm_post_clip": 0.0,
        "coefficient_inactive_support_grad_norm_post_clip": 0.0,
        "non_coefficient_grad_norm_pre_clip": 0.0,
        "non_coefficient_grad_norm_post_clip": 0.0,
        "global_grad_norm_pre_clip": 0.0,
        "global_grad_norm_post_clip": 0.0,
    }

def _support_split_fields(prefix: str, support_norms: dict[str, float]) -> dict[str, float]:
    return {
        f"coefficient_active_support_grad_norm_{prefix}": float(support_norms.get("coefficient_active_support_grad_norm", 0.0)),
        f"coefficient_inactive_support_grad_norm_{prefix}": float(support_norms.get("coefficient_inactive_support_grad_norm", 0.0)),
    }

def _clip_gradients(
    model: nn.Module,
    profile_config: dict[str, Any] | None = None,
    *,
    measure: bool = True,
) -> dict[str, float]:
    config = profile_config or {}
    zero_metrics = _zero_clip_metrics()
    enabled = bool(config.get("enabled", True))
    mode = str(config.get("mode", "global"))
    zero_inactive_before_clip = bool(config.get("zero_inactive_coefficient_grad_before_clip", False))

    # Fast path for ordinary non-measured epochs: avoid building coefficient id
    # sets or active/inactive support splits unless the clip mode or fixed-support
    # inactive-gradient zeroing needs them.
    if not measure and not zero_inactive_before_clip:
        if not enabled or mode == "none":
            return zero_metrics
        if mode != "separate_coefficient":
            max_norm = float(config.get("max_norm", 1.0))
            if max_norm > 0.0:
                grad_parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
                if grad_parameters:
                    torch.nn.utils.clip_grad_norm_(
                        grad_parameters, max_norm=max_norm, error_if_nonfinite=True
                    )
            return zero_metrics

    coefficient_ids = _coefficient_parameter_id_set(model)
    coefficient_parameters = [parameter for parameter in model.parameters() if id(parameter) in coefficient_ids and parameter.grad is not None]
    non_coefficient_parameters = [parameter for parameter in model.parameters() if id(parameter) not in coefficient_ids and parameter.grad is not None]
    all_parameters = coefficient_parameters + non_coefficient_parameters

    if not enabled:
        support_pre = _coefficient_support_grad_norms(model)
        coefficient_pre = _parameter_grad_norm(coefficient_parameters)
        non_coefficient_pre = _parameter_grad_norm(non_coefficient_parameters)
        global_pre = _parameter_grad_norm(all_parameters)
        return {
            "coefficient_grad_norm_pre_clip": coefficient_pre,
            **_support_split_fields("pre_clip", support_pre),
            "coefficient_grad_norm_post_clip": coefficient_pre,
            **_support_split_fields("post_clip", support_pre),
            "non_coefficient_grad_norm_pre_clip": non_coefficient_pre,
            "non_coefficient_grad_norm_post_clip": non_coefficient_pre,
            "global_grad_norm_pre_clip": global_pre,
            "global_grad_norm_post_clip": global_pre,
        }

    coefficient_pre = _parameter_grad_norm(coefficient_parameters) if measure else 0.0
    support_pre = _coefficient_support_grad_norms(model) if measure else {"coefficient_active_support_grad_norm": 0.0, "coefficient_inactive_support_grad_norm": 0.0}
    non_coefficient_pre = _parameter_grad_norm(non_coefficient_parameters) if measure else 0.0
    global_pre = _parameter_grad_norm(all_parameters) if measure else 0.0
    if zero_inactive_before_clip:
        _zero_inactive_coefficient_gradients(model)
    if mode == "separate_coefficient":
        coefficient_max_norm = float(config.get("coefficient_max_norm", config.get("max_norm", 1.0)))
        non_coefficient_max_norm = float(config.get("non_coefficient_max_norm", config.get("max_norm", 1.0)))
        if coefficient_parameters and coefficient_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                coefficient_parameters, max_norm=coefficient_max_norm, error_if_nonfinite=True
            )
        if non_coefficient_parameters and non_coefficient_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                non_coefficient_parameters, max_norm=non_coefficient_max_norm, error_if_nonfinite=True
            )
    elif mode == "none":
        pass
    else:
        max_norm = float(config.get("max_norm", 1.0))
        if all_parameters and max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                all_parameters, max_norm=max_norm, error_if_nonfinite=True
            )
    if not measure:
        return zero_metrics
    coefficient_post = _parameter_grad_norm(coefficient_parameters)
    support_post = _coefficient_support_grad_norms(model)
    non_coefficient_post = _parameter_grad_norm(non_coefficient_parameters)
    global_post = _parameter_grad_norm(all_parameters)
    return {
        "coefficient_grad_norm_pre_clip": coefficient_pre,
        **_support_split_fields("pre_clip", support_pre),
        "coefficient_grad_norm_post_clip": coefficient_post,
        **_support_split_fields("post_clip", support_post),
        "non_coefficient_grad_norm_pre_clip": non_coefficient_pre,
        "non_coefficient_grad_norm_post_clip": non_coefficient_post,
        "global_grad_norm_pre_clip": global_pre,
        "global_grad_norm_post_clip": global_post,
    }

def _snapshot_coefficient_vectors(model: nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, layer in iter_dictionary_layers(model):
        snapshot[name] = (layer.coefficient_magnitude).detach().cpu().float().clone()
    return snapshot

def _coefficient_update_dynamics(
    before: dict[str, torch.Tensor],
    after_optimizer: dict[str, torch.Tensor],
    after_projection: dict[str, torch.Tensor],
) -> dict[str, float]:
    final_delta_sq = 0.0
    optimizer_delta_sq = 0.0
    erased_delta_sq = 0.0
    base_sq = 0.0
    radial_sq = 0.0
    tangential_sq = 0.0
    for name, before_tensor in before.items():
        optimizer_tensor = after_optimizer.get(name)
        final_tensor = after_projection.get(name)
        if optimizer_tensor is None or final_tensor is None:
            continue
        before_flat = before_tensor.flatten().float()
        optimizer_delta = (optimizer_tensor.flatten().float() - before_flat)
        final_delta = (final_tensor.flatten().float() - before_flat)
        erased_delta = final_tensor.flatten().float() - optimizer_tensor.flatten().float()
        base_norm_sq_tensor = before_flat.pow(2).sum().clamp_min(1e-24)
        projection_scale = torch.dot(final_delta, before_flat) / base_norm_sq_tensor
        radial_delta = projection_scale * before_flat
        tangential_delta = final_delta - radial_delta
        final_delta_sq += float(final_delta.pow(2).sum().cpu())
        optimizer_delta_sq += float(optimizer_delta.pow(2).sum().cpu())
        erased_delta_sq += float(erased_delta.pow(2).sum().cpu())
        base_sq += float(before_flat.pow(2).sum().cpu())
        radial_sq += float(radial_delta.pow(2).sum().cpu())
        tangential_sq += float(tangential_delta.pow(2).sum().cpu())
    update_norm = math.sqrt(final_delta_sq)
    optimizer_update_norm = math.sqrt(optimizer_delta_sq)
    base_norm = math.sqrt(base_sq)
    tangential_norm = math.sqrt(tangential_sq)
    return {
        "coefficient_update_norm": update_norm,
        "coefficient_update_ratio": update_norm / max(1e-12, base_norm),
        "coefficient_optimizer_update_norm": optimizer_update_norm,
        "coefficient_optimizer_update_ratio": optimizer_update_norm / max(1e-12, base_norm),
        "coefficient_projection_erased_update_norm": math.sqrt(erased_delta_sq),
        "coefficient_radial_update_norm": math.sqrt(radial_sq),
        "coefficient_tangential_update_norm": tangential_norm,
        "coefficient_tangential_update_ratio": tangential_norm / max(1e-12, update_norm),
    }

def _coefficient_epoch_end_sparse_event_dynamics(
    before_event: dict[str, torch.Tensor],
    after_event: dict[str, torch.Tensor],
) -> dict[str, float]:
    update_sq = 0.0
    base_sq = 0.0
    radial_sq = 0.0
    tangential_sq = 0.0
    for name, before_tensor in before_event.items():
        after_tensor = after_event.get(name)
        if after_tensor is None:
            continue
        before_flat = before_tensor.flatten().float()
        final_delta = after_tensor.flatten().float() - before_flat
        base_norm_sq_tensor = before_flat.pow(2).sum().clamp_min(1e-24)
        projection_scale = torch.dot(final_delta, before_flat) / base_norm_sq_tensor
        radial_delta = projection_scale * before_flat
        tangential_delta = final_delta - radial_delta
        update_sq += float(final_delta.pow(2).sum().cpu())
        base_sq += float(before_flat.pow(2).sum().cpu())
        radial_sq += float(radial_delta.pow(2).sum().cpu())
        tangential_sq += float(tangential_delta.pow(2).sum().cpu())
    update_norm = math.sqrt(update_sq)
    base_norm = math.sqrt(base_sq)
    tangential_norm = math.sqrt(tangential_sq)
    return {
        "coefficient_epoch_end_sparse_event_update_norm": update_norm,
        "coefficient_epoch_end_sparse_event_update_ratio": update_norm / max(1e-12, base_norm),
        "coefficient_epoch_end_sparse_event_projection_erased_update_norm": update_norm,
        "coefficient_epoch_end_sparse_event_mask_rewrite_update_norm": update_norm,
        "coefficient_epoch_end_sparse_event_radial_update_norm": math.sqrt(radial_sq),
        "coefficient_epoch_end_sparse_event_tangential_update_norm": tangential_norm,
        "coefficient_epoch_end_sparse_event_tangential_update_ratio": tangential_norm / max(1e-12, update_norm),
    }

def _mean_step_metrics(
    step_metrics: Sequence[dict[str,
    float]],
    field_names: Sequence[str],
) -> dict[str, float]:
    if not step_metrics:
        return {name: 0.0 for name in field_names}
    result: dict[str, float] = {}
    for name in field_names:
        values = [float(item.get(name, 0.0)) for item in step_metrics]
        result[name] = sum(values) / max(1, len(values))
    return result

def _coefficient_reference_metrics(
    layer_name: str,
    coeff: torch.Tensor,
    reference_snapshot: dict[str, torch.Tensor] | None,
    *,
    top16_k: int = 16,
    top64_k: int = 64,
) -> dict[str, float | str]:
    if not reference_snapshot or layer_name not in reference_snapshot:
        return {
            "epoch0_top16_overlap_ratio": "",
            "epoch0_top16_turnover_ratio": "",
            "epoch0_top64_overlap_ratio": "",
            "epoch0_top64_turnover_ratio": "",
            "epoch0_top64_rank_correlation": "",
            "epoch0_active_rank_correlation": "",
        }
    reference = reference_snapshot[layer_name].to(coeff.device).float().flatten().abs()
    current = coeff.detach().float().flatten().abs()
    if int(reference.numel()) != int(current.numel()):
        return {
            "epoch0_top16_overlap_ratio": "",
            "epoch0_top16_turnover_ratio": "",
            "epoch0_top64_overlap_ratio": "",
            "epoch0_top64_turnover_ratio": "",
            "epoch0_top64_rank_correlation": "",
            "epoch0_active_rank_correlation": "",
        }

    def _top_set(values: torch.Tensor, k: int) -> set[int]:
        count = min(int(k), int(values.numel()))
        if count <= 0:
            return set()
        return set(int(index) for index in _topk_indices_1d(values, count).detach().cpu().tolist())

    ref_top16 = _top_set(reference, top16_k)
    cur_top16 = _top_set(current, top16_k)
    ref_top64 = _top_set(reference, top64_k)
    cur_top64 = _top_set(current, top64_k)
    top16_overlap = len(ref_top16 & cur_top16) / max(1, min(len(ref_top16), len(cur_top16)))
    top64_overlap = len(ref_top64 & cur_top64) / max(1, min(len(ref_top64), len(cur_top64)))
    top64_rank_mask = torch.zeros_like(reference, dtype=torch.bool)
    for index in ref_top64 | cur_top64:
        top64_rank_mask[int(index)] = True
    top64_rank_correlation = _rank_correlation_for_abs_values(reference, current, top64_rank_mask)
    return {
        "epoch0_top16_overlap_ratio": top16_overlap,
        "epoch0_top16_turnover_ratio": 1.0 - top16_overlap,
        "epoch0_top64_overlap_ratio": top64_overlap,
        "epoch0_top64_turnover_ratio": 1.0 - top64_overlap,
        "epoch0_top64_rank_correlation": top64_rank_correlation,
        "epoch0_active_rank_correlation": top64_rank_correlation,
    }

def _gini_from_nonnegative_mass(mass: torch.Tensor) -> float:
    values = mass.detach().float().flatten()
    if int(values.numel()) <= 0 or float(values.sum().cpu()) <= 0.0:
        return 0.0
    sorted_values, _ = torch.sort(values)
    index = torch.arange(1, int(sorted_values.numel()) + 1, device=sorted_values.device, dtype=sorted_values.dtype)
    return float(
        ((2.0 * index - float(sorted_values.numel()) - 1.0) * sorted_values)
        .sum()
        .div(float(sorted_values.numel()) * sorted_values.sum().clamp_min(1e-12))
        .cpu()
    )

def _p95_from_counts(counts: Sequence[torch.Tensor]) -> float:
    tensors = [item.detach().float().flatten().cpu() for item in counts if int(item.numel()) > 0]
    if not tensors:
        return 0.0
    values = torch.cat(tensors)
    if int(values.numel()) <= 0:
        return 0.0
    sorted_values, _ = torch.sort(values)
    index = int(math.ceil(0.95 * float(int(sorted_values.numel()))) - 1)
    index = max(0, min(index, int(sorted_values.numel()) - 1))
    return float(sorted_values[index].cpu())


# --- Activation-aware contribution reports ----------------------------------
def _begin_attention_activation_rms_measurement(model: nn.Module) -> None:
    for module in model.modules():
        begin = getattr(module, "begin_dictionary_attention_rms_measurement_", None)
        if callable(begin) and hasattr(module, "dictionary_qk_log_scale") and hasattr(module, "dictionary_vo_log_scale"):
            begin()

def _finish_attention_activation_rms_measurement(model: nn.Module) -> dict[str, Any]:
    qk_pre: list[float] = []
    qk_post: list[float] = []
    vo_pre: list[float] = []
    vo_post: list[float] = []
    qk_pre_per_block: list[str] = []
    qk_post_per_block: list[str] = []
    vo_pre_per_block: list[str] = []
    vo_post_per_block: list[str] = []
    for module_name, module in model.named_modules():
        end = getattr(module, "end_dictionary_attention_rms_measurement_", None)
        if not callable(end):
            continue
        measured = end()
        if not isinstance(measured, dict) or not measured:
            continue
        qk_pre_value = _scalar_report_float(measured.get("qk_logits_pre_scale_rms"))
        qk_post_value = _scalar_report_float(measured.get("qk_logits_post_scale_rms"))
        vo_pre_value = _scalar_report_float(measured.get("vo_output_pre_scale_rms"))
        vo_post_value = _scalar_report_float(measured.get("vo_output_post_scale_rms"))
        qk_pre.append(qk_pre_value)
        qk_post.append(qk_post_value)
        vo_pre.append(vo_pre_value)
        vo_post.append(vo_post_value)
        qk_pre_per_block.append(f"{module_name}:{qk_pre_value:.9g}")
        qk_post_per_block.append(f"{module_name}:{qk_post_value:.9g}")
        vo_pre_per_block.append(f"{module_name}:{vo_pre_value:.9g}")
        vo_post_per_block.append(f"{module_name}:{vo_post_value:.9g}")

    qk_ratio = [post / max(pre, 1e-12) for pre, post in zip(qk_pre, qk_post)]
    vo_ratio = [post / max(pre, 1e-12) for pre, post in zip(vo_pre, vo_post)]
    return {
        **_min_mean_max_metrics(qk_pre, "attention_qk_logits_rms_pre_scale"),
        **_min_mean_max_metrics(qk_post, "attention_qk_logits_rms_post_scale"),
        **_min_mean_max_metrics(qk_ratio, "attention_qk_logits_rms_scale_ratio"),
        "attention_qk_logits_rms_pre_scale_per_block": ";".join(qk_pre_per_block),
        "attention_qk_logits_rms_post_scale_per_block": ";".join(qk_post_per_block),
        **_min_mean_max_metrics(vo_pre, "attention_vo_output_rms_pre_scale"),
        **_min_mean_max_metrics(vo_post, "attention_vo_output_rms_post_scale"),
        **_min_mean_max_metrics(vo_ratio, "attention_vo_output_rms_scale_ratio"),
        "attention_vo_output_rms_pre_scale_per_block": ";".join(vo_pre_per_block),
        "attention_vo_output_rms_post_scale_per_block": ";".join(vo_post_per_block),
    }

@torch.no_grad()
def activation_aware_contribution_metrics_by_layer(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
    threshold: float = 1e-3,
    mass_target: float = 0.95,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Measure post-route atom contribution and actual attention activation RMS.

    Layers accumulate detached contribution mass after the real hard route and
    optional forward-support mask have been applied. Accumulators stay on the
    model device during the bounded eval pass and transfer only once when report
    rows are finalized.
    """

    layers = list(iter_dictionary_layers(model))
    if not layers:
        return {}, {}
    for _name, layer in layers:
        begin = getattr(layer, "begin_activation_contribution_measurement_", None)
        if not callable(begin):
            raise RuntimeError("dictionary layer lacks post-route contribution measurement support")
        begin(threshold=float(threshold))

    _begin_attention_activation_rms_measurement(model)
    attention_activation_metrics: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    was_training = bool(model.training)
    model.eval()
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            images = batch[0].to(device)
            _ = model(images)
    finally:
        for name, layer in layers:
            finish = getattr(layer, "end_activation_contribution_measurement_", None)
            state = finish() if callable(finish) else None
            if isinstance(state, dict):
                states[name] = state
        attention_activation_metrics = _finish_attention_activation_rms_measurement(model)
        if was_training:
            model.train()
        else:
            model.eval()

    results: dict[str, dict[str, Any]] = {}
    layer_by_name = {name: layer for name, layer in layers}
    for layer_name, state in states.items():
        mass_sum = state.get("mass_sum")
        sample_count = int(state.get("sample_count", 0) or 0)
        if not isinstance(mass_sum, torch.Tensor) or sample_count <= 0:
            continue
        mass = mass_sum.detach().float() / float(max(1, sample_count))
        hard_count, mass95_count, top100_mass = _contribution_support_counts_from_mass(
            mass,
            threshold=float(threshold),
            mass_target=float(mass_target),
        )
        probability = mass / mass.sum().clamp_min(1e-12)
        hard_mask = probability >= float(threshold)
        entropy = float((-(probability * probability.clamp_min(1e-12).log()).sum()).cpu())
        gini = _gini_from_nonnegative_mass(mass)
        layer = layer_by_name[layer_name]
        probability_cpu = probability.detach().cpu()
        hard_mask_cpu = hard_mask.detach().cpu()
        p95_sample = _p95_from_counts(state.get("sample_hard_counts", []))

        support_sample_count = state.get("support_sample_count")
        c_sum = state.get("c_sum")
        abs_c_sum = state.get("abs_c_sum")
        if not all(isinstance(item, torch.Tensor) for item in (support_sample_count, c_sum, abs_c_sum)):
            raise RuntimeError("post-route contribution measurement state is incomplete")
        support_sample_count_cpu = support_sample_count.detach().float().cpu()
        measured_support = support_sample_count_cpu > 0.0
        active_indices = torch.nonzero(measured_support, as_tuple=False).flatten()
        c_mean = c_sum.detach().float().cpu() / support_sample_count_cpu.clamp_min(1.0)
        abs_c_mean = abs_c_sum.detach().float().cpu() / support_sample_count_cpu.clamp_min(1.0)
        active_code = c_mean[measured_support]
        active_abs_code = abs_c_mean[measured_support]
        active_contribution_probability = probability_cpu[measured_support]
        active_contribution_probability = (
            active_contribution_probability
            / active_contribution_probability.sum().clamp_min(1e-12)
        )
        ratio_tolerance = max(
            1e-6,
            float(getattr(layer, "relative_coefficient_measurement_tolerance", 1e-3)),
        )
        near_min = active_abs_code <= float(layer.relative_coefficient_min_ratio) + ratio_tolerance
        near_max = active_abs_code >= float(layer.relative_coefficient_max_ratio) - ratio_tolerance
        non_min = ~near_min
        if int(active_abs_code.numel()) > 1:
            c_centered = active_abs_code - active_abs_code.mean()
            contribution_centered = active_contribution_probability - active_contribution_probability.mean()
            pearson = float(
                (c_centered * contribution_centered).sum().div(
                    (c_centered.norm() * contribution_centered.norm()).clamp_min(1e-12)
                )
            )
            relative_c_spearman_value = _spearman_correlation(
                active_abs_code, active_contribution_probability
            )
            relative_c_spearman = (
                1.0 if relative_c_spearman_value is None
                else float(relative_c_spearman_value)
            )
        else:
            pearson = 1.0
            relative_c_spearman = 1.0
        pair_payload = ";".join(
            f"{int(index)}:{float(code):.8g}:{float(contribution):.8g}"
            for index, code, contribution in zip(
                active_indices.tolist(),
                active_code.tolist(),
                active_contribution_probability.tolist(),
            )
        )
        metrics = {
            "contribution_metric_source": "activation_aware_post_route_eval_batch",
            "contribution_hard_active_atoms": float(hard_count),
            "contribution_mass95_atoms": float(mass95_count),
            "contribution_p95_sample_hard_active_atoms": float(p95_sample),
            "contribution_top100_mass_ratio": float(top100_mass),
            "contribution_entropy": entropy,
            "contribution_gini": gini,
            "activation_contribution_hard_active_atoms": float(hard_count),
            "activation_contribution_mass95_atoms": float(mass95_count),
            "activation_contribution_p95_sample_hard_active_atoms": float(p95_sample),
            "activation_contribution_top100_mass_ratio": float(top100_mass),
            "activation_contribution_entropy": entropy,
            "activation_contribution_gini": gini,
            "relative_c_actual_support_atom_count": float(active_abs_code.numel()),
            "relative_c_active_abs_mean": float(active_abs_code.mean()) if int(active_abs_code.numel()) > 0 else 0.0,
            "relative_c_active_abs_max": float(active_abs_code.max()) if int(active_abs_code.numel()) > 0 else 0.0,
            "relative_c_normalized_at_or_below_raw_min_atom_count": float(near_min.sum()),
            "relative_c_normalized_at_or_below_raw_min_atom_ratio": float(near_min.float().mean()) if int(near_min.numel()) > 0 else 0.0,
            "relative_c_normalized_at_or_above_raw_max_atom_count": float(near_max.sum()),
            "relative_c_normalized_at_or_above_raw_max_atom_ratio": float(near_max.float().mean()) if int(near_max.numel()) > 0 else 0.0,
            "relative_c_normalized_at_or_below_raw_min_contribution_mass_ratio": float(active_contribution_probability[near_min].sum()) if int(near_min.numel()) > 0 else 0.0,
            "relative_c_normalized_above_raw_min_contribution_mass_ratio": float(active_contribution_probability[non_min].sum()) if int(non_min.numel()) > 0 else 0.0,
            "relative_c_normalized_above_raw_min_abs_mean": float(active_abs_code[non_min].mean()) if bool(non_min.any()) else 0.0,
            "relative_c_abs_contribution_pearson": pearson,
            "relative_c_abs_contribution_spearman": relative_c_spearman,
            "relative_c_active_atom_c_contribution_pairs": pair_payload,
        }
        results[layer_name] = metrics
    return results, attention_activation_metrics

def collect_usage_rows(
    model: nn.Module,
    *,
    run_id: str,
    epoch: int,
    global_step: int,
    task_id: str,
    model_family: str,
    basis_type: str,
    coefficient_reference_snapshot: dict[str, torch.Tensor] | None = None,
    activation_contribution_metrics: dict[str, dict[str, Any]] | None = None,
    dictionary_entropy_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    activation_contribution_metrics = activation_contribution_metrics or {}
    for name, layer in iter_dictionary_layers(model):
        metrics = layer.contribution_metrics(dictionary_entropy_config=dictionary_entropy_config)
        metrics.update(activation_contribution_metrics.get(name, {}))
        coeff = layer.coefficient_magnitude
        metrics.update(_coefficient_reference_metrics(name, coeff, coefficient_reference_snapshot))
        rows.append(
            {
                "run_id": run_id,
                "usage_row_scope": "full_record_epoch",
                "task_id": task_id,
                "epoch": epoch,
                "global_step": global_step,
                "model_family": model_family,
                "basis_type": basis_type,
                "layer_basis_type": str(layer.basis_type),
                "layer_name": name,
                "block_index": _block_index_from_layer_name(name),
                "ffn_layer_kind": _ffn_layer_kind_from_layer_name(name),
                **metrics,
            }
        )
    return rows

def collect_raw_relative_c_epoch_rows(
    model: nn.Module,
    *,
    run_id: str,
    epoch: int,
    global_step: int,
    task_id: str,
    model_family: str,
    basis_type: str,
) -> list[dict[str, Any]]:
    """Collect per-layer raw-C concentration without running evaluation."""

    rows: list[dict[str, Any]] = []
    for name, layer in iter_dictionary_layers(model):
        rows.append(
            {
                "run_id": run_id,
                "usage_row_scope": "raw_c_epoch",
                "task_id": task_id,
                "epoch": epoch,
                "global_step": global_step,
                "model_family": model_family,
                "basis_type": basis_type,
                "layer_basis_type": str(layer.basis_type),
                "layer_name": name,
                "block_index": _block_index_from_layer_name(name),
                "ffn_layer_kind": _ffn_layer_kind_from_layer_name(name),
                **layer.raw_relative_c_concentration_metrics(),
            }
        )
    return rows


# --- Numerical guards -------------------------------------------------------
def _gradient_clip_config_with_zero_inactive_allowed(
    profile_config: dict[str, Any] | None,
    *,
    zero_inactive_allowed: bool,
) -> dict[str, Any] | None:
    if zero_inactive_allowed or not bool((profile_config or {}).get("zero_inactive_coefficient_grad_before_clip", False)):
        return profile_config
    adjusted = dict(profile_config or {})
    adjusted["zero_inactive_coefficient_grad_before_clip"] = False
    return adjusted

def _numerical_guard_enabled(config: dict[str, Any] | None) -> bool:
    return bool((config or {}).get("enabled", False))

def _raise_if_nonfinite_loss(
    loss: torch.Tensor,
    guard_config: dict[str, Any] | None,
    *,
    run_id: str,
    epoch: int,
    batch_index: int,
    phase: str,
) -> None:
    if not _numerical_guard_enabled(guard_config) or not bool((guard_config or {}).get("fail_fast_nonfinite_loss", True)):
        return
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError(
            f"non-finite loss detected during {phase}: "
            f"run_id={run_id} epoch={int(epoch)} batch_index={int(batch_index)} loss={loss.detach()}"
        )

def _coefficient_scale_guard(
    model: nn.Module,
    guard_config: dict[str, Any] | None,
    *,
    run_id: str,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    """Check coefficient finiteness and scale at epoch boundaries.

    The guard stays boundary-scoped to avoid coefficient-wide device
    synchronization on every optimizer step.
    """

    if not _numerical_guard_enabled(guard_config):
        return {"coefficient_guard_max_abs": 0.0, "coefficient_guard_rms_max": 0.0}
    max_abs_limit = float((guard_config or {}).get("max_coefficient_abs", float("inf")))
    rms_limit = float((guard_config or {}).get("max_coefficient_rms", float("inf")))
    max_abs_seen = 0.0
    rms_seen = 0.0
    for layer_name, layer in iter_dictionary_layers(model):
        coeff = layer.coefficient_magnitude
        detached = coeff.detach().float()
        finite = torch.isfinite(detached).all()
        if not bool(finite.item()):
            raise FloatingPointError(
                f"non-finite coefficient detected during {phase}: "
                f"run_id={run_id} epoch={int(epoch)} layer={layer_name}"
            )
        max_abs = float(detached.abs().max().cpu()) if detached.numel() else 0.0
        rms = float(detached.pow(2).mean().sqrt().cpu()) if detached.numel() else 0.0
        max_abs_seen = max(max_abs_seen, max_abs)
        rms_seen = max(rms_seen, rms)
        if math.isfinite(max_abs_limit) and max_abs > max_abs_limit:
            raise FloatingPointError(
                f"coefficient max-abs guard exceeded during {phase}: "
                f"run_id={run_id} epoch={int(epoch)} layer={layer_name} "
                f"max_abs={max_abs:.6g} limit={max_abs_limit:.6g}"
            )
        if math.isfinite(rms_limit) and rms > rms_limit:
            raise FloatingPointError(
                f"coefficient RMS guard exceeded during {phase}: "
                f"run_id={run_id} epoch={int(epoch)} layer={layer_name} "
                f"rms={rms:.6g} limit={rms_limit:.6g}"
            )
    return {"coefficient_guard_max_abs": max_abs_seen, "coefficient_guard_rms_max": rms_seen}
