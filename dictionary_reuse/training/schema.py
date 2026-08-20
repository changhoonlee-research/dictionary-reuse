"""Shared release schemas, runtime state, logging, and output field definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from torch.utils.data import Subset

def _min_mean_max_metrics(values: Sequence[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_min": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": float(sum(values) / len(values)),
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }

TRAINING_IMPLEMENTATION_VERSION = "dir_training_v1"

DENSE_MODEL_FAMILY = "dense_vit"

def _is_dense_model_family(model_family: str) -> bool:
    """Return whether ``model_family`` selects the standard dense ViT control."""

    return str(model_family) == DENSE_MODEL_FAMILY

@dataclass(frozen=True)
class LearningRateProfile:
    name: str
    coefficient_lr: float
    dictionary_lr: float
    non_dictionary_lr: float
    head_lr: float | None = None

@dataclass(frozen=True)
class RunRecord:
    run_id: str
    model_family: str
    profile: str
    seed: int
    basis_type: str
    coefficient_quantization_profile: str = ""
    natural_sparsity_profile: str = ""
    gradient_clip_profile: str = ""
    phase_schedule_profile: str = ""
    data_order_seed: int = 0


COEFFICIENT_DYNAMICS_FIELDS = [
    "coefficient_grad_norm_pre_clip",
    "coefficient_active_support_grad_norm_pre_clip",
    "coefficient_inactive_support_grad_norm_pre_clip",
    "coefficient_grad_norm_post_clip",
    "coefficient_active_support_grad_norm_post_clip",
    "coefficient_inactive_support_grad_norm_post_clip",
    "non_coefficient_grad_norm_pre_clip",
    "non_coefficient_grad_norm_post_clip",
    "global_grad_norm_pre_clip",
    "global_grad_norm_post_clip",
    "coefficient_update_norm",
    "coefficient_update_ratio",
    "coefficient_optimizer_update_norm",
    "coefficient_optimizer_update_ratio",
    "coefficient_projection_erased_update_norm",
    "coefficient_radial_update_norm",
    "coefficient_tangential_update_norm",
    "coefficient_tangential_update_ratio",
]

COEFFICIENT_EPOCH_EVENT_DYNAMICS_FIELDS = [
    "coefficient_epoch_end_sparse_event_update_norm",
    "coefficient_epoch_end_sparse_event_update_ratio",
    "coefficient_epoch_end_sparse_event_projection_erased_update_norm",
    "coefficient_epoch_end_sparse_event_mask_rewrite_update_norm",
    "coefficient_epoch_end_sparse_event_radial_update_norm",
    "coefficient_epoch_end_sparse_event_tangential_update_norm",
    "coefficient_epoch_end_sparse_event_tangential_update_ratio",
]

_DATASET_CACHE: dict[str, Any] = {}

_TASK_SUBSET_CACHE: dict[str, Subset] = {}



def _runtime_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    runtime = config.get("runtime", {})
    return runtime if isinstance(runtime, dict) else {}

def _console_logging_enabled(config: dict[str, Any] | None) -> bool:
    return bool(_runtime_config(config).get("console_logging_enabled", False))

def _console_log(config: dict[str, Any] | None, message: str) -> None:
    if _console_logging_enabled(config):
        print(message, flush=True)

def _public_training_phase(run_id: str, task_id: str) -> str:
    run_value = str(run_id)
    family = "Dense" if run_value.startswith("dense_") else "DiR" if run_value.startswith("dir_") else ""
    if "source" in run_value:
        phase = "SOURCE TASK1"
    elif run_value in {"dir_same_task", "dense_same_task"}:
        phase = "SAME TASK1"
    elif run_value == "dir_dictionary_fixed":
        phase = "DICTIONARY-FIXED TASK2"
    elif run_value == "dir_dictionary_trainable":
        phase = "DICTIONARY-TRAINABLE TASK2"
    elif run_value == "dense_different_task":
        phase = "FULL-TRANSFER TASK2"
    else:
        phase = str(task_id).upper()
    return f"{family} {phase}".strip()

_DICTIONARY_PARAMETER_SUFFIXES = (
    ".dictionary_qk_log_scale",
    ".dictionary_vo_log_scale",
    ".dictionary_log_scale",
    ".coefficient_magnitude",
    ".row_atoms",
    ".col_atoms",
    ".bias",
)

