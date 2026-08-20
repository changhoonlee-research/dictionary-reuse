"""Static configuration validation for the DiR scientific run."""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..training import TRAINING_IMPLEMENTATION_VERSION
from ..training.engine import _build_task_subset
from .matrix import CONDITION_ORDER
from .common import (
    IMPLEMENTATION_REVISION,
    SCHEMA_VERSION,
    _canonical_json_sha256,
    _dictionary_reuse_config,
    _functional_correspondence_config,
)


_EXPECTED_EPOCHS = {
    "dir_source_a_epochs": 80,
    "dir_target_epochs": 52,
    "dense_source_a_epochs": 80,
    "dense_target_epochs": 52,
}

_EXPECTED_PRIMARY_METRICS = [
    "direct_function.single_bidirectional_mean_cls_debiased_cka_12x12",
    "direct_function.single_bidirectional_mean_patch_debiased_cka_12x12",
    "ablation.block_update.post_layernorm_cls_delta_debiased_cka_12x12",
    "ablation.block_update.post_layernorm_patch_delta_debiased_cka_12x12",
    "patching.block_update.common_valid_post_layernorm_cls_recovery_debiased_cka_12x12",
    "patching.block_update.common_valid_post_layernorm_patch_recovery_debiased_cka_12x12",
    "jacobian.input_jvp.input_to_block_update_cls_debiased_cka_12x12",
    "jacobian.input_jvp.input_to_block_update_patch_debiased_cka_12x12",
]


def _validate_run_header(plan: Mapping[str, Any]) -> str:
    if not bool(plan.get("enabled", False)):
        raise ValueError("functional_correspondence.enabled must be true")
    if str(plan.get("schema_version", "")) != SCHEMA_VERSION:
        raise ValueError("DiR schema_version is stale")
    if str(plan.get("implementation_revision", "")) != IMPLEMENTATION_REVISION:
        raise ValueError("DiR implementation_revision is stale")

    mode = str(plan.get("mode", "measurement"))
    if mode != "measurement":
        raise ValueError("DiR mode must be measurement")
    return mode


def _validate_execution_contract(plan: Mapping[str, Any]) -> None:
    execution = dict(plan.get("execution", {}) or {})
    if not bool(execution.get("resume_measurements_from_checkpoints", False)):
        raise ValueError("DiR measurement checkpoint resume must remain enabled")
    if not bool(execution.get("reuse_completed_measurement_shards", False)):
        raise ValueError("DiR measurement shard reuse must remain enabled")
    if str(execution.get("causal_cache_policy", "")) != (
        "auto_exact_ram_with_workdir_spill"
    ):
        raise ValueError(
            "DiR causal cache policy must remain exact RAM-first with work_dir fallback"
        )
    if not bool(execution.get("resume_causal_pending_points_only", False)):
        raise ValueError(
            "DiR causal resume must recompute pending intervention points only"
        )
    if not bool(
        execution.get("core_sharded_before_supplementary_per_pair_task", False)
    ):
        raise ValueError(
            "DiR core measurements must be sharded before supplementary measurements "
            "per pair/task"
        )
    if int(execution.get("minimum_free_disk_bytes", 0)) < 8 * 1024**3:
        raise ValueError("DiR minimum_free_disk_bytes must be at least 8 GiB")
    if str(execution.get("post_training_failure_policy", "")) != (
        "warn_record_preserve_individual_checkpoints_continue_remaining_measurements"
    ):
        raise ValueError("DiR post-training failure policy must remain fail-soft")


def _validate_paths(raw_config: Mapping[str, Any]) -> Path:
    paths = dict(raw_config.get("paths", {}) or {})
    output_dir_value = str(paths.get("output_dir", "") or "").strip()
    work_dir_value = str(paths.get("work_dir", "") or "").strip()
    checkpoint_dir_value = str(paths.get("checkpoint_dir", "") or "").strip()
    if not output_dir_value or not work_dir_value or not checkpoint_dir_value:
        raise ValueError(
            "DiR paths.output_dir, paths.work_dir and paths.checkpoint_dir are required"
        )

    output_path = Path(output_dir_value).expanduser().resolve()
    work_path = Path(work_dir_value).expanduser().resolve()
    checkpoint_path = Path(checkpoint_dir_value).expanduser().resolve()
    try:
        work_path.relative_to(output_path)
    except ValueError as error:
        raise ValueError("DiR paths.work_dir must be inside paths.output_dir") from error
    try:
        checkpoint_path.relative_to(output_path)
    except ValueError:
        pass
    else:
        raise ValueError("DiR paths.checkpoint_dir must be outside paths.output_dir")
    return output_path


def _validate_training_contract(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    training = dict(plan.get("training", {}) or {})
    if {int(value) for value in training.get("checkpoint_epochs", [])} != {20, 52}:
        raise ValueError("DiR training.checkpoint_epochs must remain [20, 52]")
    for key, expected in _EXPECTED_EPOCHS.items():
        if int(training.get(key, -1)) != expected:
            raise ValueError(f"DiR {key} must remain {expected}")

    expected_conditions = list(CONDITION_ORDER)
    if list(plan.get("condition_order", [])) != expected_conditions:
        raise ValueError(f"DiR condition_order must equal {expected_conditions}")

    samples = dict(plan.get("samples", {}) or {})
    expected_samples = {
        "representation": 1024,
        "probe_train": 4096,
        "probe_validation": 1024,
        "probe_test": 1024,
        "response": 512,
        "direct_wide_windows": 128,
        "attention_spectral": 256,
        "gradient": 128,
        "jacobian": 128,
        "bootstrap_iterations": 1000,
        "global_permutations": 5000,
        "depth_band_permutations": 13824,
    }
    for key, expected in expected_samples.items():
        if int(samples.get(key, -1)) != expected:
            raise ValueError(f"DiR samples.{key} must remain {expected}")
    return training, expected_conditions


def _validate_patching_contract(corruption: Mapping[str, Any]) -> None:
    if list(corruption.get("corruptions", [])) != ["mask", "blur", "noise"]:
        raise ValueError("DiR core corruptions must be mask, blur, noise")
    if int(corruption.get("mask_size", 0)) != 8:
        raise ValueError("DiR mask_size must be 8")
    if list(corruption.get("mask_positions", [])) != [4, 12, 20]:
        raise ValueError("DiR mask_positions must remain [4, 12, 20]")
    if str(corruption.get("mask_fill", "")) != "channel_mean":
        raise ValueError("DiR mask_fill must remain channel_mean")
    if int(corruption.get("blur_kernel_size", 0)) != 3:
        raise ValueError("DiR blur_kernel_size must remain 3")
    if str(corruption.get("blur_padding", "")) != "reflect":
        raise ValueError("DiR blur_padding must remain reflect")
    if int(corruption.get("noise_seed", -1)) != 2026080602:
        raise ValueError("DiR noise_seed must remain 2026080602")
    if int(corruption.get("minimum_common_valid_samples", 0)) != 32:
        raise ValueError("DiR minimum_common_valid_samples must remain 32")

    expected_floats = {
        "blur_sigma": (0.8, "DiR blur_sigma must be 0.8"),
        "noise_sigma": (0.03, "DiR noise_sigma must be 0.03"),
        "minimum_relative_corruption_effect": (
            0.05,
            "DiR minimum_relative_corruption_effect must be 0.05",
        ),
        "minimum_block_recovery_fraction": (
            0.01,
            "DiR minimum_block_recovery_fraction must be 0.01",
        ),
        "minimum_median_recovery_fraction": (
            0.0,
            "DiR minimum_median_recovery_fraction must be 0.0",
        ),
        "minimum_positive_recovery_sample_fraction": (
            0.5,
            "DiR minimum_positive_recovery_sample_fraction must be 0.5",
        ),
        "minimum_prediction_retention": (
            0.8,
            "DiR minimum_prediction_retention must be 0.8",
        ),
    }
    for key, (expected, message) in expected_floats.items():
        if abs(float(corruption.get(key, -1.0)) - expected) > 1e-12:
            raise ValueError(message)


def _validate_measurement_quality(plan: Mapping[str, Any]) -> None:
    quality = dict(plan.get("measurement_quality", {}) or {})
    exact_contracts = {
        "cka_primary": "u_centered_debiased_linear_cka",
        "biased_cka_role": "auxiliary_only",
        "full_token_role": "auxiliary_only",
        "token_stage_contract": (
            "local_block_space_for_direct_and_jvp_post_layernorm_final_space_for_"
            "ablation_and_patching"
        ),
        "degenerate_cka_policy": "mask_inconclusive_not_zero_score",
        "jacobian_low_signal_policy": (
            "absolute_below_detection_responses_are_inconclusive_for_jacobian_"
            "similarity_record_raw_signal_and_classification"
        ),
        "same_index_direction_coverage_policy": (
            "retain_single_available_row_or_column_direction_for_rank_rank1_top3_"
            "and_record_direction_count_by_block"
        ),
        "direct_cka_aggregation_policy": (
            "average_only_finite_valid_condition_contributions_all_invalid_cells_"
            "remain_nan_inconclusive"
        ),
        "statistics_status_policy": (
            "finite_intersection_first;_zero_common_valid_cells_are_inconclusive_"
            "not_exception;_diagonal_rank_advantage_and_permutation_availability_"
            "reported_independently"
        ),
    }
    error_messages = {
        "cka_primary": "DiR primary CKA must be U-centered/debiased",
        "biased_cka_role": "DiR biased CKA must remain auxiliary only",
        "full_token_role": "DiR full-token metrics must remain auxiliary only",
        "token_stage_contract": (
            "DiR token_stage_contract must keep direct/JVP in local block space and "
            "ablation/patching in post-LayerNorm final space"
        ),
        "degenerate_cka_policy": (
            "DiR non-Jacobian degenerate CKA cells must remain masked as inconclusive"
        ),
        "jacobian_low_signal_policy": (
            "DiR Jacobian below-detection responses must remain inconclusive for similarity"
        ),
        "same_index_direction_coverage_policy": (
            "DiR one-direction same-index statistics must be retained with coverage recorded"
        ),
        "direct_cka_aggregation_policy": (
            "DiR direct CKA aggregation must exclude undefined contributions before averaging"
        ),
        "statistics_status_policy": (
            "DiR sparse matrix statistics must report component availability independently"
        ),
    }
    for key, expected in exact_contracts.items():
        if str(quality.get(key, "")) != expected:
            raise ValueError(error_messages[key])

    if abs(float(quality.get("minimum_signal_rms_absolute", -1.0)) - 1e-8) > 1e-20:
        raise ValueError("DiR minimum_signal_rms_absolute must be 1e-8")
    if (
        abs(float(quality.get("minimum_signal_rms_relative_to_median", -1.0)) - 0.05)
        > 1e-12
    ):
        raise ValueError("DiR minimum_signal_rms_relative_to_median must be 0.05")


def _validate_jacobian_contract(jacobian: Mapping[str, Any]) -> None:
    probe_count = int(jacobian.get("probe_count", 0))
    randomized_svd_rank = int(jacobian.get("randomized_svd_rank", 0))
    oversampling = int(jacobian.get("oversampling", -1))
    if probe_count != 40:
        raise ValueError("DiR Jacobian range probe count must remain 40")
    if randomized_svd_rank != 32:
        raise ValueError("DiR randomized Jacobian SVD rank must remain 32")
    if oversampling != probe_count - randomized_svd_rank:
        raise ValueError(
            "DiR Jacobian oversampling must equal probe_count - randomized_svd_rank"
        )
    if int(jacobian.get("microbatch_size", 0)) != 8:
        raise ValueError("DiR Jacobian microbatch_size must remain 8")
    if (
        abs(
            float(jacobian.get("range_holdout_relative_residual_maximum", -1.0))
            - 0.50
        )
        > 1e-12
    ):
        raise ValueError("DiR Jacobian range holdout residual maximum must remain 0.50")
    if str(jacobian.get("stability_role", "")) != "advisory_only_no_probe_fallback":
        raise ValueError("DiR Jacobian stability audit must remain advisory only")
    if str(jacobian.get("range_holdout_quality_role", "")) != (
        "advisory_approximation_quality_not_validity_gate"
    ):
        raise ValueError(
            "DiR Jacobian holdout quality must be advisory rather than a validity gate"
        )
    if str(jacobian.get("degenerate_operator_policy", "")) != (
        "below_detection_and_numerical_rank_zero_are_distinct_inconclusive_states_"
        "without_numeric_alignment_score"
    ):
        raise ValueError("DiR Jacobian below-detection/rank-zero convention is stale")
    if str(jacobian.get("constant_response_policy", "")) != (
        "detected_sample_constant_response_is_primary_CKA_inconclusive_shared_"
        "projection_mean_direction_cosine_auxiliary_only"
    ):
        raise ValueError("DiR Jacobian constant-response convention is stale")
    if str(jacobian.get("low_rank_policy", "")) != (
        "use_actual_numerical_rank_without_padding_null_space"
    ):
        raise ValueError("DiR Jacobian low-rank policy is stale")

    split_half_contract = {
        "split_half_spearman_minimum": 0.8,
        "split_half_diagonal_difference_maximum": 0.05,
        "split_half_norm_relative_difference_maximum": 0.15,
    }
    for key, expected in split_half_contract.items():
        if abs(float(jacobian.get(key, -1.0)) - expected) > 1e-12:
            raise ValueError(f"DiR Jacobian {key} must remain {expected}")
    if int(jacobian.get("internal_vjp_descriptor_rank", 0)) != 4:
        raise ValueError("DiR internal VJP descriptor rank must remain 4")


def _validate_measurement_contract(plan: Mapping[str, Any]) -> None:
    corruption = dict(plan.get("patching", {}) or {})
    _validate_patching_contract(corruption)

    atom_ablation = dict(plan.get("atom_group_ablation", {}) or {})
    if (
        abs(float(atom_ablation.get("maximum_relative_mass_mismatch", -1.0)) - 0.10)
        > 1e-12
    ):
        raise ValueError(
            "DiR atom_group_ablation.maximum_relative_mass_mismatch must be 0.10"
        )
    if list(plan.get("primary_metrics", [])) != _EXPECTED_PRIMARY_METRICS:
        raise ValueError("DiR primary_metrics contract is stale")
    if str(plan.get("supplementary_weighted_cca_contract", "")) != (
        "symmetric_singular_value_weighted_canonical_correlation_proxy"
    ):
        raise ValueError("DiR weighted CCA proxy naming contract is stale")

    _validate_measurement_quality(plan)
    _validate_jacobian_contract(dict(plan.get("jacobian", {}) or {}))


def _validate_support_commit_contract(role: Mapping[str, Any]) -> None:
    source_run = dict(role.get("source_run", {}) or {})
    natural_profile_name = str(source_run.get("natural_sparsity_profile", ""))
    natural_profile = dict(
        (role.get("natural_sparsity_profiles", {}) or {}).get(
            natural_profile_name, {}
        )
        or {}
    )
    parity_contract = {
        "forward_routed_hard_support_commit_output_parity_check": True,
        "forward_routed_hard_support_commit_output_parity_sample_count": 128,
        "forward_routed_hard_support_commit_output_parity_max_abs_tolerance": 5e-5,
        "forward_routed_hard_support_commit_output_parity_relative_l2_tolerance": 1e-6,
        "forward_routed_hard_support_commit_output_parity_prediction_mismatch_maximum": 0,
        "forward_routed_hard_support_commit_output_parity_accuracy_difference_maximum": 0.0,
    }
    for key, expected in parity_contract.items():
        actual = natural_profile.get(key)
        if isinstance(expected, float):
            valid = (
                actual is not None
                and np.isfinite(float(actual))
                and abs(float(actual) - expected) <= 1e-12
            )
            if not valid:
                raise ValueError(f"DiR support-commit parity contract mismatch: {key}")
        elif actual != expected:
            raise ValueError(f"DiR support-commit parity contract mismatch: {key}")


def _validate_reproducibility_and_ownership(
    role: Mapping[str, Any], training: Mapping[str, Any]
) -> None:
    _validate_support_commit_contract(role)
    if str(role.get("implementation_version", "")) != TRAINING_IMPLEMENTATION_VERSION:
        raise ValueError("Underlying DiR training implementation is stale")
    if list(training.get("transform_contract", [])) != [
        "to_tensor",
        "cifar100_normalize",
    ]:
        raise ValueError(
            "DiR transform_contract must remain ToTensor plus CIFAR-100 normalization"
        )

    seed_keys = [
        "dir_source_seed",
        "dir_same_task_seed",
        "dir_different_task_head_seed",
        "dense_source_seed",
        "dense_same_task_seed",
        "dense_different_task_head_seed",
    ]
    seeds = [int(training.get(key, -1)) for key in seed_keys]
    if len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise ValueError(
            "DiR/Dense model and head seeds must be explicit and pairwise distinct"
        )

    paired_order_keys = [
        ("dir_source_data_order_seed", "dense_source_data_order_seed"),
        ("dir_same_task_data_order_seed", "dense_same_task_data_order_seed"),
    ]
    paired_order_seeds: list[int] = []
    for dir_key, dense_key in paired_order_keys:
        dir_seed = int(training.get(dir_key, -1))
        dense_seed = int(training.get(dense_key, -1))
        if min(dir_seed, dense_seed) <= 0 or dir_seed != dense_seed:
            raise ValueError(
                "DiR and Dense data-order seeds must match within source/same-task "
                "comparisons"
            )
        paired_order_seeds.append(dir_seed)

    different_order = int(training.get("different_task_data_order_seed", -1))
    if different_order <= 0:
        raise ValueError("Different-task data-order seed must be explicit and positive")
    if len(set([*paired_order_seeds, different_order])) != 3:
        raise ValueError(
            "Source, same-task, and different-task data-order seeds must remain distinct"
        )

    if str(training.get("dense_learning_rate_profile", "")) not in role.get(
        "learning_rate_profiles", {}
    ):
        raise ValueError("Dense learning-rate profile is missing")
    if str(training.get("dense_gradient_clip_profile", "")) not in role.get(
        "gradient_clip_profiles", {}
    ):
        raise ValueError("Dense gradient-clip profile is missing")

    if str(training.get("ownership_contract", "")) != (
        "same-task: fresh Target with Source-active D and D-owned scales fixed, Source-"
        "inactive D trainable only on phase-allowed dictionary coordinates (internal-facing block D plus included head D), "
        "C/route/support fresh; different-task Dictionary-Fixed/Dictionary-Trainable: "
        "identical Source-full-backbone plus identical fresh head, differing only in Source-"
        "active D/scale anchoring while Source-inactive D remains trainable only on phase-"
        "allowed dictionary coordinates (internal-facing block D plus included head D) in "
        "Dictionary-Fixed; Dense different-"
        "task: Source endpoint copy "
        "plus fresh head"
    ):
        raise ValueError("Final-paper ownership contract is stale")
    if str(training.get("control_contract", "")) != (
        "different-task Dictionary-Fixed and Dictionary-Trainable share exact initial "
        "backbone, fresh head and data order; only Source-active D/D-owned-scale "
        "anchoring differs, and Source-inactive D remains trainable only on phase-allowed "
        "dictionary coordinates (internal-facing block D plus included head D) in "
        "Dictionary-Fixed"
    ):
        raise ValueError(
            "DiR Dictionary-Fixed/Dictionary-Trainable control contract is stale"
        )

    dataset = dict(role.get("dataset", {}) or {})
    if str(dataset.get("train_split")) == str(dataset.get("eval_split")):
        raise ValueError("DiR requires disjoint train/eval dataset splits")


def _validate_output_contract(
    raw_config: Mapping[str, Any], output_path: Path
) -> None:
    paths = dict(raw_config.get("paths", {}) or {})
    zip_paths = [
        str(paths.get(key, "")) for key in ("raw_report_zip", "summary_report_zip")
    ]
    if any(not value for value in zip_paths) or len(set(zip_paths)) != 2:
        raise ValueError("DiR raw/summary report ZIP paths must be non-empty and distinct")
    for value in zip_paths:
        zip_path = Path(value).expanduser().resolve()
        if zip_path.parent != output_path:
            raise ValueError(
                "DiR report ZIPs must be written directly inside paths.output_dir"
            )

    outputs = list(raw_config.get("active_output_files", []))
    if len(outputs) != len(set(outputs)):
        raise ValueError("DiR active_output_files contains a collision")


def validate_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed final-paper run contract and return execution metadata."""

    plan = _functional_correspondence_config(raw_config)
    role = _dictionary_reuse_config(raw_config)

    mode = _validate_run_header(plan)
    _validate_execution_contract(plan)
    output_path = _validate_paths(raw_config)
    training, expected_conditions = _validate_training_contract(plan)
    _validate_measurement_contract(plan)
    _validate_reproducibility_and_ownership(role, training)
    _validate_output_contract(raw_config, output_path)

    require_cuda = bool(plan.get("require_cuda", True))
    execution_kind = "training"
    return {
        "mode": mode,
        "stage": "functional_correspondence",
        "execution_kind": execution_kind,
        "training_enabled": execution_kind == "training",
        "analysis_enabled": mode == "measurement",
        "require_cuda": require_cuda,
        "condition_ids": expected_conditions,
        "epochs": dict(_EXPECTED_EPOCHS),
    }


def runtime_environment_snapshot() -> dict[str, Any]:
    """Capture reproducibility metadata for the scientific run."""

    try:
        torchvision_version = str(importlib.import_module("torchvision").__version__)
    except Exception as error:
        torchvision_version = f"unavailable: {type(error).__name__}: {error}"
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    nvidia_driver_version: str | None = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if values:
                nvidia_driver_version = values[0]
    except Exception:
        nvidia_driver_version = None
    colab_release_tag = os.environ.get("COLAB_RELEASE_TAG")
    colab_backend_version = os.environ.get("COLAB_BACKEND_VERSION")
    payload = {
        "python_version": str(sys.version).replace("\n", " "),
        "python_executable": str(sys.executable),
        "platform": platform.platform(),
        "system": platform.system(),
        "system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "cuda_runtime_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
        "nvidia_driver_version": nvidia_driver_version,
        "execution_environment": (
            "google_colab"
            if colab_release_tag is not None or "COLAB_GPU" in os.environ
            else "standard_python"
        ),
        "colab_release_tag": colab_release_tag,
        "colab_backend_version": colab_backend_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.device("cuda"))
        payload.update(
            {
                "gpu_device_name": str(properties.name),
                "gpu_compute_capability": list(torch.cuda.get_device_capability(torch.device("cuda"))),
                "gpu_total_memory_bytes": int(properties.total_memory),
            }
        )
    return payload


def _ordered_subset_source_ids(subset: Any, *, task_key: str, split: str) -> list[int]:
    source_indices = list(getattr(subset, "indices", list(range(len(subset)))))
    order = sorted(
        range(len(source_indices)),
        key=lambda local: hashlib.sha256(
            f"{task_key}/{split}/{int(source_indices[local])}".encode("utf-8")
        ).hexdigest(),
    )
    return [int(source_indices[local]) for local in order]


def build_dataset_sample_reference(
    role: dict[str, Any], samples: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate CIFAR capacity and freeze deterministic sample IDs before training."""

    required_train = int(samples["probe_train"]) + int(samples["probe_validation"])
    required_eval = max(int(samples["representation"]), int(samples["probe_test"]))
    representation_count = int(samples["representation"])
    probe_train_count = int(samples["probe_train"])
    probe_validation_count = int(samples["probe_validation"])
    probe_test_count = int(samples["probe_test"])
    tasks: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "selection": "sha256(task/split/original_index) ascending",
        "nested": "128⊂256⊂512⊂1024",
        "probe_split_contract": "train[0:4096], train[4096:5120], eval[0:1024]",
        "tasks": {},
    }
    for task_key in ("task1", "task2"):
        train_subset = _build_task_subset(role, task_key=task_key, loader_split_name="train")
        eval_subset = _build_task_subset(role, task_key=task_key, loader_split_name="eval")
        train_ids = _ordered_subset_source_ids(train_subset, task_key=task_key, split="train")
        eval_ids = _ordered_subset_source_ids(eval_subset, task_key=task_key, split="eval")
        selected_eval_ids = eval_ids[:representation_count]
        probe_train_ids = train_ids[:probe_train_count]
        probe_validation_ids = train_ids[
            probe_train_count : probe_train_count + probe_validation_count
        ]
        probe_test_ids = eval_ids[:probe_test_count]
        task_passed = bool(
            len(train_ids) >= required_train
            and len(eval_ids) >= required_eval
            and len(selected_eval_ids) == representation_count
            and len(probe_train_ids) == probe_train_count
            and len(probe_validation_ids) == probe_validation_count
            and len(probe_test_ids) == probe_test_count
            and set(probe_train_ids).isdisjoint(probe_validation_ids)
        )
        tasks[task_key] = {
            "train_count": len(train_ids),
            "eval_count": len(eval_ids),
            "required_train_count": required_train,
            "required_eval_count": required_eval,
            "passed": task_passed,
        }
        manifest["tasks"][task_key] = {
            "ids_128": selected_eval_ids[:128],
            "ids_256": selected_eval_ids[:256],
            "ids_512": selected_eval_ids[:512],
            "ids_1024": selected_eval_ids[:1024],
            "probe_train_ids": probe_train_ids,
            "probe_validation_ids": probe_validation_ids,
            "probe_test_ids": probe_test_ids,
            "probe_splits_disjoint": bool(set(probe_train_ids).isdisjoint(probe_validation_ids)),
        }
    if not all(value["passed"] for value in tasks.values()):
        raise RuntimeError(f"DiR dataset/sample capacity validation failed: {tasks}")
    return {
        "dataset_capacity": {"passed": True, "tasks": tasks},
        "sample_manifest": manifest,
        "sample_manifest_sha256": _canonical_json_sha256(manifest),
    }
