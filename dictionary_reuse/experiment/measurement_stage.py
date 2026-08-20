"""Post-training functional-correspondence measurement suite and stage orchestration."""

from __future__ import annotations


# Pair measurement suite
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from ..dictionary_transfer import mapped_endpoint_active_masks
from ..interventions import capture_model_runtime_signature
from ..measurements import (
    ablation_response_alignment_suite,
    block_update_alignment,
    direct_block_function_alignment,
    jacobian_input_response_alignment,
    jacobian_internal_vjp_alignment,
    patching_recovery_alignment_suite,
)
from ..supplementary import (
    atom_group_ablation,
    attention_transport_alignment,
    cross_model_activation_patching_alignment,
    full_block_swap_alignment,
    gradient_profile_alignment,
    linear_probe_profiles,
    representation_geometry_alignment,
    spectral_perturbation_alignment,
)
from .measurement_support import (
    MeasurementBatch,
    MeasurementModuleRunner,
    MeasurementPair,
    PatchingFamilyValidityCache,
    truncate_batches,
)

CAUSAL_INTERVENTION_POINTS = (
    "block_update",
    "post_o_attention_output",
    "post_w2_mlp_output",
)


@dataclass
class PairSuiteContext:
    pair_name: str
    pair: MeasurementPair
    task_batches: Mapping[str, Mapping[str, Any]]
    samples: Mapping[str, Any]
    quality: Mapping[str, Any]
    plan: Mapping[str, Any]
    device: torch.device
    mean: Sequence[float]
    std: Sequence[float]
    work_dir: Path
    runner: MeasurementModuleRunner
    patching_validity: PatchingFamilyValidityCache
    jacobian_model_descriptor_cache: dict[tuple[Any, ...], Any]

    @property
    def task_key(self) -> str:
        return self.pair.task_key

    @property
    def prefix(self) -> str:
        return f"{self.pair_name}.{self.task_key}"

    @property
    def shared_head(self) -> bool:
        return bool(self.pair.same_task)

    @property
    def label_semantics_valid(self) -> bool:
        return bool(
            self.shared_head and str(self.pair.shared_head_task) == str(self.task_key)
        )

    def batches(self, sample_key: str) -> list[MeasurementBatch]:
        return truncate_batches(
            self.task_batches[self.task_key]["all"], int(self.samples[sample_key])
        )


class CausalSuiteCache:
    """Compute each pending causal suite once and expose per-point results."""

    def __init__(self, context: PairSuiteContext, response_batches: Sequence[MeasurementBatch]):
        self._context = context
        self._response_batches = response_batches
        self._ablation_results: dict[str, Any] | None = None
        self._patching_results: dict[str, dict[str, Any]] = {}

    def ablation_result(self, point: str) -> Any:
        if self._ablation_results is None:
            pending = self._pending_ablation_points()
            suite_points = pending or (str(point),)
            before = self._runtime_signatures()
            results = ablation_response_alignment_suite(
                self._context.pair.left,
                self._context.pair.right,
                self._response_batches,
                device=self._context.device,
                intervention_points=suite_points,
                same_head=self._context.shared_head,
                label_semantics_valid=self._context.label_semantics_valid,
                minimum_signal_rms_absolute=float(
                    self._context.quality["minimum_signal_rms_absolute"]
                ),
                minimum_signal_rms_relative_to_median=float(
                    self._context.quality["minimum_signal_rms_relative_to_median"]
                ),
                cache_directory=self._context.work_dir / "causal_cache",
            )
            self._ablation_results = _attach_shared_suite_audit(
                results, before=before, after=self._runtime_signatures()
            )
        return self._ablation_results[str(point)]

    def patching_result(self, corruption: str, point: str) -> Any:
        corruption_key = str(corruption)
        if corruption_key not in self._patching_results:
            pending = self._pending_patching_points(corruption_key)
            suite_points = pending or (str(point),)
            family_context = self._context.patching_validity.get(
                current_pair_name=self._context.pair_name,
                task_key=self._context.task_key,
                corruption=corruption_key,
                batches=self._response_batches,
                same_head=self._context.shared_head,
                current_left=self._context.pair.left,
                current_right=self._context.pair.right,
            )
            pair_audits = family_context["pair_audits"][self._context.pair_name]
            before = self._runtime_signatures()
            patching = self._context.plan["patching"]
            results = patching_recovery_alignment_suite(
                self._context.pair.left,
                self._context.pair.right,
                self._response_batches,
                device=self._context.device,
                corruption=corruption_key,
                intervention_points=suite_points,
                mean=self._context.mean,
                std=self._context.std,
                minimum_relative_effect=float(patching["minimum_relative_corruption_effect"]),
                minimum_prediction_retention=float(patching["minimum_prediction_retention"]),
                minimum_common_valid_samples=int(patching["minimum_common_valid_samples"]),
                minimum_block_recovery_fraction=float(patching["minimum_block_recovery_fraction"]),
                minimum_median_recovery_fraction=float(patching["minimum_median_recovery_fraction"]),
                minimum_positive_recovery_sample_fraction=float(
                    patching["minimum_positive_recovery_sample_fraction"]
                ),
                same_head=self._context.shared_head,
                label_semantics_valid=self._context.label_semantics_valid,
                minimum_signal_rms_absolute=float(
                    self._context.quality["minimum_signal_rms_absolute"]
                ),
                minimum_signal_rms_relative_to_median=float(
                    self._context.quality["minimum_signal_rms_relative_to_median"]
                ),
                noise_sigma=float(patching["noise_sigma"]),
                noise_seed=int(patching["noise_seed"]),
                blur_sigma=float(patching["blur_sigma"]),
                blur_kernel_size=int(patching["blur_kernel_size"]),
                blur_padding=str(patching["blur_padding"]),
                mask_size=int(patching["mask_size"]),
                mask_positions=tuple(patching["mask_positions"]),
                mask_fill=str(patching["mask_fill"]),
                cache_directory=self._context.work_dir / "causal_cache",
                precomputed_left_corruption_audit=pair_audits["left"],
                precomputed_right_corruption_audit=pair_audits["right"],
                external_common_valid_masks=family_context["common_valid_masks"],
            )
            self._patching_results[corruption_key] = _attach_shared_suite_audit(
                results, before=before, after=self._runtime_signatures()
            )
        return self._patching_results[corruption_key][str(point)]

    def _pending_ablation_points(self) -> tuple[str, ...]:
        return tuple(
            point
            for point in CAUSAL_INTERVENTION_POINTS
            if not self._context.runner.reusable_shard_exists(
                f"{self._context.prefix}.core.ablation.{point}"
            )
        )

    def _pending_patching_points(self, corruption: str) -> tuple[str, ...]:
        return tuple(
            point
            for point in CAUSAL_INTERVENTION_POINTS
            if not self._context.runner.reusable_shard_exists(
                f"{self._context.prefix}.core.patching.{corruption}.{point}"
            )
        )

    def _runtime_signatures(self) -> dict[str, Any]:
        return {
            "left": capture_model_runtime_signature(self._context.pair.left),
            "right": capture_model_runtime_signature(self._context.pair.right),
        }


def run_all_pair_measurements(
    *,
    pairs: Mapping[str, MeasurementPair],
    task_batches: Mapping[str, Mapping[str, Any]],
    samples: Mapping[str, Any],
    quality: Mapping[str, Any],
    plan: Mapping[str, Any],
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
    work_dir: Path,
    runner: MeasurementModuleRunner,
    patching_validity: PatchingFamilyValidityCache,
    jacobian_model_descriptor_cache: dict[tuple[Any, ...], Any],
) -> None:
    """Apply the same core/supplementary suite to every canonical pair."""

    for pair_name, pair in pairs.items():
        context = PairSuiteContext(
            pair_name=pair_name,
            pair=pair,
            task_batches=task_batches,
            samples=samples,
            quality=quality,
            plan=plan,
            device=device,
            mean=mean,
            std=std,
            work_dir=work_dir,
            runner=runner,
            patching_validity=patching_validity,
            jacobian_model_descriptor_cache=jacobian_model_descriptor_cache,
        )
        _run_pair_suite(context)


def _run_pair_suite(context: PairSuiteContext) -> None:
    representation_batches = context.batches("representation")
    response_batches = context.batches("response")
    wide_window_batches = context.batches("direct_wide_windows")
    attention_batches = context.batches("attention_spectral")
    gradient_batches = context.batches("gradient")
    jacobian_batches = context.batches("jacobian")

    _run_direct_suite(context, representation_batches, response_batches, wide_window_batches)
    _run_causal_suite(context, response_batches)
    _run_jacobian_suite(context, jacobian_batches)
    _run_supplementary_suite(
        context,
        representation_batches=representation_batches,
        response_batches=response_batches,
        attention_batches=attention_batches,
        gradient_batches=gradient_batches,
    )


def _run_direct_suite(
    context: PairSuiteContext,
    representation_batches: Sequence[MeasurementBatch],
    response_batches: Sequence[MeasurementBatch],
    wide_window_batches: Sequence[MeasurementBatch],
) -> None:
    left, right = context.pair.left, context.pair.right
    quality = context.quality
    common_quality = {
        "minimum_signal_rms_absolute": float(quality["minimum_signal_rms_absolute"]),
        "minimum_signal_rms_relative_to_median": float(
            quality["minimum_signal_rms_relative_to_median"]
        ),
    }
    context.runner.run(
        name=f"{context.prefix}.core.block_update",
        function=lambda: block_update_alignment(
            left, right, representation_batches, device=context.device, **common_quality
        ),
        module_models={"left": left, "right": right},
    )
    context.runner.run(
        name=f"{context.prefix}.core.direct_function",
        function=lambda: direct_block_function_alignment(
            left,
            right,
            response_batches,
            device=context.device,
            window_widths=(1, 2),
            **common_quality,
        ),
        module_models={"left": left, "right": right},
    )
    context.runner.run(
        name=f"{context.prefix}.supplementary.direct_wide_windows",
        function=lambda: direct_block_function_alignment(
            left,
            right,
            wide_window_batches,
            device=context.device,
            window_widths=(3, 4, 6, 12),
            include_single_if_missing=False,
            **common_quality,
        ),
        module_models={"left": left, "right": right},
    )


def _run_causal_suite(
    context: PairSuiteContext, response_batches: Sequence[MeasurementBatch]
) -> None:
    left, right = context.pair.left, context.pair.right
    suite_cache = CausalSuiteCache(context, response_batches)
    for point in CAUSAL_INTERVENTION_POINTS:
        context.runner.run(
            name=f"{context.prefix}.core.ablation.{point}",
            function=lambda point=point: suite_cache.ablation_result(point),
            module_models={"left": left, "right": right},
        )
        for corruption in context.plan["patching"]["corruptions"]:
            context.runner.run(
                name=f"{context.prefix}.core.patching.{corruption}.{point}",
                function=lambda corruption=corruption, point=point: suite_cache.patching_result(
                    str(corruption), point
                ),
                module_models={"left": left, "right": right},
            )


def _run_jacobian_suite(
    context: PairSuiteContext, jacobian_batches: Sequence[MeasurementBatch]
) -> None:
    left, right = context.pair.left, context.pair.right
    jacobian = context.plan["jacobian"]
    quality = context.quality
    shared_arguments = {
        "device": context.device,
        "probe_count": int(jacobian["probe_count"]),
        "probe_seed": int(jacobian["probe_seed"]),
        "split_half_spearman_minimum": float(jacobian["split_half_spearman_minimum"]),
        "split_half_diagonal_difference_maximum": float(
            jacobian["split_half_diagonal_difference_maximum"]
        ),
        "split_half_norm_relative_difference_maximum": float(
            jacobian["split_half_norm_relative_difference_maximum"]
        ),
        "minimum_signal_rms_absolute": float(quality["minimum_signal_rms_absolute"]),
        "minimum_signal_rms_relative_to_median": float(
            quality["minimum_signal_rms_relative_to_median"]
        ),
    }
    context.runner.run(
        name=f"{context.prefix}.core.jacobian.input_jvp",
        function=lambda: jacobian_input_response_alignment(
            left,
            right,
            jacobian_batches,
            normalization_std=context.std,
            randomized_svd_rank=int(jacobian["randomized_svd_rank"]),
            range_holdout_relative_residual_maximum=float(
                jacobian["range_holdout_relative_residual_maximum"]
            ),
            microbatch_size=int(jacobian["microbatch_size"]),
            model_descriptor_cache=context.jacobian_model_descriptor_cache,
            **shared_arguments,
        ),
        module_models={"left": left, "right": right},
    )
    internal_arguments = dict(shared_arguments)
    internal_arguments["dominant_subspace_rank"] = int(jacobian["internal_vjp_descriptor_rank"])
    context.runner.run(
        name=f"{context.prefix}.core.jacobian.internal_vjp",
        function=lambda: jacobian_internal_vjp_alignment(
            left, right, jacobian_batches, **internal_arguments
        ),
        module_models={"left": left, "right": right},
    )


def _run_supplementary_suite(
    context: PairSuiteContext,
    *,
    representation_batches: Sequence[MeasurementBatch],
    response_batches: Sequence[MeasurementBatch],
    attention_batches: Sequence[MeasurementBatch],
    gradient_batches: Sequence[MeasurementBatch],
) -> None:
    left, right = context.pair.left, context.pair.right
    modules: list[tuple[str, Any, Sequence[MeasurementBatch]]] = [
        (
            "representation_geometry",
            representation_geometry_alignment,
            representation_batches,
        ),
        ("attention_transport", attention_transport_alignment, attention_batches),
        ("full_block_swap", full_block_swap_alignment, response_batches),
        (
            "cross_model_activation_patching",
            cross_model_activation_patching_alignment,
            response_batches,
        ),
    ]
    for name, function, batches in modules:
        context.runner.run(
            name=f"{context.prefix}.supplementary.{name}",
            function=lambda function=function, batches=batches: function(
                left, right, batches, device=context.device
            ),
            module_models={"left": left, "right": right},
        )

    include_task_loss = bool(context.pair.same_task and context.task_key == "task1")
    task_scope = "shared_native_task1" if include_task_loss else "not_shared_native_task"
    context.runner.run(
        name=f"{context.prefix}.supplementary.gradient_profile",
        function=lambda: gradient_profile_alignment(
            left,
            right,
            gradient_batches,
            device=context.device,
            include_task_loss=include_task_loss,
            task_loss_scope=task_scope,
        ),
        module_models={"left": left, "right": right},
    )
    context.runner.run(
        name=f"{context.prefix}.supplementary.spectral_response",
        function=lambda: spectral_perturbation_alignment(
            left,
            right,
            attention_batches,
            device=context.device,
            mean=context.mean,
            std=context.std,
        ),
        module_models={"left": left, "right": right},
    )
    if (context.pair.same_task and context.task_key == "task1") or not context.pair.same_task:
        context.runner.run(
            name=f"{context.prefix}.supplementary.probes",
            function=lambda: linear_probe_profiles(
                left,
                right,
                context.task_batches[context.task_key]["probe_train"],
                context.task_batches[context.task_key]["probe_validation"],
                context.task_batches[context.task_key]["probe_test"],
                device=context.device,
                same_task=context.pair.same_task,
            ),
            module_models={"left": left, "right": right},
        )


def run_dir_atom_ablation_measurements(
    *,
    dir_source: nn.Module,
    dir_targets: Mapping[str, nn.Module],
    ownership: Mapping[str, Any],
    task_batches: Mapping[str, Mapping[str, Any]],
    samples: Mapping[str, Any],
    plan: Mapping[str, Any],
    device: torch.device,
    runner: MeasurementModuleRunner,
    console: Any,
) -> None:
    console("  DiR atom ablation")
    conditions = (
        ("dir_same_task", "dir_same_task", "task1", "DiR Same-task | task1"),
        (
            "dir_dictionary_fixed",
            "dir_dictionary_fixed",
            "task2",
            "DiR Dictionary-Fixed | task2",
        ),
        (
            "dir_dictionary_trainable",
            "dir_dictionary_trainable",
            "task2",
            "DiR Dictionary-Trainable | task2",
        ),
    )
    for condition_name, target_key, task_key, condition_label in conditions:
        console(f"    {condition_label}")
        target = dir_targets[target_key]
        mapping = dict(
            dict(ownership.get(condition_name, {}) or {}).get(
                "block_mapping_target_to_source", {}
            )
            or {}
        )
        source_masks = mapped_endpoint_active_masks(
            dir_source, target, block_mapping=mapping
        )
        atom_batches = truncate_batches(
            task_batches[task_key]["all"], int(samples["response"])
        )
        runner.run(
            name=f"{condition_name}.{task_key}.supplementary.atom_group_ablation",
            function=lambda target=target, batches=atom_batches, masks=source_masks: atom_group_ablation(
                dir_source,
                target,
                batches,
                device=device,
                source_active_masks=masks,
                maximum_relative_mass_mismatch=float(
                    plan["atom_group_ablation"]["maximum_relative_mass_mismatch"]
                ),
            ),
            module_models={"source": dir_source, "target": target},
            log_progress=False,
        )


def _attach_shared_suite_audit(
    results: Mapping[str, Any], *, before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    unchanged = dict(before) == dict(after)
    audited: dict[str, Any] = dict(results)
    audit_payload = {
        "model_state_unchanged": bool(unchanged),
        "scope": "shared_pending_causal_suite",
        "intervention_points": sorted(str(key) for key in audited),
        "intervention_point_count": len(audited),
        "contract": (
            "same_pending_suite_model_state_audit_is_attached_to_every_computed_"
            "intervention_point"
        ),
    }
    for point_key, raw_payload in list(audited.items()):
        if not isinstance(raw_payload, Mapping):
            continue
        point_payload = dict(raw_payload)
        point_payload["shared_suite_model_state_audit"] = dict(audit_payload)
        if not unchanged and str(point_payload.get("measurement_status", "completed")) == "completed":
            point_payload["measurement_status"] = "warning_shared_suite_model_state_changed"
        audited[str(point_key)] = point_payload
    return audited


# Measurement stage orchestration
from .common import (
    MEASUREMENT_SHARD_DIRECTORY,
    MODULE_MANIFEST_FILE,
    SAMPLE_MANIFEST_FILE,
    _canonical_json_sha256,
)
from .measurement_support import finalize_measurement_statistics
from .measurement_support import (
    MeasurementModuleRunner,
    PatchingFamilyValidityCache,
    build_measurement_pairs,
    prepare_task_batches,
)
from .runtime import _remove_stale_measurement_shards


@dataclass
class MeasurementStageResult:
    completed: int
    warning_count: int
    inconclusive_count: int
    superseded_count: int
    core_names: list[str]
    supplementary_names: list[str]
    core_measurement_status: str
    supplementary_measurement_status: str
    measurement_status: str
    training_valid: bool
    condition_comparisons: dict[str, Any]
    primary_results: dict[str, Any]
    supporting_results: dict[str, Any]
    jacobian_rank32_results: dict[str, Any]
    jacobian_stability_contract: dict[str, Any]
    measurement_shard_dir: Path
    module_status: dict[str, Any]
    run_fingerprint: str
    statistics_fingerprint: str
    stale_measurement_shards_removed: int
    current_measurement_shard_count: int


def run_measurement_stage(
    *,
    artifact_run_id: str,
    checkpoint_provenance_sha256: str,
    dataset_sample_reference: Mapping[str, Any],
    device: torch.device,
    dir_source: nn.Module,
    dir_targets: Mapping[str, nn.Module],
    endpoint_paths: Mapping[str, Path],
    measurement_contract_sha256: str,
    models: Mapping[str, nn.Module],
    output_dir: Path,
    ownership: Mapping[str, Any],
    plan: Mapping[str, Any],
    role: dict[str, Any],
    statistics_contract_sha256: str,
    training_contract_sha256: str,
    training_csv_sha256: str,
    work_dir: Path,
    write_post_training_json: Callable[..., bool],
    write_post_training_progress: Callable[..., bool],
    record_post_training_warning: Callable[..., None],
) -> MeasurementStageResult:
    """Run all post-training measurements and fail-soft summary statistics."""

    samples = dict(plan["samples"])
    quality = dict(plan["measurement_quality"])
    console = _console_writer(role)

    console("  samples")
    task_batches, sample_manifest = prepare_task_batches(role, samples)
    write_post_training_json(
        output_dir / SAMPLE_MANIFEST_FILE,
        sample_manifest,
        stage="sample_manifest_persistence",
        message=(
            "sample manifest persistence failed after training; the in-memory deterministic "
            "sample manifest remains authoritative for this process and measurement continues"
        ),
    )

    measurement_shard_dir = work_dir / MEASUREMENT_SHARD_DIRECTORY
    _create_shard_directory(
        measurement_shard_dir, record_warning=record_post_training_warning
    )
    checkpoint_file_identity = _checkpoint_file_identity(
        endpoint_paths, record_warning=record_post_training_warning
    )
    sample_manifest_sha256 = _canonical_json_sha256(sample_manifest)
    _warn_on_sample_manifest_mismatch(
        sample_manifest_sha256=sample_manifest_sha256,
        dataset_sample_reference=dataset_sample_reference,
        record_warning=record_post_training_warning,
    )
    run_fingerprint, statistics_fingerprint = _measurement_fingerprints(
        measurement_contract_sha256=measurement_contract_sha256,
        training_contract_sha256=training_contract_sha256,
        training_csv_sha256=training_csv_sha256,
        sample_manifest_sha256=sample_manifest_sha256,
        checkpoint_provenance_sha256=checkpoint_provenance_sha256,
        checkpoint_file_identity=checkpoint_file_identity,
        statistics_contract_sha256=statistics_contract_sha256,
    )
    stale_removed = _remove_stale_shards_failsoft(
        measurement_shard_dir,
        run_fingerprint=run_fingerprint,
        record_warning=record_post_training_warning,
    )

    module_results: dict[str, Any] = {}
    module_status: dict[str, Any] = {}
    _offload_measurement_models(models)
    runner = MeasurementModuleRunner(
        all_models=models,
        device=device,
        module_results=module_results,
        module_status=module_status,
        shard_dir=measurement_shard_dir,
        module_manifest_path=output_dir / MODULE_MANIFEST_FILE,
        run_fingerprint=run_fingerprint,
        model_checkpoint_paths={id(models[name]): endpoint_paths[name] for name in models},
        warning_recorder=record_post_training_warning,
        console=console,
    )

    pairs = build_measurement_pairs(models)
    mean = role["dataset"]["normalization_mean"]
    std = role["dataset"]["normalization_standard_deviation"]
    patching_validity = PatchingFamilyValidityCache(
        pairs=pairs,
        runner=runner,
        plan=plan,
        device=device,
        mean=mean,
        std=std,
    )
    jacobian_descriptor_cache: dict[tuple[Any, ...], Any] = {}

    run_all_pair_measurements(
        pairs=pairs,
        task_batches=task_batches,
        samples=samples,
        quality=quality,
        plan=plan,
        device=device,
        mean=mean,
        std=std,
        work_dir=work_dir,
        runner=runner,
        patching_validity=patching_validity,
        jacobian_model_descriptor_cache=jacobian_descriptor_cache,
    )
    run_dir_atom_ablation_measurements(
        dir_source=dir_source,
        dir_targets=dir_targets,
        ownership=ownership,
        task_batches=task_batches,
        samples=samples,
        plan=plan,
        device=device,
        runner=runner,
        console=console,
    )
    runner.activate_models({})
    jacobian_descriptor_cache.clear()

    jacobian_stability_contract = _jacobian_stability_contract(plan)
    statistics = finalize_measurement_statistics(
        artifact_run_id=artifact_run_id,
        checkpoint_file_identity=checkpoint_file_identity,
        checkpoint_provenance_sha256=checkpoint_provenance_sha256,
        console=console,
        jacobian_stability_contract=jacobian_stability_contract,
        measurement_shard_dir=measurement_shard_dir,
        module_results=module_results,
        module_status=module_status,
        output_dir=output_dir,
        plan=plan,
        record_warning=record_post_training_warning,
        run_fingerprint=run_fingerprint,
        sample_manifest_sha256=sample_manifest_sha256,
        statistics_contract_sha256=statistics_contract_sha256,
        statistics_fingerprint=statistics_fingerprint,
        training_contract_sha256=training_contract_sha256,
        training_csv_sha256=training_csv_sha256,
        write_json=write_post_training_json,
        write_progress=write_post_training_progress,
    )

    return MeasurementStageResult(
        completed=statistics.completed,
        warning_count=statistics.warning_count,
        inconclusive_count=statistics.inconclusive_count,
        superseded_count=statistics.superseded_count,
        core_names=statistics.core_names,
        supplementary_names=statistics.supplementary_names,
        core_measurement_status=statistics.core_measurement_status,
        supplementary_measurement_status=statistics.supplementary_measurement_status,
        measurement_status=statistics.measurement_status,
        training_valid=bool(all(value.get("passed", False) for value in ownership.values())),
        condition_comparisons=statistics.condition_comparisons,
        primary_results=statistics.primary_results,
        supporting_results=statistics.supporting_results,
        jacobian_rank32_results=statistics.jacobian_rank32_results,
        jacobian_stability_contract=jacobian_stability_contract,
        measurement_shard_dir=measurement_shard_dir,
        module_status=module_status,
        run_fingerprint=run_fingerprint,
        statistics_fingerprint=statistics_fingerprint,
        stale_measurement_shards_removed=len(stale_removed),
        current_measurement_shard_count=statistics.current_measurement_shard_count,
    )


def _console_writer(role: Mapping[str, Any]) -> Callable[[str], None]:
    enabled = bool(dict(role.get("runtime", {}) or {}).get("console_logging_enabled", False))

    def console(message: str) -> None:
        if enabled:
            print(message, flush=True)

    return console


def _create_shard_directory(
    shard_dir: Path, *, record_warning: Callable[..., None]
) -> None:
    try:
        shard_dir.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        record_warning(
            stage="measurement_shard_directory",
            message=(
                "measurement shard directory could not be created after training; measurements "
                "continue in memory and individual shard writes will remain fail-soft"
            ),
            error=error,
        )


def _checkpoint_file_identity(
    endpoint_paths: Mapping[str, Path], *, record_warning: Callable[..., None]
) -> dict[str, dict[str, Any]]:
    identity: dict[str, dict[str, Any]] = {}
    for endpoint_name, endpoint_path in endpoint_paths.items():
        path = Path(endpoint_path)
        try:
            stat = path.stat()
            identity[endpoint_name] = {
                "filename": path.name,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        except OSError as error:
            identity[endpoint_name] = {"filename": path.name, "status": "unavailable"}
            record_warning(
                stage=f"endpoint_checkpoint_identity:{endpoint_name}",
                message=(
                    "endpoint checkpoint file identity could not be read after training; the "
                    "in-memory endpoint model is retained and measurement continues"
                ),
                error=error,
            )
    return identity


def _warn_on_sample_manifest_mismatch(
    *,
    sample_manifest_sha256: str,
    dataset_sample_reference: Mapping[str, Any],
    record_warning: Callable[..., None],
) -> None:
    reference = str(dataset_sample_reference.get("sample_manifest_sha256", ""))
    if reference and sample_manifest_sha256 != reference:
        record_warning(
            stage="sample_manifest_reference_parity",
            message=(
                "post-training measurement sample IDs differ from the deterministic manifest "
                "frozen before training; preserve checkpoints/results but mark measurement "
                "validity for review"
            ),
        )


def _measurement_fingerprints(
    *,
    measurement_contract_sha256: str,
    training_contract_sha256: str,
    training_csv_sha256: str,
    sample_manifest_sha256: str,
    checkpoint_provenance_sha256: str,
    checkpoint_file_identity: Mapping[str, Any],
    statistics_contract_sha256: str,
) -> tuple[str, str]:
    run_fingerprint = _canonical_json_sha256(
        {
            "measurement_contract_sha256": measurement_contract_sha256,
            "training_contract_sha256": training_contract_sha256,
            "training_csv_sha256": training_csv_sha256,
            "sample_manifest_sha256": sample_manifest_sha256,
            "checkpoint_provenance_sha256": checkpoint_provenance_sha256,
            "checkpoint_file_identity": checkpoint_file_identity,
        }
    )
    statistics_fingerprint = _canonical_json_sha256(
        {
            "raw_measurement_run_fingerprint": run_fingerprint,
            "statistics_contract_sha256": statistics_contract_sha256,
        }
    )
    return run_fingerprint, statistics_fingerprint


def _remove_stale_shards_failsoft(
    shard_dir: Path,
    *,
    run_fingerprint: str,
    record_warning: Callable[..., None],
) -> list[Path]:
    try:
        return _remove_stale_measurement_shards(
            shard_dir, run_fingerprint=run_fingerprint
        )
    except Exception as error:
        record_warning(
            stage="stale_measurement_shard_cleanup",
            message=(
                "stale measurement shard cleanup failed after training; shard reuse remains guarded "
                "by run fingerprint and measurement continues"
            ),
            error=error,
        )
        return []


def _offload_measurement_models(models: Mapping[str, nn.Module]) -> None:
    for model in models.values():
        model.to("cpu")
        model.eval()


def _jacobian_stability_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    jacobian = plan["jacobian"]
    return {
        "probe_count": int(jacobian["probe_count"]),
        "randomized_svd_rank": int(jacobian["randomized_svd_rank"]),
        "oversampling": int(jacobian["probe_count"]) - int(jacobian["randomized_svd_rank"]),
        "split_half_role": "advisory_only",
        "automatic_probe_fallback": False,
    }
