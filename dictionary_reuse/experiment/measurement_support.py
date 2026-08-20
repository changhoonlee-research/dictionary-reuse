"""Shared measurement runtime helpers plus fail-soft statistics finalization."""

from __future__ import annotations


# Measurement runtime support
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from ..measurements.corruption import _patching_corruption_validity_audit
from .common import (
    _read_gzip_json,
    _reusable_shard_status,
    _safe_module_file_name,
)
from .matrix import MEASUREMENT_PAIRS
from .reporting import _run_measurement_module
from .runtime import _measurement_batches

MeasurementBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class MeasurementPair:
    """One canonical Source–Target comparison from the final matrix."""

    left: nn.Module
    right: nn.Module
    same_task: bool
    shared_head_task: str | None
    task_key: str
    family: str


class MeasurementModuleRunner:
    """Run one measurement module while owning model residency and progress logs."""

    _PAIR_LABELS = {
        "dir_same_task": "DiR Same-task",
        "dir_dictionary_fixed": "DiR Dictionary-Fixed",
        "dir_dictionary_trainable": "DiR Dictionary-Trainable",
        "dense_same_task": "Dense Same-task",
        "dense_different_task": "Dense Full-Transfer",
    }

    def __init__(
        self,
        *,
        all_models: Mapping[str, nn.Module],
        device: torch.device,
        module_results: dict[str, Any],
        module_status: dict[str, Any],
        shard_dir: Path,
        module_manifest_path: Path,
        run_fingerprint: str,
        model_checkpoint_paths: Mapping[int, Path],
        warning_recorder: Callable[..., None],
        console: Callable[[str], None],
    ) -> None:
        self._all_models = all_models
        self._device = device
        self._module_results = module_results
        self._module_status = module_status
        self._shard_dir = shard_dir
        self._module_manifest_path = module_manifest_path
        self._run_fingerprint = run_fingerprint
        self._model_checkpoint_paths = model_checkpoint_paths
        self._warning_recorder = warning_recorder
        self._console = console
        self._active_model_ids: set[int] = set()
        self._last_context: tuple[str, str] | None = None
        self._last_group: str | None = None

    def activate_models(self, module_models: Mapping[str, nn.Module]) -> None:
        """Keep only models required by the current module on the execution device."""

        requested_ids = {id(model) for model in module_models.values()}
        if requested_ids == self._active_model_ids:
            return
        for model in self._all_models.values():
            target = self._device if id(model) in requested_ids else torch.device("cpu")
            model.to(target)
            model.eval()
        self._active_model_ids = requested_ids
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

    def run(
        self,
        *,
        name: str,
        function: Callable[[], Any],
        module_models: Mapping[str, nn.Module],
        log_progress: bool = True,
    ) -> None:
        if log_progress:
            self._log_module_group(name)
        self.activate_models(module_models)
        phase = "core" if ".core." in name else "supplementary"
        _run_measurement_module(
            name=name,
            phase=phase,
            function=function,
            models=module_models,
            module_results=self._module_results,
            module_status=self._module_status,
            shard_dir=self._shard_dir,
            module_manifest_path=self._module_manifest_path,
            run_fingerprint=self._run_fingerprint,
            model_checkpoint_paths=self._model_checkpoint_paths,
            warning_recorder=self._warning_recorder,
        )

    def reusable_shard_exists(self, module_name: str) -> bool:
        """Return whether resume may reuse this exact deterministic module shard."""

        shard_path = self._shard_dir / _safe_module_file_name(module_name)
        if not shard_path.is_file():
            return False
        try:
            cached = _read_gzip_json(shard_path)
            if str(cached.get("run_fingerprint", "")) != self._run_fingerprint:
                return False
            if str(cached.get("name", "")) != str(module_name):
                return False
            status = dict(cached.get("status", {}) or {})
            return _reusable_shard_status(str(status.get("status", "")))
        except Exception:
            return False

    def _log_module_group(self, module_name: str) -> None:
        pair_name, task_name, group = _measurement_group(module_name)
        context = (pair_name, task_name)
        if context != self._last_context:
            pair_label = self._PAIR_LABELS.get(pair_name, pair_name.replace("_", " "))
            task_suffix = f" | {task_name}" if task_name else ""
            self._console(f"  {pair_label}{task_suffix}")
            self._last_context = context
            self._last_group = None
        if group != self._last_group:
            self._console(f"    {group}")
            self._last_group = group


class PatchingFamilyValidityCache:
    """Compute one shared valid-sample intersection per task-family/corruption."""

    def __init__(
        self,
        *,
        pairs: Mapping[str, MeasurementPair],
        runner: MeasurementModuleRunner,
        plan: Mapping[str, Any],
        device: torch.device,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self._pairs = pairs
        self._runner = runner
        self._plan = plan
        self._device = device
        self._mean = mean
        self._std = std
        self._cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def get(
        self,
        *,
        current_pair_name: str,
        task_key: str,
        corruption: str,
        batches: Sequence[MeasurementBatch],
        same_head: bool,
        current_left: nn.Module,
        current_right: nn.Module,
    ) -> dict[str, Any]:
        current_pair = self._pairs.get(current_pair_name)
        if current_pair is None:
            raise ValueError(f"Unknown DiR patching condition: {current_pair_name}")
        family = "same_task" if current_pair.same_task else "different_task"
        cache_key = (family, str(task_key), str(corruption))
        if cache_key not in self._cache:
            self._cache[cache_key] = self._compute_family_context(
                family=family,
                task_key=task_key,
                corruption=corruption,
                batches=batches,
                same_head=same_head,
                current_left=current_left,
                current_right=current_right,
            )
        return self._cache[cache_key]

    def _compute_family_context(
        self,
        *,
        family: str,
        task_key: str,
        corruption: str,
        batches: Sequence[MeasurementBatch],
        same_head: bool,
        current_left: nn.Module,
        current_right: nn.Module,
    ) -> dict[str, Any]:
        pair_names = tuple(
            name
            for name, pair in self._pairs.items()
            if ("same_task" if pair.same_task else "different_task") == family
        )
        pair_audits: dict[str, dict[str, Any]] = {}
        all_audits: list[Mapping[str, Any]] = []
        model_audit_cache: dict[int, Mapping[str, Any]] = {}
        try:
            for pair_name in pair_names:
                pair = self._pairs[pair_name]
                pair_audits[pair_name] = {}
                for side, model in (("left", pair.left), ("right", pair.right)):
                    audit = model_audit_cache.get(id(model))
                    if audit is None:
                        self._runner.activate_models({"audit": model})
                        audit = self._audit_model(
                            model=model,
                            batches=batches,
                            corruption=corruption,
                            same_head=same_head,
                        )
                        model_audit_cache[id(model)] = audit
                        all_audits.append(audit)
                    pair_audits[pair_name][side] = audit
            common_masks = _intersect_validity_masks(all_audits)
            return {
                "family": family,
                "pair_names": list(pair_names),
                "endpoint_role_count": len(all_audits),
                "pair_audits": pair_audits,
                "common_valid_masks": common_masks,
                "common_valid_sample_count_by_view": {
                    key: int(mask.sum()) for key, mask in common_masks.items()
                },
                "contract": (
                    "all_compared_methods_in_the_same_task_family_share_one_common_valid_"
                    "sample_intersection_per_view"
                ),
            }
        finally:
            self._runner.activate_models({"left": current_left, "right": current_right})

    def _audit_model(
        self,
        *,
        model: nn.Module,
        batches: Sequence[MeasurementBatch],
        corruption: str,
        same_head: bool,
    ) -> Mapping[str, Any]:
        patching = self._plan["patching"]
        return _patching_corruption_validity_audit(
            model,
            batches,
            device=self._device,
            corruption=str(corruption),
            mean=self._mean,
            std=self._std,
            same_head=bool(same_head),
            minimum_relative_effect=float(patching["minimum_relative_corruption_effect"]),
            minimum_prediction_retention=float(patching["minimum_prediction_retention"]),
            noise_sigma=float(patching["noise_sigma"]),
            noise_seed=int(patching["noise_seed"]),
            blur_sigma=float(patching["blur_sigma"]),
            blur_kernel_size=int(patching["blur_kernel_size"]),
            blur_padding=str(patching["blur_padding"]),
            mask_size=int(patching["mask_size"]),
            mask_positions=tuple(patching["mask_positions"]),
            mask_fill=str(patching["mask_fill"]),
        )


def build_measurement_pairs(models: Mapping[str, nn.Module]) -> dict[str, MeasurementPair]:
    return {
        spec.condition: MeasurementPair(
            left=models[spec.left_model],
            right=models[spec.right_model],
            same_task=bool(spec.same_task),
            shared_head_task=(spec.task_key if spec.same_task else None),
            task_key=str(spec.task_key),
            family=str(spec.family),
        )
        for spec in MEASUREMENT_PAIRS
    }


def prepare_task_batches(
    role: Mapping[str, Any], samples: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Prepare deterministic nested measurement/probe samples for both tasks."""

    max_count = int(samples["representation"])
    task_batches: dict[str, dict[str, Any]] = {}
    sample_manifest: dict[str, Any] = {
        "selection": "sha256(task/split/original_index) ascending",
        "nested": "128⊂256⊂512⊂1024",
        "probe_split_contract": "train[0:4096], train[4096:5120], eval[0:1024]",
        "tasks": {},
    }
    for task_key in ("task1", "task2"):
        batches, ids = _measurement_batches(role, task_key=task_key, count=max_count)
        probe_train, probe_train_ids = _measurement_batches(
            role,
            task_key=task_key,
            count=int(samples["probe_train"]),
            loader_split_name="train",
            start=0,
        )
        probe_validation, probe_validation_ids = _measurement_batches(
            role,
            task_key=task_key,
            count=int(samples["probe_validation"]),
            loader_split_name="train",
            start=int(samples["probe_train"]),
        )
        probe_test, probe_test_ids = _measurement_batches(
            role,
            task_key=task_key,
            count=int(samples["probe_test"]),
            loader_split_name="eval",
            start=0,
        )
        task_batches[task_key] = {
            "all": batches,
            "probe_train": probe_train,
            "probe_validation": probe_validation,
            "probe_test": probe_test,
        }
        sample_manifest["tasks"][task_key] = {
            "ids_128": ids[:128],
            "ids_256": ids[:256],
            "ids_512": ids[:512],
            "ids_1024": ids[:1024],
            "probe_train_ids": probe_train_ids,
            "probe_validation_ids": probe_validation_ids,
            "probe_test_ids": probe_test_ids,
            "probe_splits_disjoint": bool(
                set(probe_train_ids).isdisjoint(probe_validation_ids)
            ),
        }
    return task_batches, sample_manifest


def truncate_batches(
    batches: Sequence[MeasurementBatch], count: int
) -> list[MeasurementBatch]:
    output: list[MeasurementBatch] = []
    remaining = int(count)
    for images, labels, ids in batches:
        if remaining <= 0:
            break
        take = min(remaining, int(images.shape[0]))
        output.append((images[:take], labels[:take], ids[:take]))
        remaining -= take
    if remaining != 0:
        raise RuntimeError(f"DiR requested more samples than available: missing={remaining}")
    return output


def phase_status(module_status: Mapping[str, Any], names: Sequence[str]) -> str:
    from .common import _completed_status, _superseded_status

    statuses = [
        str(module_status[name].get("status", ""))
        for name in names
        if not _superseded_status(str(module_status[name].get("status", "")))
    ]
    if any(value.startswith("warning_") for value in statuses):
        return "partial"
    if any(value.startswith("inconclusive_") for value in statuses):
        return "inconclusive"
    if statuses and all(_completed_status(value) for value in statuses):
        return "complete"
    return "partial"


def _measurement_group(module_name: str) -> tuple[str, str, str]:
    parts = str(module_name).split(".")
    pair_name = parts[0] if parts else "measurement"
    task_name = parts[1] if len(parts) > 1 else ""
    family = parts[3] if len(parts) > 3 else "measurement"
    if family in {"block_update", "direct_function", "direct_wide_windows"}:
        group = "direct"
    elif family in {"ablation", "patching"}:
        group = "causal"
    elif family == "jacobian":
        group = "jacobian"
    elif family in {"representation_geometry", "attention_transport"}:
        group = "representation/attention"
    elif family in {"gradient_profile", "spectral_response"}:
        group = "gradient/spectral"
    elif family in {"full_block_swap", "cross_model_activation_patching", "probes"}:
        group = "swap/patch/probe"
    elif family == "atom_group_ablation":
        group = "atom ablation"
    else:
        group = family.replace("_", " ")
    return pair_name, task_name, group


def _intersect_validity_masks(audits: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    if not audits:
        raise RuntimeError("DiR patching family audit has no endpoint roles")
    common_keys = set(dict(audits[0]["valid_masks"]))
    for audit in audits[1:]:
        common_keys &= set(dict(audit["valid_masks"]))
    if not common_keys:
        raise RuntimeError("DiR patching family audit has no common validity views")

    common_masks: dict[str, torch.Tensor] = {}
    for key in sorted(common_keys):
        masks = [dict(audit["valid_masks"])[key].detach().bool().cpu() for audit in audits]
        sample_count = int(masks[0].numel())
        if any(int(mask.numel()) != sample_count for mask in masks[1:]):
            raise ValueError("DiR patching family validity masks have unequal sample counts")
        common = masks[0].clone()
        for mask in masks[1:]:
            common &= mask
        common_masks[key] = common
    return common_masks


# Statistics and status finalization
from .common import CORE_MEASUREMENTS_FILE, _completed_status, _superseded_status
from .reporting import _control_comparisons, _jacobian_rank32_summary, _primary_metric_summary, _supporting_metric_summary
from .reporting import _matrix_reports
from .runtime import _current_measurement_shards


@dataclass
class MeasurementStatisticsResult:
    completed: int
    warning_count: int
    inconclusive_count: int
    superseded_count: int
    core_names: list[str]
    supplementary_names: list[str]
    core_measurement_status: str
    supplementary_measurement_status: str
    measurement_status: str
    condition_comparisons: dict[str, Any]
    primary_results: dict[str, Any]
    supporting_results: dict[str, Any]
    jacobian_rank32_results: dict[str, Any]
    current_measurement_shard_count: int


def finalize_measurement_statistics(
    *,
    artifact_run_id: str,
    checkpoint_file_identity: Mapping[str, Any],
    checkpoint_provenance_sha256: str,
    console: Callable[[str], None],
    jacobian_stability_contract: Mapping[str, Any],
    measurement_shard_dir: Path,
    module_results: Mapping[str, Any],
    module_status: Mapping[str, Any],
    output_dir: Path,
    plan: Mapping[str, Any],
    record_warning: Callable[..., None],
    run_fingerprint: str,
    sample_manifest_sha256: str,
    statistics_contract_sha256: str,
    statistics_fingerprint: str,
    training_contract_sha256: str,
    training_csv_sha256: str,
    write_json: Callable[..., bool],
    write_progress: Callable[..., bool],
) -> MeasurementStatisticsResult:
    samples = dict(plan["samples"])
    statistics_seed = int(plan["statistics_seed"])

    console("\n[STATISTICS]")
    console("  summaries")
    write_progress(
        {"stage": "statistics", "status": "running"},
        stage="statistics_progress",
    )

    reportable = _failsoft_statistics_call(
        function=lambda: _matrix_reports(
            module_results, samples=samples, seed=statistics_seed
        ),
        fallback={},
        stage="matrix_statistics",
        message=(
            "matrix statistics generation failed after raw measurements; preserve raw module "
            "results and continue summary/artifact finalization with statistics unavailable"
        ),
        record_warning=record_warning,
    )
    control_comparisons = _failsoft_statistics_call(
        function=lambda: _control_comparisons(
            module_results,
            module_status,
            bootstrap_iterations=int(samples["bootstrap_iterations"]),
            seed=statistics_seed + 500000,
        ),
        fallback={},
        stage="control_comparison_statistics",
        message=(
            "final matrix comparison statistics failed after raw measurements; preserve raw "
            "results and continue with comparison statistics unavailable"
        ),
        record_warning=record_warning,
    )

    write_json(
        output_dir / CORE_MEASUREMENTS_FILE,
        {
            "modules": reportable,
            "condition_comparisons": control_comparisons,
            "status": module_status,
            "jacobian_stability_contract": dict(jacobian_stability_contract),
            "measurement_run_fingerprint": run_fingerprint,
            "sample_manifest_sha256": sample_manifest_sha256,
            "checkpoint_provenance_sha256": checkpoint_provenance_sha256,
            "checkpoint_file_identity": checkpoint_file_identity,
            "training_contract_sha256": training_contract_sha256,
            "training_csv_sha256": training_csv_sha256,
            "statistics_contract_sha256": statistics_contract_sha256,
            "statistics_fingerprint": statistics_fingerprint,
            "artifact_run_id": artifact_run_id,
        },
        stage="core_report_persistence",
        message=(
            "core measurement JSON persistence failed after measurement; in-memory module "
            "results remain available and summary/artifact finalization continues"
        ),
    )

    core_names = [name for name in module_status if ".core." in name]
    supplementary_names = [name for name in module_status if ".supplementary." in name]
    core_status = phase_status(module_status, core_names)
    supplementary_status = phase_status(module_status, supplementary_names)
    overall_status = _combine_phase_status(core_status, supplementary_status)

    primary_results = _summary_or_warning(
        function=lambda: _primary_metric_summary(
            module_results,
            module_status,
            bootstrap_iterations=int(samples["bootstrap_iterations"]),
            global_permutations=int(samples["global_permutations"]),
            depth_band_permutations=int(samples["depth_band_permutations"]),
            seed=statistics_seed + 900000,
            reportable_modules=reportable,
        ),
        stage="primary_metric_summary",
        message="primary metric summary failed after raw measurements; preserve raw results and continue",
        record_warning=record_warning,
    )
    supporting_results = _summary_or_warning(
        function=lambda: _supporting_metric_summary(
            module_results,
            module_status,
            global_permutations=int(samples["global_permutations"]),
            depth_band_permutations=int(samples["depth_band_permutations"]),
            seed=statistics_seed + 1200000,
            reportable_modules=reportable,
        ),
        stage="supporting_metric_summary",
        message="supporting metric summary failed after raw measurements; preserve raw results and continue",
        record_warning=record_warning,
    )
    jacobian_rank32_results = _summary_or_warning(
        function=lambda: _jacobian_rank32_summary(
            module_results,
            module_status,
            bootstrap_iterations=int(samples["bootstrap_iterations"]),
            global_permutations=int(samples["global_permutations"]),
            depth_band_permutations=int(samples["depth_band_permutations"]),
            seed=statistics_seed + 1500000,
            reportable_modules=reportable,
        ),
        stage="jacobian_rank32_summary",
        message="rank-32 Jacobian summary failed after raw measurements; preserve raw results and continue",
        record_warning=record_warning,
    )
    shard_count = _current_shard_count_or_warning(
        measurement_shard_dir=measurement_shard_dir,
        run_fingerprint=run_fingerprint,
        module_status=module_status,
        record_warning=record_warning,
    )
    console("  done")

    return MeasurementStatisticsResult(
        completed=sum(
            _completed_status(str(value.get("status", "")))
            for value in module_status.values()
        ),
        warning_count=sum(
            str(value.get("status", "")).startswith("warning_")
            for value in module_status.values()
        ),
        inconclusive_count=sum(
            str(value.get("status", "")).startswith("inconclusive_")
            for value in module_status.values()
        ),
        superseded_count=sum(
            _superseded_status(str(value.get("status", "")))
            for value in module_status.values()
        ),
        core_names=core_names,
        supplementary_names=supplementary_names,
        core_measurement_status=core_status,
        supplementary_measurement_status=supplementary_status,
        measurement_status=overall_status,
        condition_comparisons=control_comparisons,
        primary_results=primary_results,
        supporting_results=supporting_results,
        jacobian_rank32_results=jacobian_rank32_results,
        current_measurement_shard_count=shard_count,
    )


def _failsoft_statistics_call(
    *,
    function: Callable[[], dict[str, Any]],
    fallback: dict[str, Any],
    stage: str,
    message: str,
    record_warning: Callable[..., None],
) -> dict[str, Any]:
    try:
        return function()
    except Exception as error:
        record_warning(stage=stage, message=message, error=error)
        return fallback


def _summary_or_warning(
    *,
    function: Callable[[], dict[str, Any]],
    stage: str,
    message: str,
    record_warning: Callable[..., None],
) -> dict[str, Any]:
    try:
        return function()
    except Exception as error:
        record_warning(stage=stage, message=message, error=error)
        return {
            "status": "warning_statistics_exception",
            "error": f"{type(error).__name__}: {error}",
        }


def _combine_phase_status(core_status: str, supplementary_status: str) -> str:
    if "partial" in {core_status, supplementary_status}:
        return "partial"
    if "inconclusive" in {core_status, supplementary_status}:
        return "inconclusive"
    return "complete"


def _current_shard_count_or_warning(
    *,
    measurement_shard_dir: Path,
    run_fingerprint: str,
    module_status: Mapping[str, Any],
    record_warning: Callable[..., None],
) -> int:
    try:
        return len(
            _current_measurement_shards(
                measurement_shard_dir,
                run_fingerprint=run_fingerprint,
                module_status=module_status,
            )
        )
    except Exception as error:
        record_warning(
            stage="measurement_shard_inventory_summary",
            message=(
                "measurement shard inventory could not be summarized; in-memory results remain "
                "authoritative and summary/artifact finalization continues"
            ),
            error=error,
        )
        return 0
