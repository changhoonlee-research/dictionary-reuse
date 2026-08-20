"""End-to-end DiR v1.2.4 final-paper experiment pipeline."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from ..artifacts import read_json_file as _read_json, write_json_file as _write_json
from ..training.sparsity import _dictionary_config_for_record
from ..training.trainer import _device_from_config, _make_record

from .common import (
    ARTIFACT_RUN_FILE,
    ARTIFACT_IDENTITY_SCHEMA_VERSION,
    PROGRESS_FILE,
    _canonical_json_sha256,
    _clear_previous_report_files,
    _dictionary_reuse_config,
    _functional_correspondence_config,
    _invalidate_configured_final_artifacts,
    _measurement_contract_sha256,
    _new_artifact_run_id,
    _sha256_file,
    _statistics_contract_sha256,
    _training_contract_sha256,
    _utc_now_iso,
)
from .validation import (
    build_dataset_sample_reference,
    runtime_environment_snapshot,
    validate_config,
)
from .training_stage import run_training_stage
from .measurement_stage import run_measurement_stage
from .finalization import finalize_experiment_artifacts


def _prepare_training_stage_inputs(
    role: dict[str, Any], plan: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    source_payload = dict(role["source_run"])
    source_record = _make_record(source_payload, role)
    source_dictionary = _dictionary_config_for_record(role, source_record)
    training = dict(plan["training"])
    return source_record, source_dictionary, training


def _validate_runtime_before_training(
    *, output_dir: Path, plan: Mapping[str, Any], role: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Keep only essential hard guards before long training."""

    device = _device_from_config(role)
    if bool(plan.get("require_cuda", True)) and device.type != "cuda":
        raise RuntimeError("DiR scientific measurement requires CUDA")
    minimum_free = int(dict(plan.get("execution", {}) or {}).get("minimum_free_disk_bytes", 0))
    free_bytes = int(shutil.disk_usage(output_dir).free)
    if free_bytes < minimum_free:
        raise RuntimeError(
            f"DiR requires at least {minimum_free} free bytes before training; available={free_bytes}"
        )
    environment = runtime_environment_snapshot()
    if bool(plan.get("require_cuda", True)) and not bool(environment.get("cuda_available", False)):
        raise RuntimeError("DiR CUDA runtime is unavailable before long training")
    return device, environment


def run_experiment(config_path: str | Path) -> str:
    config_path = Path(config_path).expanduser().resolve()
    raw_config = _read_json(config_path)
    validation = validate_config(raw_config)
    effective_config_sha256 = _canonical_json_sha256(raw_config)
    training_contract_sha256 = _training_contract_sha256(raw_config)
    measurement_contract_sha256 = _measurement_contract_sha256(raw_config)
    statistics_contract_sha256 = _statistics_contract_sha256(raw_config)
    if not validation["training_enabled"]:
        raise ValueError("run_experiment requires measurement mode")

    plan = _functional_correspondence_config(raw_config)
    role = _dictionary_reuse_config(raw_config)
    console_enabled = bool(
        dict(role.get("runtime", {}) or {}).get("console_logging_enabled", False)
    )

    def console(message: str) -> None:
        if console_enabled:
            print(message, flush=True)

    output_dir = Path(str(raw_config["paths"]["output_dir"])).expanduser().resolve()
    work_dir = Path(str(raw_config["paths"]["work_dir"])).expanduser().resolve()
    checkpoint_dir = Path(str(raw_config["paths"]["checkpoint_dir"])).expanduser().resolve()
    try:
        work_dir.relative_to(output_dir)
    except ValueError as error:
        raise ValueError("DiR paths.work_dir must be inside paths.output_dir") from error
    try:
        checkpoint_dir.relative_to(output_dir)
    except ValueError:
        pass
    else:
        raise ValueError("DiR paths.checkpoint_dir must be outside paths.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_report_files(output_dir)
    invalidated_final_artifacts = _invalidate_configured_final_artifacts(raw_config)

    artifact_run_id = _new_artifact_run_id()
    artifact_created_at = _utc_now_iso()
    artifact_run_path = output_dir / ARTIFACT_RUN_FILE
    artifact_identity = {
        "schema_version": ARTIFACT_IDENTITY_SCHEMA_VERSION,
        "artifact_run_id": artifact_run_id,
        "created_at": artifact_created_at,
        "effective_config_sha256": effective_config_sha256,
    }
    _write_json(artifact_run_path, artifact_identity)
    artifact_identity_sha256 = _sha256_file(artifact_run_path)

    def write_progress(payload: Mapping[str, Any]) -> None:
        progress = dict(payload)
        progress["artifact_run_id"] = artifact_run_id
        _write_json(output_dir / PROGRESS_FILE, progress)

    post_training_warnings: list[dict[str, Any]] = []

    def record_post_training_warning(
        *, stage: str, message: str, error: Exception | None = None
    ) -> None:
        warning = {
            "stage": str(stage),
            "message": str(message),
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "recorded_at": _utc_now_iso(),
        }
        post_training_warnings.append(warning)
        detail = f"{type(error).__name__}: {error}" if error is not None else str(message)
        console(f"[WARNING] {stage} | {detail}")
        try:
            write_progress(
                {
                    "stage": "post_training_measurement",
                    "status": "warning_continue",
                    "warning": warning,
                    "post_training_warning_count": len(post_training_warnings),
                }
            )
        except Exception as progress_error:
            warning["progress_write_error_type"] = type(progress_error).__name__
            warning["progress_write_error"] = str(progress_error)

    def write_post_training_progress(payload: Mapping[str, Any], *, stage: str) -> bool:
        try:
            write_progress(payload)
            return True
        except Exception as error:
            record_post_training_warning(
                stage=stage,
                message="post-training progress persistence failed; scientific work continues",
                error=error,
            )
            return False

    def write_post_training_json(
        path: Path, payload: Any, *, stage: str, message: str
    ) -> bool:
        try:
            _write_json(path, payload)
            return True
        except Exception as error:
            record_post_training_warning(stage=stage, message=message, error=error)
            return False

    write_progress(
        {
            "stage": "validation",
            "status": "running",
            "invalidated_previous_final_artifacts": invalidated_final_artifacts,
        }
    )
    device, runtime_environment = _validate_runtime_before_training(
        output_dir=output_dir, plan=plan, role=role
    )
    dataset_sample_reference = build_dataset_sample_reference(
        role, dict(plan["samples"])
    )
    write_progress({"stage": "validation", "status": "completed"})

    console("\n[TRAIN]")
    source_record, source_dictionary, training = _prepare_training_stage_inputs(role, plan)
    training_stage = run_training_stage(
        artifact_run_id=artifact_run_id,
        checkpoint_dir=checkpoint_dir,
        device=device,
        effective_config_sha256=effective_config_sha256,
        output_dir=output_dir,
        plan=plan,
        post_training_warnings=post_training_warnings,
        role=role,
        source_dictionary=source_dictionary,
        source_record=source_record,
        training=training,
        training_contract_sha256=training_contract_sha256,
        write_progress=write_progress,
        write_post_training_json=write_post_training_json,
        write_post_training_progress=write_post_training_progress,
        record_post_training_warning=record_post_training_warning,
    )

    console("\n[MEASURE]")
    measurement_stage = run_measurement_stage(
        artifact_run_id=artifact_run_id,
        checkpoint_provenance_sha256=training_stage.checkpoint_provenance_sha256,
        dataset_sample_reference=dataset_sample_reference,
        device=device,
        dir_source=training_stage.dir_source,
        dir_targets=training_stage.dir_targets,
        endpoint_paths=training_stage.endpoint_paths,
        measurement_contract_sha256=measurement_contract_sha256,
        models=training_stage.models,
        output_dir=output_dir,
        ownership=training_stage.ownership,
        plan=plan,
        role=role,
        statistics_contract_sha256=statistics_contract_sha256,
        training_contract_sha256=training_contract_sha256,
        training_csv_sha256=training_stage.training_csv_sha256,
        work_dir=work_dir,
        write_post_training_json=write_post_training_json,
        write_post_training_progress=write_post_training_progress,
        record_post_training_warning=record_post_training_warning,
    )

    console("\n[REPORT]")
    final_result = finalize_experiment_artifacts(
        raw_config=raw_config,
        config_path=config_path,
        output_dir=output_dir,
        work_dir=work_dir,
        checkpoint_dir=checkpoint_dir,
        artifact_run_id=artifact_run_id,
        artifact_run_path=artifact_run_path,
        artifact_identity_sha256=artifact_identity_sha256,
        plan=plan,
        training=training,
        runtime_environment=runtime_environment,
        dataset_sample_reference=dataset_sample_reference,
        post_training_warnings=post_training_warnings,
        training_stage=training_stage,
        measurement_stage=measurement_stage,
        measurement_contract_sha256=measurement_contract_sha256,
        statistics_contract_sha256=statistics_contract_sha256,
        write_progress=write_progress,
        write_post_training_json=write_post_training_json,
        record_post_training_warning=record_post_training_warning,
    )
    return final_result
