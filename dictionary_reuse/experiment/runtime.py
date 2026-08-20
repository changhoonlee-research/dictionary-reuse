"""Training orchestration, checkpoints, resume state, model persistence, and measurement shards."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..artifacts import write_json_file as _write_json
from ..training import LearningRateProfile, build_eval_loader, build_train_loader, train_model
from ..training.engine import _build_task_subset
from ..training.trainer import _make_profile

from .common import (
    ARTIFACT_IDENTITY_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_CHECKPOINT_COUNT,
    MEASUREMENT_SHARD_DIRECTORY,
    ZIP_MANIFEST_FILE,
    _best_effort_unlink,
    _canonical_json_sha256,
    _read_gzip_json,
    _safe_module_file_name,
    _sha256_file,
    _utc_now_iso,
    _write_csv,
)

class _IndexedDataset(Dataset):
    def __init__(self, subset: Any, *, task_key: str, split: str) -> None:
        self.subset = subset
        self.task_key = str(task_key)
        self.split = str(split)
        source_indices = getattr(subset, "indices", list(range(len(subset))))
        self.order = sorted(
            range(len(subset)),
            key=lambda local: hashlib.sha256(
                f"{self.task_key}/{self.split}/{int(source_indices[local])}".encode("utf-8")
            ).hexdigest(),
        )
        self.source_indices = source_indices

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int]:
        local = self.order[item]
        image, label = self.subset[local]
        return image, int(label), int(self.source_indices[local])

def _measurement_batches(
    role: dict[str, Any],
    *,
    task_key: str,
    count: int,
    loader_split_name: str = "eval",
    start: int = 0,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], list[int]]:
    subset = _build_task_subset(role, task_key=task_key, loader_split_name=loader_split_name)
    dataset = _IndexedDataset(subset, task_key=task_key, split=loader_split_name)
    start = max(0, int(start))
    stop = min(start + int(count), len(dataset))
    selected = torch.utils.data.Subset(dataset, list(range(start, stop)))
    loader = DataLoader(
        selected,
        batch_size=int(role["runtime"].get("eval_batch_size", 128)),
        shuffle=False,
        num_workers=int(role["runtime"].get("num_workers", 2)),
        pin_memory=bool(torch.cuda.is_available()),
    )
    batches = [(batch[0], batch[1], batch[2]) for batch in loader]
    ids = [int(value) for _images, _labels, values in batches for value in values.tolist()]
    return batches, ids

def _profile_config(role: dict[str, Any], name: str) -> LearningRateProfile:
    return _make_profile(role, name)

def _training_options(
    role: dict[str, Any],
    *,
    natural_profile: str,
    phase_profile: str,
    gradient_profile: str,
) -> dict[str, Any]:
    return {
        "natural_sparsity_config": deepcopy(role.get("natural_sparsity_profiles", {}).get(natural_profile, {})),
        "phase_schedule_config": deepcopy(role.get("phase_schedule_profiles", {}).get(phase_profile, {})),
        "gradient_clip_config": deepcopy(role.get("gradient_clip_profiles", {}).get(gradient_profile, {})),
        "numerical_guard_config": deepcopy(role.get("numerical_guard", {})),
    }

def _offload_trained_model(model: nn.Module, *, execution_device: torch.device) -> nn.Module:
    """Move a completed endpoint off the accelerator before the next training run."""

    model.to(torch.device("cpu"))
    model.eval()
    if execution_device.type == "cuda":
        torch.cuda.empty_cache()
    return model

def _train_one(
    *,
    model: nn.Module,
    role: dict[str, Any],
    task_key: str,
    run_id: str,
    model_family: str,
    basis_type: str,
    profile_name: str,
    epochs: int,
    record_epochs: set[int],
    data_order_seed: int,
    device: torch.device,
    natural_profile: str = "",
    phase_profile: str = "",
    gradient_profile: str = "",
    step_observer: Any | None = None,
    epoch_start_observer: Any | None = None,
    post_epoch_observer: Any | None = None,
    support_commit_post_observer: Any | None = None,
    preserve_relative_coordinate_corrections_at_commit: bool = False,
    snapshot_epochs: set[int] | None = None,
    skip_initial_dictionary_normalization: bool = False,
) -> tuple[nn.Module, list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    train_loader = build_train_loader(role, task_key=task_key, data_order_seed=int(data_order_seed))
    eval_loader = build_eval_loader(role, task_key=task_key)
    options = _training_options(
        role,
        natural_profile=natural_profile,
        phase_profile=phase_profile,
        gradient_profile=gradient_profile,
    )
    options["natural_sparsity_config"][
        "forward_routed_hard_support_commit_fold_relative_coordinate_into_dictionary_scale"
    ] = not bool(preserve_relative_coordinate_corrections_at_commit)
    trained, curves, _usage, _phase, _step, snapshots, _optimizer = train_model(
        model,
        train_loader,
        eval_loader,
        device=device,
        profile=_profile_config(role, profile_name),
        model_family=model_family,
        run_id=run_id,
        task_id=task_key,
        basis_type=basis_type,
        total_epochs=int(epochs),
        max_batches_per_epoch=len(train_loader),
        record_epochs=set(int(value) for value in record_epochs),
        include_epoch0_eval=True,
        record_eval_max_batches=4,
        final_eval_max_batches=None,
        console_config=role,
        snapshot_epochs=snapshot_epochs or set(),
        natural_sparsity_config=options["natural_sparsity_config"],
        phase_schedule_config=options["phase_schedule_config"],
        gradient_clip_config=options["gradient_clip_config"],
        numerical_guard_config=options["numerical_guard_config"],
        step_observer=step_observer,
        epoch_start_observer=epoch_start_observer,
        post_epoch_training_observer=post_epoch_observer,
        support_commit_post_observer=support_commit_post_observer,
        skip_initial_dictionary_normalization=bool(skip_initial_dictionary_normalization),
    )
    return trained, curves, snapshots

def _reset_dense_head(model: nn.Module, *, seed: int) -> None:
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        nn.init.trunc_normal_(model.classification_head.weight, std=0.02)
        if model.classification_head.bias is not None:
            nn.init.zeros_(model.classification_head.bias)
    finally:
        torch.random.set_rng_state(generator_state)

def _save_model(path: Path, model: nn.Module, *, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "metadata": dict(metadata),
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)

def _checkpoint_specs(
    checkpoint_dir: Path,
    training: Mapping[str, Any],
    *,
    dir_model_family: str,
) -> dict[str, dict[str, Any]]:
    """Return the immutable 12-file checkpoint inventory for the final matrix."""

    def spec(
        key: str,
        filename: str,
        *,
        condition: str,
        epoch: int,
        checkpoint_role: str,
        model_family: str,
        task_key: str,
        model_seed: int | None,
        data_order_seed: int,
        initialization_source: str = "independent_seed_initialization",
        head_reset_seed: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "path": checkpoint_dir / filename,
            "checkpoint_filename": filename,
            "condition": condition,
            "epoch": int(epoch),
            "checkpoint_role": checkpoint_role,
            "model_family": model_family,
            "task_key": task_key,
            "data_order_seed": int(data_order_seed),
            "initialization_source": str(initialization_source),
        }
        if model_seed is not None:
            payload["model_seed"] = int(model_seed)
        if head_reset_seed is not None:
            payload["head_reset_seed"] = int(head_reset_seed)
        return key, payload

    dir_source_epoch = int(training["dir_source_a_epochs"])
    dir_target_epoch = int(training["dir_target_epochs"])
    dense_source_epoch = int(training["dense_source_a_epochs"])
    dense_target_epoch = int(training["dense_target_epochs"])
    intermediate_epochs = sorted(
        {int(value) for value in training["checkpoint_epochs"]} - {dir_target_epoch}
    )
    if len(intermediate_epochs) != 1:
        raise ValueError("DiR requires exactly one Target intermediate checkpoint epoch")
    intermediate_epoch = int(intermediate_epochs[0])

    def target_pair(
        key: str,
        filename_prefix: str,
        *,
        condition: str,
        model_family: str,
        task_key: str,
        model_seed: int | None,
        data_order_seed: int,
        initialization_source: str,
        head_reset_seed: int | None = None,
        target_epoch: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        return [
            spec(
                f"{key}_e{intermediate_epoch}",
                f"{filename_prefix}_e{intermediate_epoch}.pt",
                condition=condition,
                epoch=intermediate_epoch,
                checkpoint_role="snapshot",
                model_family=model_family,
                task_key=task_key,
                model_seed=model_seed,
                data_order_seed=data_order_seed,
                initialization_source=initialization_source,
                head_reset_seed=head_reset_seed,
            ),
            spec(
                key,
                f"{filename_prefix}_e{target_epoch}.pt",
                condition=condition,
                epoch=target_epoch,
                checkpoint_role="endpoint",
                model_family=model_family,
                task_key=task_key,
                model_seed=model_seed,
                data_order_seed=data_order_seed,
                initialization_source=initialization_source,
                head_reset_seed=head_reset_seed,
            ),
        ]

    entries: list[tuple[str, dict[str, Any]]] = [
        spec(
            "dir_source",
            f"dir_source_task1_e{dir_source_epoch}.pt",
            condition="dir_source_a",
            epoch=dir_source_epoch,
            checkpoint_role="endpoint",
            model_family=dir_model_family,
            task_key="task1",
            model_seed=int(training["dir_source_seed"]),
            data_order_seed=int(training["dir_source_data_order_seed"]),
        ),
    ]
    entries += target_pair(
        "dir_same_task",
        "dir_same_task",
        condition="dir_same_task",
        model_family=dir_model_family,
        task_key="task1",
        model_seed=int(training["dir_same_task_seed"]),
        data_order_seed=int(training["dir_same_task_data_order_seed"]),
        initialization_source="fresh_target_then_source_active_D_and_D_owned_scales_copy_C_fresh",
        target_epoch=dir_target_epoch,
    )
    entries += target_pair(
        "dir_dictionary_fixed",
        "dir_dictionary_fixed_task2",
        condition="dir_dictionary_fixed",
        model_family=dir_model_family,
        task_key="task2",
        model_seed=None,
        data_order_seed=int(training["different_task_data_order_seed"]),
        initialization_source="source_full_backbone_exact_copy_fresh_head_active_D_and_D_owned_scales_fixed",
        head_reset_seed=int(training["dir_different_task_head_seed"]),
        target_epoch=dir_target_epoch,
    )
    entries += target_pair(
        "dir_dictionary_trainable",
        "dir_dictionary_trainable_task2",
        condition="dir_dictionary_trainable",
        model_family=dir_model_family,
        task_key="task2",
        model_seed=None,
        data_order_seed=int(training["different_task_data_order_seed"]),
        initialization_source="source_full_backbone_exact_copy_fresh_head_all_backbone_trainable",
        head_reset_seed=int(training["dir_different_task_head_seed"]),
        target_epoch=dir_target_epoch,
    )
    entries.append(
        spec(
            "dense_source",
            f"dense_source_task1_e{dense_source_epoch}.pt",
            condition="dense_source_a",
            epoch=dense_source_epoch,
            checkpoint_role="endpoint",
            model_family="dense_vit",
            task_key="task1",
            model_seed=int(training["dense_source_seed"]),
            data_order_seed=int(training["dense_source_data_order_seed"]),
        )
    )
    entries += target_pair(
        "dense_same_task",
        "dense_same_task",
        condition="dense_same_task",
        model_family="dense_vit",
        task_key="task1",
        model_seed=int(training["dense_same_task_seed"]),
        data_order_seed=int(training["dense_same_task_data_order_seed"]),
        initialization_source="independent_seed_initialization",
        target_epoch=dense_target_epoch,
    )
    entries += target_pair(
        "dense_different_task",
        "dense_different_task",
        condition="dense_different_task",
        model_family="dense_vit",
        task_key="task2",
        model_seed=None,
        data_order_seed=int(training["different_task_data_order_seed"]),
        initialization_source="dense_source_endpoint_copy_then_head_reset",
        head_reset_seed=int(training["dense_different_task_head_seed"]),
        target_epoch=dense_target_epoch,
    )
    return dict(entries)

def _checkpoint_metadata_for_spec(
    metadata_base: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(metadata_base),
        **{key: value for key, value in spec.items() if key != "path"},
    }

def _validate_checkpoint_identity(
    metadata: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    for key, expected in expected_identity.items():
        if key == "path":
            continue
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"DiR checkpoint identity mismatch: {path.name} field={key} "
                f"expected={expected!r} actual={actual!r}"
            )

def _checkpoint_groups(
    checkpoint_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key, spec in checkpoint_specs.items():
        groups.setdefault(str(spec["condition"]), []).append(str(key))
    return {name: sorted(keys) for name, keys in groups.items()}

def _checkpoint_group_complete(
    checkpoint_specs: Mapping[str, Mapping[str, Any]],
    condition: str,
) -> bool:
    groups = _checkpoint_groups(checkpoint_specs)
    if condition not in groups:
        raise KeyError(f"Unknown DiR checkpoint condition: {condition}")
    return all(Path(checkpoint_specs[key]["path"]).is_file() for key in groups[condition])

def _validate_checkpoint_inventory(
    checkpoint_dir: Path, checkpoint_specs: Mapping[str, Mapping[str, Any]]
) -> str:
    """Validate path ownership while permitting condition-level partial resume."""

    expected_paths = {Path(spec["path"]).resolve() for spec in checkpoint_specs.values()}
    existing_paths = {path.resolve() for path in checkpoint_dir.glob("*.pt") if path.is_file()}
    extras = sorted(str(path) for path in existing_paths - expected_paths)
    if extras:
        raise RuntimeError(
            "DiR found unexpected checkpoint files; isolate or remove them before execution: "
            f"extras={extras}"
        )
    if not existing_paths:
        return "fresh"
    if existing_paths == expected_paths:
        if len(existing_paths) != EXPECTED_CHECKPOINT_COUNT:
            raise RuntimeError(
                f"DiR checkpoint inventory must contain exactly {EXPECTED_CHECKPOINT_COUNT} files"
            )
        return "complete"
    return "partial"

def _validate_existing_checkpoint_files(
    checkpoint_specs: Mapping[str, Mapping[str, Any]],
    *,
    training_contract_sha256: str,
) -> None:
    """Hard-fail stale/corrupt resume files before any new long training starts."""

    for spec in checkpoint_specs.values():
        path = Path(spec["path"])
        if not path.is_file():
            continue
        payload = _load_checkpoint_payload(path)
        metadata = dict(payload.get("metadata", {}) or {})
        _validate_checkpoint_identity(metadata, spec, path=path)
        if str(metadata.get("training_contract_sha256", "")) != str(
            training_contract_sha256
        ):
            raise ValueError(f"DiR checkpoint training contract hash mismatch: {path.name}")

def _training_rows_have_run(
    rows: Sequence[Mapping[str, Any]], run_id: str
) -> bool:
    return any(str(row.get("run_id", "")) == str(run_id) for row in rows)

def _replace_training_rows_for_run(
    rows: Sequence[Mapping[str, Any]],
    run_id: str,
    replacement: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    retained = [
        dict(row) for row in rows if str(row.get("run_id", "")) != str(run_id)
    ]
    retained.extend(dict(row) for row in replacement)
    return retained

def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".temporary")
    shutil.copy2(source, temporary)
    temporary.replace(destination)

def _persist_training_csv_for_resume(
    output_csv_path: Path,
    checkpoint_csv_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    _write_csv(output_csv_path, rows)
    _copy_file_atomic(output_csv_path, checkpoint_csv_path)
    output_hash = _sha256_file(output_csv_path)
    if _sha256_file(checkpoint_csv_path) != output_hash:
        raise RuntimeError("DiR persisted training CSV hash mismatch")
    return output_hash

def _restore_training_csv_from_checkpoint(
    checkpoint_csv_path: Path, output_csv_path: Path
) -> str:
    if not checkpoint_csv_path.is_file():
        raise RuntimeError(
            f"DiR checkpoint resume training CSV is missing: {checkpoint_csv_path}"
        )
    _copy_file_atomic(checkpoint_csv_path, output_csv_path)
    checkpoint_hash = _sha256_file(checkpoint_csv_path)
    if _sha256_file(output_csv_path) != checkpoint_hash:
        raise RuntimeError("DiR restored training CSV hash mismatch")
    return checkpoint_hash

def _remove_stale_measurement_shards(shard_dir: Path, *, run_fingerprint: str) -> list[str]:
    removed: list[str] = []
    for path in sorted(shard_dir.glob("*.json.gz")):
        keep = False
        try:
            payload = _read_gzip_json(path)
            keep = bool(
                isinstance(payload, Mapping)
                and str(payload.get("schema_version", "")) == "dir_measurement_shard_v2"
                and str(payload.get("run_fingerprint", "")) == str(run_fingerprint)
                and str(payload.get("name", ""))
            )
        except Exception:
            keep = False
        if not keep:
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed

def _current_measurement_shards(
    shard_dir: Path,
    *,
    run_fingerprint: str,
    module_status: Mapping[str, Any],
) -> list[Path]:
    current: list[Path] = []
    for name in sorted(module_status):
        path = shard_dir / _safe_module_file_name(name)
        if not path.is_file():
            continue
        try:
            payload = _read_gzip_json(path)
        except Exception:
            continue
        if (
            str(payload.get("schema_version", "")) == "dir_measurement_shard_v2"
            and str(payload.get("run_fingerprint", "")) == str(run_fingerprint)
            and str(payload.get("name", "")) == str(name)
        ):
            current.append(path)
    return current

def _safe_archive_component(value: str) -> str:
    """Return a readable path component safe for a public report archive."""

    cleaned = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in str(value)
    ).strip("._")
    return cleaned or "unnamed"


def _measurement_archive_member_name(path: Path) -> str:
    """Expose one internal hashed shard under a stable semantic archive path."""

    payload = _read_gzip_json(path)
    module_name = str(payload.get("name", "")).strip()
    if not module_name:
        raise RuntimeError(f"measurement shard is missing its module name: {path.name}")
    parts = [_safe_archive_component(part) for part in module_name.split(".") if part]
    if len(parts) < 4:
        raise RuntimeError(f"measurement shard has an invalid module name: {module_name!r}")
    condition, task, phase, *detail = parts
    return Path("measurements", condition, task, phase, *detail).with_suffix(".json.gz").as_posix()


def _archive_member_name(path: Path, root: Path) -> str:
    """Return a compact, stable public path independent of runtime work files."""

    relative = path.expanduser().resolve().relative_to(root.expanduser().resolve())
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == ".work" and parts[1] == MEASUREMENT_SHARD_DIRECTORY:
        return _measurement_archive_member_name(path)

    public_root_mapping = {
        "training_metrics.csv": "training/training_metrics.csv",
        "core_measurements.json": "measurements/core_measurements.json",
        "parameter_ownership.json": "metadata/parameter_ownership.json",
        "sample_manifest.json": "metadata/sample_manifest.json",
        "measurement_module_manifest.json": "metadata/measurement_module_manifest.json",
        "run_identity.json": "provenance/run_identity.json",
        "effective_run_config.json": "provenance/effective_run_config.json",
    }
    return public_root_mapping.get(relative.as_posix(), relative.as_posix())

def _verify_zip_against_manifest(
    zip_path: Path,
    *,
    zip_manifest: Mapping[str, Any],
) -> None:
    """Verify ZIP structure and CRC once before atomic publish."""

    expected_names = {
        str(record["path"]) for record in zip_manifest.get("files", [])
    } | {ZIP_MANIFEST_FILE}
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"corrupt_member:{bad_member}")
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("duplicate_archive_member")
            if set(names) != expected_names:
                raise RuntimeError(
                    f"archive_member_mismatch:expected={sorted(expected_names)} actual={sorted(names)}"
                )
            embedded = json.loads(archive.read(ZIP_MANIFEST_FILE))
            if embedded != dict(zip_manifest):
                raise RuntimeError("embedded_zip_manifest_mismatch")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise RuntimeError(f"DiR ZIP verification failed: {error}") from error

def _zip_files(
    zip_path: Path,
    root: Path,
    files: Sequence[Path],
    *,
    artifact_run_id: str,
    artifact_kind: str,
    run_marker_path: Path,
) -> dict[str, Any]:
    """Create one complete, nonempty, hash-manifested ZIP atomically."""

    root = root.expanduser().resolve()
    zip_path = zip_path.expanduser().resolve()
    run_marker_path = run_marker_path.expanduser().resolve()
    unique_files = [Path(path).expanduser().resolve() for path in files]
    if len(unique_files) != len(set(unique_files)):
        raise RuntimeError(f"DiR {artifact_kind} artifact list contains duplicates")
    failures: list[str] = []
    for path in unique_files:
        try:
            path.relative_to(root)
        except ValueError:
            failures.append(f"outside_root:{path}")
            continue
        if not path.is_file():
            failures.append(f"missing:{path}")
        elif path.stat().st_size <= 0:
            failures.append(f"empty:{path}")
    if not run_marker_path.is_file():
        failures.append(f"missing_run_marker:{run_marker_path}")
    else:
        marker = json.loads(run_marker_path.read_text(encoding="utf-8"))
        if str(marker.get("artifact_run_id", "")) != str(artifact_run_id):
            failures.append("run_marker_id_mismatch")
        if str(marker.get("schema_version", "")) != ARTIFACT_IDENTITY_SCHEMA_VERSION:
            failures.append("run_marker_schema_mismatch")
    if failures:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"DiR {artifact_kind} artifact completeness check failed: {failures}"
        )

    archive_members = [(path, _archive_member_name(path, root)) for path in unique_files]
    archive_names = [name for _, name in archive_members]
    if len(archive_names) != len(set(archive_names)):
        raise RuntimeError(f"DiR {artifact_kind} archive member names contain duplicates")
    file_records = [
        {
            "path": archive_name,
            "size_bytes": int(path.stat().st_size),
        }
        for path, archive_name in archive_members
    ]
    zip_manifest = {
        "schema_version": "dir_zip_manifest_v3",
        "artifact_run_id": str(artifact_run_id),
        "artifact_kind": str(artifact_kind),
        "created_at": _utc_now_iso(),
        "file_count": len(file_records),
        "files": file_records,
        "verification_contract": "atomic_publish_after_exact_member_set_and_single_CRC_test",
    }
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_name(zip_path.name + ".temporary")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, archive_name in archive_members:
                archive.write(path, arcname=archive_name)
            archive.writestr(
                ZIP_MANIFEST_FILE,
                json.dumps(zip_manifest, indent=2, sort_keys=True) + "\n",
            )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"DiR {artifact_kind} temporary ZIP was not created")
        _verify_zip_against_manifest(temporary, zip_manifest=zip_manifest)
        temporary.replace(zip_path)
    except Exception:
        _best_effort_unlink(temporary)
        _best_effort_unlink(zip_path)
        raise
    return zip_manifest

def _finalize_report_archive_fail_soft(
    *,
    zip_path: Path,
    root: Path,
    files: Sequence[Path],
    artifact_run_id: str,
    artifact_kind: str,
    run_marker_path: Path,
    warning_stage: str,
    warning_message: str,
    warning_recorder: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Finalize one post-training report archive without aborting later artifacts."""

    try:
        zip_manifest = _zip_files(
            zip_path,
            root,
            files,
            artifact_run_id=artifact_run_id,
            artifact_kind=artifact_kind,
            run_marker_path=run_marker_path,
        )
        return zip_manifest, "completed"
    except Exception as error:
        _best_effort_unlink(zip_path)
        warning_recorder(
            stage=warning_stage,
            message=warning_message,
            error=error,
        )
        return None, "warning_failed"

def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid DiR model checkpoint: {path}")
    if str(payload.get("schema_version", "")) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Stale DiR checkpoint schema: {path.name}")
    return dict(payload)

def _checkpoint_provenance_payload(
    *,
    checkpoint_specs: Mapping[str, Mapping[str, Any]],
    training_run_id: str | None,
    provenance_origin: str,
    training_contract_sha256: str,
    training_csv_sha256: str,
    support_commit_output_parity: Mapping[str, Any],
    support_commit_output_parity_sha256: str,
) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    for key, spec in checkpoint_specs.items():
        path = Path(spec["path"]).expanduser().resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"DiR checkpoint provenance missing file: {path}")
        checkpoints[str(key)] = {
            "filename": path.name,
            "size_bytes": int(path.stat().st_size),
            "identity": {name: value for name, value in spec.items() if name != "path"},
        }
    return {
        "schema_version": "dir_checkpoint_provenance_v3",
        "training_run_id": (str(training_run_id) if training_run_id else None),
        "provenance_origin": str(provenance_origin),
        "created_at": _utc_now_iso(),
        "training_contract_sha256": str(training_contract_sha256),
        "training_csv_sha256": str(training_csv_sha256),
        "support_commit_output_parity": dict(support_commit_output_parity),
        "support_commit_output_parity_sha256": str(
            support_commit_output_parity_sha256
        ),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
    }

def _write_checkpoint_provenance(path: Path, payload: Mapping[str, Any]) -> str:
    _write_json(path, dict(payload))
    return _sha256_file(path)

def _load_or_rebuild_checkpoint_provenance(
    path: Path,
    *,
    checkpoint_specs: Mapping[str, Mapping[str, Any]],
    current_artifact_run_id: str,
    training_contract_sha256: str,
    training_csv_sha256: str,
    support_commit_output_parity: Mapping[str, Any],
    support_commit_output_parity_sha256: str,
) -> tuple[dict[str, Any], str, bool]:
    """Load a sidecar or rebuild it without inventing a run ID."""

    rebuilt = False
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = str(payload.get("schema_version", ""))
        if schema in {"dir_checkpoint_provenance_v1", "dir_checkpoint_provenance_v2"}:
            if schema == "dir_checkpoint_provenance_v1":
                training_run_id = str(payload.get("artifact_run_id", "")) or None
                origin = "migrated_v1_without_checkpoint_byte_hashes"
            else:
                training_run_id = payload.get("training_run_id")
                origin = "migrated_v2_without_checkpoint_byte_hashes"
            payload = _checkpoint_provenance_payload(
                checkpoint_specs=checkpoint_specs,
                training_run_id=training_run_id,
                provenance_origin=origin,
                training_contract_sha256=training_contract_sha256,
                training_csv_sha256=training_csv_sha256,
                support_commit_output_parity=support_commit_output_parity,
                support_commit_output_parity_sha256=support_commit_output_parity_sha256,
            )
            _write_checkpoint_provenance(path, payload)
            rebuilt = True
    else:
        payload = _checkpoint_provenance_payload(
            checkpoint_specs=checkpoint_specs,
            training_run_id=None,
            provenance_origin="unknown_rebuilt",
            training_contract_sha256=training_contract_sha256,
            training_csv_sha256=training_csv_sha256,
            support_commit_output_parity=support_commit_output_parity,
            support_commit_output_parity_sha256=support_commit_output_parity_sha256,
        )
        _write_checkpoint_provenance(path, payload)
        rebuilt = True
    expected = _checkpoint_provenance_payload(
        checkpoint_specs=checkpoint_specs,
        training_run_id=payload.get("training_run_id"),
        provenance_origin=str(payload.get("provenance_origin", "")),
        training_contract_sha256=training_contract_sha256,
        training_csv_sha256=training_csv_sha256,
        support_commit_output_parity=support_commit_output_parity,
        support_commit_output_parity_sha256=support_commit_output_parity_sha256,
    )
    for key in (
        "schema_version",
        "training_run_id",
        "provenance_origin",
        "training_contract_sha256",
        "training_csv_sha256",
        "support_commit_output_parity_sha256",
        "checkpoint_count",
        "checkpoints",
    ):
        if payload.get(key) != expected.get(key):
            raise ValueError(f"DiR checkpoint provenance mismatch: field={key}")
    if _canonical_json_sha256(payload.get("support_commit_output_parity")) != str(
        support_commit_output_parity_sha256
    ):
        raise ValueError("DiR checkpoint provenance support parity mismatch")
    if payload.get("training_run_id") == current_artifact_run_id and str(
        payload.get("provenance_origin", "")
    ) == "unknown_rebuilt":
        raise ValueError("DiR rebuilt provenance must not claim the current packaging run")
    return dict(payload), _sha256_file(path), rebuilt

def _load_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    expected_training_contract_sha256: str | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load_checkpoint_payload(path)
    metadata = dict(payload.get("metadata", {}) or {})
    if expected_training_contract_sha256 is not None and str(
        metadata.get("training_contract_sha256", "")
    ) != str(expected_training_contract_sha256):
        raise ValueError(f"DiR checkpoint training contract hash mismatch: {path.name}")
    if expected_identity is not None:
        _validate_checkpoint_identity(metadata, expected_identity, path=path)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return metadata

def _restore_models_from_checkpoints(
    models: Mapping[str, nn.Module],
    model_checkpoint_paths: Mapping[int, Path],
) -> dict[str, str]:
    restored: dict[str, str] = {}
    for label, model in models.items():
        checkpoint_path = model_checkpoint_paths.get(id(model))
        if checkpoint_path is None or not checkpoint_path.is_file():
            restored[label] = "checkpoint_unavailable"
            continue
        _load_model_checkpoint(checkpoint_path, model)
        model.eval()
        restored[label] = str(checkpoint_path)
    return restored

def _write_module_manifest(
    path: Path,
    *,
    run_fingerprint: str,
    module_status: Mapping[str, Any],
) -> None:
    _write_json(
        path,
        {
            "schema_version": "dir_measurement_module_manifest_v2",
            "run_fingerprint": str(run_fingerprint),
            "module_count": len(module_status),
            "modules": dict(module_status),
        },
    )
