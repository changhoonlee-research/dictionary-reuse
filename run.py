"""Run the DiR paper experiment for two training seeds and aggregate the results."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPOSITORY_ROOT / "config.json"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dictionary_reuse.artifacts import (  # noqa: E402
    read_resolved_config as _load_json,
    write_json_file as _write_json,
)
from dictionary_reuse.replicates import (  # noqa: E402
    REPLICATE_LABELS,
    RUN_B_TRAINING_SEED_OFFSET,
    apply_training_seed_offset,
    replicate_reports_complete,
    replicate_runtime_paths,
    resolve_batch_name,
    write_aggregate_reports,
)

PUBLIC_CONSOLE_PREFIX = "[DiR v1.2.4]"
_BASE_REPORT_ALLOWLIST = {"effective_run_config.json", "experiment_error.json"}


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{whole_seconds:02d}s"
    if minutes:
        return f"{minutes}m{whole_seconds:02d}s"
    return f"{seconds:.1f}s"


def _resolve_configured_path(path_value: str | None) -> Path:
    if not path_value:
        return Path("")
    path = Path(str(path_value)).expanduser()
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _resolve_config_path() -> Path:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    return CONFIG_PATH.resolve()


def _build_runtime_config(
    base_config: dict[str, Any],
    *,
    batch_name: str,
    replicate_label: str,
    seed_offset: int,
) -> dict[str, Any]:
    config = apply_training_seed_offset(base_config, seed_offset)
    results_dir_value = str(config.get("results_dir", "")).strip()
    if not results_dir_value:
        raise ValueError("config.json requires non-empty results_dir")
    results_dir = _resolve_configured_path(results_dir_value)
    checkpoints_dir_value = str(config.get("checkpoints_dir", "")).strip()
    if not checkpoints_dir_value:
        raise ValueError("config.json requires non-empty checkpoints_dir")
    checkpoints_dir = _resolve_configured_path(checkpoints_dir_value)

    plan = dict(config.get("functional_correspondence", {}) or {})
    role = dict(config.get("dictionary_reuse", {}) or {})
    runtime = dict(role.get("runtime", {}) or {})
    plan["mode"] = "measurement"
    plan["require_cuda"] = True
    runtime["default_preset"] = "measurement"
    runtime["device"] = "cuda"
    runtime["fail_fast_if_cuda_unavailable"] = True
    role["runtime"] = runtime
    config["functional_correspondence"] = plan
    config["dictionary_reuse"] = role
    config["paths"] = replicate_runtime_paths(
        results_dir,
        checkpoints_dir,
        batch_name=batch_name,
        replicate_label=replicate_label,
    )
    return config


def _validate_config_inputs(config: dict[str, Any]) -> dict[str, Any]:
    from dictionary_reuse.experiment import validate_config

    return validate_config(config)


def _completion_status(output_dir: Path) -> str:
    """Return the authoritative completion-receipt status before output cleanup."""

    receipt_path = output_dir.expanduser().resolve() / "completion_receipt.json"
    if not receipt_path.is_file():
        return "completion_status_unavailable"
    try:
        receipt = _load_json(receipt_path)
    except Exception:
        return "completion_status_unavailable"
    status = str(receipt.get("status", "")).strip()
    if status in {"completed", "completed_with_warnings"}:
        return status
    return "completion_status_unavailable"


def _successful_output_allowlist(config: dict[str, Any]) -> set[str]:
    allowed = set(_BASE_REPORT_ALLOWLIST)
    configured = config.get("active_output_files", [])
    if configured is None:
        configured = []
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise ValueError("active_output_files must be a list of relative file names.")
    for item in configured:
        is_prefix = item.endswith("/")
        relative = Path(item.rstrip("/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"active_output_files contains an unsafe path: {item}")
        normalized = relative.as_posix()
        allowed.add(normalized + "/" if is_prefix else normalized)
    return allowed


def _current_run_reached_post_training(output_dir: Path) -> bool:
    output_dir = output_dir.expanduser().resolve()
    identity_path = output_dir / "run_identity.json"
    progress_path = output_dir / "progress.json"
    if not identity_path.is_file() or not progress_path.is_file():
        return False
    try:
        identity = _load_json(identity_path)
        progress = _load_json(progress_path)
    except Exception:
        return False
    identity_run_id = str(identity.get("artifact_run_id", "")).strip()
    progress_run_id = str(progress.get("artifact_run_id", "")).strip()
    if not identity_run_id or progress_run_id != identity_run_id:
        return False
    return str(progress.get("stage", "")) in {
        "post_training_measurement",
        "statistics",
        "artifact_packaging",
        "complete",
    }


def _clear_stale_active_outputs(output_dir: Path, allowed_relative_paths: set[str]) -> None:
    output_dir = output_dir.expanduser().resolve()
    stale_names = set(allowed_relative_paths) | {"archive_profile_manifest.json"}
    for relative_name in sorted(stale_names):
        is_prefix = relative_name.endswith("/")
        relative = Path(relative_name.rstrip("/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe stale-output path: {relative_name}")
        candidate = output_dir / relative
        if is_prefix and candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.is_file() or candidate.is_symlink():
            candidate.unlink(missing_ok=True)


def _finalize_completed_run_layout(config: dict[str, Any]) -> str | None:
    """After clean completion keep only both report ZIPs in results; checkpoints stay separate."""

    try:
        paths = dict(config.get("paths", {}) or {})
        output_dir = Path(str(paths["output_dir"])).expanduser().resolve()
        checkpoint_dir = Path(str(paths["checkpoint_dir"])).expanduser().resolve()
        raw_zip = Path(str(paths["raw_report_zip"])).expanduser().resolve()
        summary_zip = Path(str(paths["summary_report_zip"])).expanduser().resolve()
        for child in (raw_zip, summary_zip):
            child.relative_to(output_dir)
        try:
            checkpoint_dir.relative_to(output_dir)
        except ValueError:
            pass
        else:
            raise ValueError("paths.checkpoint_dir must be outside paths.output_dir")

        root_receipt = output_dir / "completion_receipt.json"
        if not root_receipt.is_file():
            return None
        completion = _load_json(root_receipt)
        archive_statuses = dict(completion.get("archive_statuses", {}) or {})
        if str(completion.get("status", "")) not in {"completed", "completed_with_warnings"}:
            return None
        if archive_statuses != {"raw_report": "completed", "summary_report": "completed"}:
            return None
        if not raw_zip.is_file() or not summary_zip.is_file() or not checkpoint_dir.is_dir():
            return None

        checkpoint_receipt = checkpoint_dir / "completion_receipt.json"
        temporary_receipt = checkpoint_receipt.with_name(checkpoint_receipt.name + ".temporary")
        temporary_receipt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root_receipt, temporary_receipt)
        temporary_receipt.replace(checkpoint_receipt)

        preserved = {raw_zip.name, summary_zip.name}
        for child in list(output_dir.iterdir()):
            if child.name in preserved:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        return None
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate", choices=REPLICATE_LABELS, help=argparse.SUPPRESS)
    parser.add_argument("--batch-name", help=argparse.SUPPRESS)
    parser.add_argument("--seed-offset", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _run_single_replicate(*, replicate_label: str, batch_name: str, seed_offset: int) -> int:
    pipeline_started = time.perf_counter()
    base_config = _load_json(_resolve_config_path())
    config = _build_runtime_config(
        base_config,
        batch_name=batch_name,
        replicate_label=replicate_label,
        seed_offset=seed_offset,
    )
    validation = _validate_config_inputs(config)
    if not bool(validation.get("training_enabled", False)):
        raise RuntimeError("DiR release run must resolve to scientific training mode")

    output_dir = _resolve_configured_path(config.get("paths", {}).get("output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_active_outputs(output_dir, _successful_output_allowlist(config))
    effective_config_path = output_dir / "effective_run_config.json"
    _write_json(effective_config_path, config)

    role_runtime = dict(config.get("dictionary_reuse", {}).get("runtime", {}) or {})
    training_runtime = dict(config.get("functional_correspondence", {}).get("training", {}) or {})
    print(
        f"{PUBLIC_CONSOLE_PREFIX} Run {replicate_label} | "
        f"source_epochs={int(training_runtime['dir_source_a_epochs'])} | "
        f"target_epochs={int(training_runtime['dir_target_epochs'])} | "
        f"device={role_runtime.get('device', 'unknown')}",
        flush=True,
    )
    print("[VALIDATE] static config passed; runtime guards execute before training", flush=True)

    error_path = output_dir / "experiment_error.json"
    error_path.unlink(missing_ok=True)
    try:
        from dictionary_reuse.experiment import run_experiment

        run_experiment(effective_config_path)
        completion_status = _completion_status(output_dir)
        cleanup_warning = _finalize_completed_run_layout(config)
        if cleanup_warning:
            print(f"[WARNING] completed-run layout cleanup: {cleanup_warning}", flush=True)
    except KeyboardInterrupt:
        _write_json(error_path, {"status": "interrupted", "error_name": "KeyboardInterrupt"})
        return 130
    except Exception as error:
        existing_error: dict[str, Any] = {}
        if error_path.is_file():
            try:
                loaded_error = _load_json(error_path)
                if isinstance(loaded_error, dict):
                    existing_error.update(loaded_error)
            except Exception:
                pass
        post_training_warning = _current_run_reached_post_training(output_dir)
        existing_error.update(
            {
                "status": "completed_with_post_training_warning" if post_training_warning else "failed",
                "error_name": type(error).__name__,
                "error_message": str(error),
                **({"post_training_failure_is_nonfatal": True} if post_training_warning else {}),
            }
        )
        _write_json(error_path, existing_error)
        severity = "warning" if post_training_warning else "error"
        print(f"[{severity.upper()}] {type(error).__name__}: {error}", flush=True)
        return 0 if post_training_warning else 1

    print(
        f"[COMPLETE] Run {replicate_label} | {completion_status} | "
        f"total={_format_elapsed(time.perf_counter() - pipeline_started)}",
        flush=True,
    )
    return 0


def _run_two_replicates() -> int:
    pipeline_started = time.perf_counter()
    base_config = _load_json(_resolve_config_path())
    results_value = str(base_config.get("results_dir", "")).strip()
    checkpoints_value = str(base_config.get("checkpoints_dir", "")).strip()
    if not results_value or not checkpoints_value:
        raise ValueError("config.json requires non-empty results_dir and checkpoints_dir")
    results_root = _resolve_configured_path(results_value)
    checkpoints_root = _resolve_configured_path(checkpoints_value)
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    batch_name, resumed = resolve_batch_name(results_root, checkpoints_root, base_config)
    if resumed:
        print(f"[RESUME] reusing incomplete two-seed batch {batch_name}", flush=True)
    print(f"{PUBLIC_CONSOLE_PREFIX} two-seed paper run | batch={batch_name}", flush=True)

    batch_dir = results_root / batch_name
    checkpoint_batch_dir = checkpoints_root / batch_name
    for label, offset in (("A", 0), ("B", RUN_B_TRAINING_SEED_OFFSET)):
        if replicate_reports_complete(batch_dir, checkpoint_batch_dir, label):
            print(f"[RESUME] Run {label} already complete; reusing existing reports", flush=True)
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--replicate",
            label,
            "--batch-name",
            batch_name,
            "--seed-offset",
            str(offset),
        ]
        result = subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=False)
        if result.returncode != 0:
            print(f"[ERROR] Run {label} failed; aggregate report was not created", flush=True)
            return int(result.returncode)

    incomplete = [
        label
        for label in REPLICATE_LABELS
        if not replicate_reports_complete(batch_dir, checkpoint_batch_dir, label)
    ]
    if incomplete:
        print(
            f"[WARNING] aggregate report unavailable because completed report ZIPs are missing for: {incomplete}",
            flush=True,
        )
        return 0

    write_aggregate_reports(batch_dir, checkpoint_batch_dir)
    print(
        f"[COMPLETE] Run A + Run B + aggregate | "
        f"total={_format_elapsed(time.perf_counter() - pipeline_started)}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.replicate is not None:
        if not args.batch_name:
            raise ValueError("Internal replicate execution requires --batch-name")
        return _run_single_replicate(
            replicate_label=str(args.replicate),
            batch_name=str(args.batch_name),
            seed_offset=int(args.seed_offset),
        )
    return _run_two_replicates()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
