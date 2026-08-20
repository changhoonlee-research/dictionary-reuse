"""Small filesystem/configuration helpers used by the DiR execution path."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def read_json_file(input_file_path: Path) -> Any:
    with input_file_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json_file(output_file_path: Path, payload: Any) -> None:
    """Atomically write a deterministic, human-readable JSON file."""

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_file_path.with_name(output_file_path.name + ".temporary")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(output_file_path)


def deep_merge_json_objects(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge objects; lists and scalar values replace the base."""

    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_json_objects(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_base_config_path(config_path: Path, configured_value: str) -> Path:
    candidate = Path(str(configured_value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    local = (config_path.parent / candidate).resolve()
    if local.is_file():
        return local
    return (config_path.parent.parent / candidate).resolve()


def read_resolved_config(input_file_path: Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a complete config, resolving an optional immutable base plus overrides."""

    path = input_file_path.expanduser().resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Circular config inheritance: {chain}")
    payload = read_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")

    base_value = payload.get("base_config")
    if not base_value:
        if "overrides" in payload:
            raise ValueError(f"Config has overrides without base_config: {path}")
        return deepcopy(payload)

    allowed_wrapper_keys = {
        "config_reference_schema",
        "config_version",
        "base_config",
        "base_version",
        "overrides",
    }
    unexpected = sorted(set(payload) - allowed_wrapper_keys)
    if unexpected:
        raise ValueError(f"Derived config contains executable values outside overrides: {path}: {unexpected}")
    if payload.get("config_reference_schema") != "dir_config_override_v1":
        raise ValueError(f"Unsupported config_reference_schema: {path}")
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Config overrides must be an object: {path}")
    if "config_version" in overrides:
        raise ValueError(f"Derived config must declare config_version only at wrapper level: {path}")
    declared_version = str(payload.get("config_version", "")).strip()
    declared_base_version = str(payload.get("base_version", "")).strip()
    if not declared_version or not declared_base_version:
        raise ValueError(f"Derived config requires non-empty config_version and base_version: {path}")

    base_path = _resolve_base_config_path(path, str(base_value))
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base config {base_path} referenced by {path}")
    base = read_resolved_config(base_path, _stack=(*_stack, path))
    actual_base_version = str(base.get("config_version", "")).strip()
    if actual_base_version != declared_base_version:
        raise ValueError(
            f"base_version mismatch for {path}: declared {declared_base_version!r}, "
            f"resolved {actual_base_version!r} from {base_path}"
        )
    resolved = deep_merge_json_objects(base, overrides)
    resolved["config_version"] = declared_version
    return resolved
