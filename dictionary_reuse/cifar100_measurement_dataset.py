"""CIFAR-100 dataset utilities for DiR training and measurement.

Core post-training correspondence measurements use the official CIFAR-100 test
split (runtime ``eval``). Linear-probe fitting and validation use disjoint slices of
the training split, while linear-probe testing uses the same official test split.

CIFAR-100 downloading is handled here, not by ``torchvision``'s implicit
``download=True`` path, so slow Toronto downloads can fail over to verified
mirrors while preserving the canonical CIFAR-100 MD5 checks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tarfile
import time
from typing import Any, Deque, Mapping, Sequence
from urllib import request
from urllib.error import HTTPError, URLError


_CIFAR100_ARCHIVE_FILENAME = "cifar-100-python.tar.gz"
_CIFAR100_ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
_CIFAR100_ARCHIVE_BYTES = 169001437
_CIFAR100_EXTRACTED_DIRECTORY_NAME = "cifar-100-python"
_CIFAR100_REQUIRED_FILES = (
    ("train", "16019d7e3df5f24257cddd939b257f8d"),
    ("test", "f0ef6b0ae62326f3e7ffdfab6717acfc"),
    ("meta", "7973b15100ade9c7d40fb424638fde48"),
)

_CIFAR100_DOWNLOAD_MIRRORS: dict[str, dict[str, object]] = {
    "toronto_official": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    },
    "huggingface_lfs_mirror": {
        "url": "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/201a32345d2c6b970e1a36c582930c83e09c96d2/cifar-100-python.tar.gz",
    },
}

_DEFAULT_CIFAR100_DOWNLOAD_OPTIONS: dict[str, object] = {
    "enabled": True,
    "low_speed_bytes_per_second": 100 * 1024,
    "low_speed_grace_seconds": 60,
    "low_speed_window_seconds": 60,
    "min_remaining_bytes_for_failover": 8 * 1024 * 1024,
    "connect_timeout_seconds": 15,
    "read_timeout_seconds": 30,
    "mirror_order": [
        "toronto_official",
        "huggingface_lfs_mirror",
    ],
    "custom_mirrors": [],
}


class CIFAR100DownloadError(RuntimeError):
    """Raised when the verified CIFAR-100 archive cannot be obtained."""


class LowSpeedDownloadError(CIFAR100DownloadError):
    """Raised to trigger mirror failover after sustained low throughput."""


@dataclass(frozen=True)
class CIFAR100DownloadMirror:
    name: str
    url: str


def _merged_cifar100_download_options(download_options: Mapping[str, Any] | None = None) -> dict[str, object]:
    merged = dict(_DEFAULT_CIFAR100_DOWNLOAD_OPTIONS)
    if download_options:
        unknown_keys = sorted(str(key) for key in download_options if key not in merged)
        if unknown_keys:
            raise ValueError(f"Unknown cifar100_download option(s): {', '.join(unknown_keys)}")
        for key, value in download_options.items():
            merged[key] = value
    return merged


def _cifar100_mirror_sequence(
    download_options: Mapping[str, Any] | None = None,
) -> list[CIFAR100DownloadMirror]:
    options = _merged_cifar100_download_options(download_options)
    configured_order = options.get("mirror_order", [])
    if not isinstance(configured_order, Iterable) or isinstance(configured_order, (str, bytes)):
        raise ValueError("cifar100_download.mirror_order must be a list of known mirror names")

    mirror_definitions = {name: dict(payload) for name, payload in _CIFAR100_DOWNLOAD_MIRRORS.items()}
    custom_mirror_names: list[str] = []
    custom_mirrors = options.get("custom_mirrors", [])
    if custom_mirrors is not None and not isinstance(custom_mirrors, Iterable):
        raise ValueError("cifar100_download.custom_mirrors must be a list when provided")
    for index, custom_mirror in enumerate(custom_mirrors or []):
        if not isinstance(custom_mirror, Mapping):
            raise ValueError("Each CIFAR-100 custom mirror must be an object with a url")
        url = str(custom_mirror.get("url", "")).strip()
        if not url:
            raise ValueError("Each CIFAR-100 custom mirror must define a non-empty url")
        name = str(custom_mirror.get("name", f"custom_mirror_{index + 1}")).strip() or f"custom_mirror_{index + 1}"
        mirror_definitions[name] = {"url": url}
        custom_mirror_names.append(name)

    mirrors: list[CIFAR100DownloadMirror] = []
    seen: set[str] = set()

    # User-supplied mirrors are tried before public mirrors so a local campus,
    # object-store, or Drive-hosted copy can avoid a slow Toronto attempt.
    for mirror_key in custom_mirror_names:
        if mirror_key in seen:
            continue
        seen.add(mirror_key)
        mirror_payload = mirror_definitions[mirror_key]
        mirrors.append(CIFAR100DownloadMirror(name=mirror_key, url=str(mirror_payload["url"])))

    for mirror_name in configured_order:
        mirror_key = str(mirror_name)
        if mirror_key in seen:
            continue
        seen.add(mirror_key)
        mirror_payload = mirror_definitions.get(mirror_key)
        if mirror_payload is None:
            raise ValueError(f"Unknown CIFAR-100 download mirror: {mirror_key}")
        mirrors.append(
            CIFAR100DownloadMirror(
                name=mirror_key,
                url=str(mirror_payload["url"]),
            )
        )
    if not mirrors:
        raise ValueError("No CIFAR-100 download mirrors are configured")
    return mirrors


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cifar100_integrity_ok(dataset_root_directory: Path) -> bool:
    extracted_directory = dataset_root_directory / _CIFAR100_EXTRACTED_DIRECTORY_NAME
    return all((extracted_directory / filename).is_file() and _file_md5(extracted_directory / filename) == md5 for filename, md5 in _CIFAR100_REQUIRED_FILES)


def _safe_extract_tar_gz(archive_path: Path, extract_root: Path) -> None:
    resolved_root = extract_root.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_destination = (extract_root / member.name).resolve()
            if not str(member_destination).startswith(str(resolved_root) + "/") and member_destination != resolved_root:
                raise CIFAR100DownloadError(f"Refusing unsafe archive member path: {member.name}")
        archive.extractall(extract_root)


def _extract_cifar100_archive(archive_path: Path, dataset_root_directory: Path) -> None:
    _safe_extract_tar_gz(archive_path, dataset_root_directory)


def _should_failover_for_low_speed(
    *,
    elapsed_seconds: float,
    recent_speed_bytes_per_second: float,
    total_bytes: int | None,
    downloaded_bytes: int,
    download_options: Mapping[str, Any] | None = None,
) -> bool:
    options = _merged_cifar100_download_options(download_options)
    low_speed_threshold = int(options["low_speed_bytes_per_second"])
    grace_seconds = float(options["low_speed_grace_seconds"])
    min_remaining_bytes = int(options["min_remaining_bytes_for_failover"])

    if elapsed_seconds < grace_seconds:
        return False
    if recent_speed_bytes_per_second >= low_speed_threshold:
        return False
    expected_total_bytes = _CIFAR100_ARCHIVE_BYTES if total_bytes is None else int(total_bytes)
    remaining_bytes = max(0, expected_total_bytes - int(downloaded_bytes))
    return remaining_bytes > min_remaining_bytes


def _stream_download_to_file(
    url: str,
    destination_path: Path,
    download_options: Mapping[str, Any] | None = None,
) -> None:
    options = _merged_cifar100_download_options(download_options)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination_path.with_name(destination_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    connect_timeout_seconds = float(options["connect_timeout_seconds"])
    read_timeout_seconds = float(options["read_timeout_seconds"])
    window_seconds = float(options["low_speed_window_seconds"])
    request_object = request.Request(url, headers={"User-Agent": "DiR-CIFAR100-downloader/1.0"})

    start_time = time.monotonic()
    downloaded_bytes = 0
    recent_points: Deque[tuple[float, int]] = deque([(start_time, 0)])

    try:
        # S310 is safe here because the download URLs are fixed and trusted.
        with request.urlopen(request_object, timeout=connect_timeout_seconds) as response:  # noqa: S310
            try:
                # Access depends on urllib internals and is unavailable on some runtimes.
                response.fp.raw._sock.settimeout(read_timeout_seconds)  # type: ignore[attr-defined]  # pragma: no cover
            except Exception:
                pass
            content_length_header = response.headers.get("Content-Length")
            total_bytes = int(content_length_header) if content_length_header and content_length_header.isdigit() else None
            with partial_path.open("wb") as output_file:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    now = time.monotonic()
                    recent_points.append((now, downloaded_bytes))
                    while len(recent_points) > 2 and now - recent_points[1][0] > window_seconds:
                        recent_points.popleft()
                    oldest_time, oldest_bytes = recent_points[0]
                    window_elapsed = max(now - oldest_time, 1e-6)
                    recent_speed = (downloaded_bytes - oldest_bytes) / window_elapsed
                    if _should_failover_for_low_speed(
                        elapsed_seconds=now - start_time,
                        recent_speed_bytes_per_second=recent_speed,
                        total_bytes=total_bytes,
                        downloaded_bytes=downloaded_bytes,
                        download_options=options,
                    ):
                        raise LowSpeedDownloadError(
                            f"download below {int(options['low_speed_bytes_per_second'])} B/s "
                            f"for CIFAR-100 mirror url={url} recent_speed={recent_speed:.1f} B/s"
                        )
    except (HTTPError, TimeoutError, URLError, OSError, LowSpeedDownloadError) as exc:
        if partial_path.exists():
            partial_path.unlink()
        if isinstance(exc, LowSpeedDownloadError):
            raise
        raise CIFAR100DownloadError(f"download failed for {url}: {type(exc).__name__}: {exc}") from exc

    partial_path.replace(destination_path)


def _download_cifar100_with_mirrors(
    dataset_root_directory: Path,
    download_options: Mapping[str, Any] | None = None,
) -> None:
    options = _merged_cifar100_download_options(download_options)
    dataset_root_directory.mkdir(parents=True, exist_ok=True)
    if _cifar100_integrity_ok(dataset_root_directory):
        return

    archive_path = dataset_root_directory / _CIFAR100_ARCHIVE_FILENAME
    if archive_path.is_file() and _file_md5(archive_path) == _CIFAR100_ARCHIVE_MD5:
        _extract_cifar100_archive(archive_path, dataset_root_directory)
        if _cifar100_integrity_ok(dataset_root_directory):
            return
    elif archive_path.exists():
        archive_path.unlink()

    errors: list[str] = []
    for mirror in _cifar100_mirror_sequence(options):
        try:
            _stream_download_to_file(mirror.url, archive_path, options)
            actual_md5 = _file_md5(archive_path)
            if actual_md5 != _CIFAR100_ARCHIVE_MD5:
                archive_path.unlink(missing_ok=True)
                errors.append(f"{mirror.name}: archive md5 mismatch {actual_md5} != {_CIFAR100_ARCHIVE_MD5}")
                continue
            _extract_cifar100_archive(archive_path, dataset_root_directory)
            if _cifar100_integrity_ok(dataset_root_directory):
                return
            errors.append(f"{mirror.name}: archive extracted but required CIFAR-100 files failed integrity checks")
        except Exception as exc:  # noqa: BLE001 - collect all mirror failures and report them together.
            archive_path.unlink(missing_ok=True)
            errors.append(f"{mirror.name}: {type(exc).__name__}: {exc}")

    joined_errors = "\n".join(errors)
    raise CIFAR100DownloadError("Failed to download verified CIFAR-100 archive from all configured mirrors.\n" + joined_errors)


def _build_cifar100_dataset(
    dataset_root_directory: Path,
    normalization_mean: Sequence[float],
    normalization_standard_deviation: Sequence[float],
    train: bool,
    cifar100_download_options: Mapping[str, Any] | None = None,
):
    # Keep torchvision import lazy so repository policy tests can import this
    # module even in lightweight environments that do not have a matching
    # torchvision binary installed. Runtime dataset construction still requires
    # torchvision.
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_standard_deviation),
        ]
    )
    dataset_root_directory.mkdir(parents=True, exist_ok=True)
    download_options = _merged_cifar100_download_options(cifar100_download_options)
    custom_downloader_enabled = bool(download_options["enabled"])
    if custom_downloader_enabled:
        _download_cifar100_with_mirrors(dataset_root_directory, download_options)
    return datasets.CIFAR100(
        root=str(dataset_root_directory),
        train=bool(train),
        download=not custom_downloader_enabled,
        transform=transform,
    )


def build_cifar100_training_dataset(
    dataset_root_directory: Path,
    normalization_mean: Sequence[float],
    normalization_standard_deviation: Sequence[float],
    cifar100_download_options: Mapping[str, Any] | None = None,
):
    return _build_cifar100_dataset(
        dataset_root_directory=dataset_root_directory,
        normalization_mean=normalization_mean,
        normalization_standard_deviation=normalization_standard_deviation,
        train=True,
        cifar100_download_options=cifar100_download_options,
    )


def build_cifar100_evaluation_dataset(
    dataset_root_directory: Path,
    normalization_mean: Sequence[float],
    normalization_standard_deviation: Sequence[float],
    cifar100_download_options: Mapping[str, Any] | None = None,
):
    """Build the official CIFAR-100 test split for evaluation and measurement.

    Core correspondence measurements and linear-probe testing use this split.
    Linear-probe fitting/validation explicitly request the training split instead.
    """

    return _build_cifar100_dataset(
        dataset_root_directory=dataset_root_directory,
        normalization_mean=normalization_mean,
        normalization_standard_deviation=normalization_standard_deviation,
        train=False,
        cifar100_download_options=cifar100_download_options,
    )
