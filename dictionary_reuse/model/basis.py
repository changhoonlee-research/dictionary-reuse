"""Dictionary basis construction and deterministic basis utilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from typing import Any, Iterator, Sequence

import torch

_BASIS_BANK_CACHE: dict[tuple[int, int, str, int, int, tuple[tuple[str, int], ...]], tuple[torch.Tensor, torch.Tensor, str]] = {}

_GROUP_NAME_TO_ID = {"dct": 0, "wave": 1, "random": 2, "mask": 3}
_GROUP_ID_TO_NAME = {value: key for key, value in _GROUP_NAME_TO_ID.items()}

def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def _column_normalize(matrix: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    return matrix / matrix.norm(dim=0, keepdim=True).clamp_min(epsilon)

def _topk_indices_1d(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return largest-k indices without torch.topk, which can hang in minimal CPU runtimes."""

    flat = values.reshape(-1)
    limit = max(0, min(int(k), int(flat.numel())))
    if limit <= 0:
        return torch.empty(0, dtype=torch.long, device=flat.device)
    return torch.argsort(flat, descending=True)[:limit]

def _topk_abs_sum(values: torch.Tensor, k: int) -> torch.Tensor:
    indices = _topk_indices_1d(values.abs(), int(k))
    if int(indices.numel()) == 0:
        return values.new_tensor(0.0)
    return values.abs().reshape(-1)[indices].sum()

def _primitive_spec_cache_key(
    primitive_spec: Sequence[dict[str, int | str]] | None,
) -> tuple[tuple[str, int], ...]:
    if primitive_spec is None:
        return ()
    return tuple((str(item["group"]), int(item["count"])) for item in primitive_spec)

def _primitive_spec_uses_random_seed(primitive_spec: Sequence[dict[str, int | str]] | None) -> bool:
    return any(str(item["group"]).strip().lower() == "random" and int(item["count"]) > 0 for item in (primitive_spec or ()))

def _basis_bank_effective_seed(seed: int, primitive_spec: Sequence[dict[str, int | str]] | None) -> int:
    # DCT, Wave, and Mask banks are deterministic. Normalizing their seed lets
    # cache warm-up with seed 0 be reused by source row/col layers whose
    # construction seeds differ only for unrelated stochastic components.
    return int(seed) if _primitive_spec_uses_random_seed(primitive_spec) else 0

def _basis_bank_cache_key(
    length: int,
    count: int,
    *,
    basis_type: str,
    low_count: int,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]] | None = None,
) -> tuple[int, int, str, int, int, tuple[tuple[str, int], ...]]:
    effective_seed = _basis_bank_effective_seed(seed, primitive_spec)
    return (int(length), int(count), str(basis_type), int(low_count), effective_seed, _primitive_spec_cache_key(primitive_spec))

def _dct_frequency_indices(
    length: int,
    count: int,
    low_count: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the exact DCT frequency ids used by the basis policy."""

    low_count = max(0, min(int(low_count), int(count)))
    high_count = int(count) - low_count
    low_frequencies = torch.arange(low_count, dtype=torch.long, device=device)
    if high_count > 0:
        high_start = max(low_count, int(length) - high_count)
        high_frequencies = torch.arange(high_start, high_start + high_count, dtype=torch.long, device=device)
        high_frequencies = torch.remainder(high_frequencies, max(1, int(length)))
        return torch.cat([low_frequencies, high_frequencies], dim=0)
    return low_frequencies

def _dct_basis_from_frequencies(
    length: int,
    frequencies: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    frequencies = frequencies.detach().to(dtype=torch.float32, device=device).reshape(-1)
    if frequencies.numel() == 0:
        return torch.empty((length, 0), dtype=torch.float32, device=device)
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    basis = torch.cos(math.pi * (positions + 0.5) * frequencies.unsqueeze(0) / max(1, int(length)))
    zero_frequency = frequencies == 0
    if bool(zero_frequency.any().cpu()):
        basis[:, zero_frequency] = 1.0
    return _column_normalize(basis)

def _iter_haar_wave_candidates(length: int, *, device: torch.device | None = None) -> Iterator[torch.Tensor]:
    """Yield deterministic local step-wave candidates without duplicate columns."""

    if length <= 0:
        return
    positions = torch.arange(length, dtype=torch.float32, device=device)
    for half_width in [2 ** exponent for exponent in range(int(math.ceil(math.log2(max(2, length)))) + 1)]:
        width = int(2 * half_width)
        if width > length * 2:
            continue
        stride = max(1, int(half_width))
        for start in range(0, length, stride):
            midpoint = min(length, start + half_width)
            end = min(length, start + width)
            if midpoint <= start or end <= midpoint:
                continue
            vector = torch.zeros(length, dtype=torch.float32, device=device)
            vector[start:midpoint] = 1.0
            vector[midpoint:end] = -1.0
            if vector.norm() > 0:
                yield vector
        if half_width >= length:
            break
    # Add a few global sinusoidal sign patterns as deterministic fallbacks.
    max_frequency = min(length, 4 * int(math.sqrt(max(2, length))))
    for frequency in range(1, max_frequency + 1):
        vector = torch.sign(torch.sin(2.0 * math.pi * frequency * (positions + 0.5) / max(1, length)))
        if vector.norm() > 0:
            yield vector

def _wave_scale_bucket_counts(count: int) -> list[tuple[str, int]]:
    """Return the fixed coarse/mid/fine slot allocation for Wave atoms.

    A1d-W uses 64 Wave slots as 8 coarse, 16 mid, and 40 fine atoms. Smaller
    toy banks keep the same ordering idea while avoiding empty negative counts.
    """

    count = int(count)
    if count <= 0:
        return []
    if count == 1:
        return [("fine", 1)]
    if count == 2:
        return [("mid", 1), ("fine", 1)]
    if count < 8:
        coarse = 1
        mid = 1
    else:
        coarse = max(1, count // 8)
        mid = max(1, count // 4)
    fine = max(0, count - coarse - mid)
    return [("coarse", coarse), ("mid", mid), ("fine", fine)]

def _wave_half_widths_for_bucket(length: int, bucket: str) -> list[int]:
    """Deterministic Haar half-widths for a scale bucket."""

    length = int(length)
    if length <= 1:
        return [1]
    max_half = max(1, length // 2)
    if bucket == "coarse":
        raw = [length // 4, length // 5, length // 6, length // 8, length // 3]
    elif bucket == "mid":
        raw = [length // 16, length // 20, length // 24, length // 32, length // 12]
    else:
        raw = [1, 2, 3, 4, 5, 8, length // 96, length // 64, length // 48]
    widths: list[int] = []
    for value in raw:
        half_width = max(1, min(max_half, int(value)))
        if half_width not in widths:
            widths.append(half_width)
    return widths or [1]

def _haar_step_vector(
    length: int,
    start: int,
    half_width: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    vector = torch.zeros(int(length), dtype=torch.float64, device=device)
    start = max(0, min(int(start), max(0, int(length) - 1)))
    half_width = max(1, int(half_width))
    midpoint = min(int(length), start + half_width)
    end = min(int(length), start + 2 * half_width)
    if midpoint <= start or end <= midpoint:
        return vector
    vector[start:midpoint] = 1.0
    vector[midpoint:end] = -1.0
    return vector

def _iter_multiscale_wave_candidates(
    length: int,
    *,
    bucket: str,
    target_count: int,
    device: torch.device | None = None,
) -> Iterator[torch.Tensor]:
    """Yield ordered Haar candidates for one scale bucket.

    Slot order is coarse → mid → fine, and within each bucket left-to-right
    position. The later residualization step removes DCT-prefix overlap while
    keeping this deterministic scale/position ordering as much as possible.
    """

    length = int(length)
    target_count = max(1, int(target_count))
    for half_width in _wave_half_widths_for_bucket(length, bucket):
        width = 2 * int(half_width)
        if width > length:
            continue
        max_start = max(0, length - width)
        position_count = max(target_count * 4, 8)
        if max_start == 0:
            starts = [0]
        else:
            starts = sorted({int(round(index * max_start / max(1, position_count - 1))) for index in range(position_count)})
        for start in starts:
            vector = _haar_step_vector(length, start, half_width, device=device)
            if vector.norm() > 0:
                yield vector

def _project_out_columns(matrix: torch.Tensor, projection: torch.Tensor | None) -> torch.Tensor:
    residual = matrix
    if residual.ndim == 1:
        residual = residual.reshape(residual.shape[0], 1)
    if projection is not None and projection.numel() > 0:
        residual = residual - projection @ (projection.transpose(0, 1) @ residual)
    return residual

def _ordered_residual_gram_schmidt(
    candidates: Iterable[torch.Tensor],
    *,
    length: int,
    count: int,
    prefix: torch.Tensor | None,
    device: torch.device | None = None,
    min_norm: float = 1e-8,
) -> list[torch.Tensor]:
    """Select candidates in order after projecting out prefix and accepted atoms."""

    accepted: list[torch.Tensor] = []
    for candidate in candidates:
        residual = candidate.detach().to(dtype=torch.float64, device=device).reshape(int(length), 1)
        for _pass in range(2):
            residual = _project_out_columns(residual, prefix)
            if accepted:
                accepted_matrix = torch.stack(accepted, dim=1)
                residual = residual - accepted_matrix @ (accepted_matrix.transpose(0, 1) @ residual)
        norm = residual.norm()
        if float(norm.detach().cpu()) <= float(min_norm):
            continue
        accepted.append((residual[:, 0] / norm).detach())
        if len(accepted) >= int(count):
            break
    return accepted

def _projection_with_accepted(
    prefix: torch.Tensor | None,
    accepted: list[torch.Tensor],
) -> torch.Tensor | None:
    if accepted:
        accepted_matrix = torch.stack(accepted, dim=1)
        if prefix is not None and prefix.numel() > 0:
            return torch.cat([prefix, accepted_matrix], dim=1)
        return accepted_matrix
    return prefix

def _select_wave_basis(
    length: int,
    count: int,
    *,
    selected_prefix: list[torch.Tensor] | None = None,
    seed: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build DCT-residualized multiscale Haar-like Wave atoms.

    Active A1d-W frontWave banks use a deterministic 8/16/40 coarse/mid/fine
    slot structure for the 64 Wave atoms. Candidate columns are generated in
    fixed scale/position order, projected out of the DCT prefix, and selected by
    ordered Gram-Schmidt. This preserves transplant-stable slot semantics while
    keeping DCT↔Wave and Wave↔Wave coherence near numerical zero.
    """

    del seed  # active full-rank Wave banks are deterministic, not random
    count = int(count)
    length = int(length)
    if count <= 0:
        return torch.empty((length, 0), dtype=torch.float32, device=device)

    prefix = None
    if selected_prefix:
        prefix_matrix = torch.stack(
            [item.detach().to(dtype=torch.float64, device=device).reshape(length) for item in selected_prefix],
            dim=1,
        )
        if prefix_matrix.numel() > 0:
            prefix, _ = torch.linalg.qr(prefix_matrix, mode="reduced")

    residual_budget = max(0, length - (int(prefix.shape[1]) if prefix is not None else 0))
    accepted: list[torch.Tensor] = []
    for bucket, bucket_target in _wave_scale_bucket_counts(count):
        if bucket_target <= 0:
            continue
        bucket_candidates = _iter_multiscale_wave_candidates(
            length,
            bucket=bucket,
            target_count=bucket_target,
            device=device,
        )
        bucket_accepts = _ordered_residual_gram_schmidt(
            bucket_candidates,
            length=length,
            count=bucket_target,
            prefix=_projection_with_accepted(prefix, accepted),
            device=device,
        )
        accepted.extend(bucket_accepts)

    if len(accepted) < count:
        fallback_candidates = _iter_haar_wave_candidates(length, device=device)
        more_accepts = _ordered_residual_gram_schmidt(
            fallback_candidates,
            length=length,
            count=count - len(accepted),
            prefix=_projection_with_accepted(prefix, accepted),
            device=device,
        )
        accepted.extend(more_accepts)

    if len(accepted) < count:
        if residual_budget >= count:
            raise ValueError(
                f"Structured multiscale Wave basis produced only {len(accepted)} usable residual atoms "
                f"for length={length}, count={count}; random fallback is disabled for full-rank Wave banks"
            )
        # Tiny/overcomplete regression layers can request more Wave atoms than
        # there are residual dimensions, especially after a toy DCT prefix spans
        # the whole space. Keep those compact construction paths runnable with a
        # deterministic non-claim filler. Active A1d-W row/column dimensions have
        # residual_budget >= count and therefore never use this branch.
        filler = _random_orthogonal_basis(
            length,
            count - len(accepted),
            seed=15485863 * length + 32452843 * count,
            device=device,
        ).to(dtype=torch.float64, device=device)
        accepted.extend([filler[:, index].detach() for index in range(filler.shape[1])])

    if len(accepted) < count:
        raise ValueError(f"Unable to build {count} Wave atoms for length={length}; built {len(accepted)}")

    wave = torch.stack(accepted[:count], dim=1)
    if residual_budget >= count:
        accepted = _ordered_residual_gram_schmidt(
            [wave[:, index] for index in range(wave.shape[1])],
            length=length,
            count=count,
            prefix=prefix,
            device=device,
            min_norm=1e-12,
        )
        if len(accepted) < count:
            raise ValueError(f"Final Wave re-orthogonalization kept only {len(accepted)} of {count} atoms")
        wave = torch.stack(accepted[:count], dim=1)
    return _column_normalize(wave).to(dtype=torch.float32)

def _random_orthogonal_basis(
    length: int,
    count: int,
    *,
    seed: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(length, count, generator=generator, dtype=torch.float32)
    if count <= length:
        q_matrix, _r = torch.linalg.qr(matrix, mode="reduced")
        matrix = q_matrix[:, :count]
    else:
        matrix = _column_normalize(matrix)
    if device is not None:
        matrix = matrix.to(device)
    return _column_normalize(matrix)

def _random_residual_basis(
    length: int,
    count: int,
    *,
    selected_prefix: list[torch.Tensor] | None = None,
    seed: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build random control atoms in the same residual complement as Wave atoms."""

    count = int(count)
    if count <= 0:
        return torch.empty((length, 0), dtype=torch.float32, device=device)
    if not selected_prefix:
        return _random_orthogonal_basis(length, count, seed=seed, device=device)

    prefix = torch.stack([item.detach().to(dtype=torch.float32, device=device) for item in selected_prefix], dim=1)
    if prefix.numel() > 0:
        prefix, _ = torch.linalg.qr(prefix, mode="reduced")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 271828 * int(length) + 314159 * int(count))
    pieces: list[torch.Tensor] = []
    attempts = 0
    while sum(piece.shape[1] for piece in pieces) < count and attempts < 8:
        attempts += 1
        needed = count - sum(piece.shape[1] for piece in pieces)
        random_matrix = torch.randn(length, max(needed * 4, needed + 16), generator=generator, dtype=torch.float32)
        if device is not None:
            random_matrix = random_matrix.to(device)
        projection_parts = [prefix] if prefix.numel() else []
        projection_parts.extend(piece for piece in pieces if piece.numel() > 0)
        if projection_parts:
            projection = torch.cat(projection_parts, dim=1)
            projection, _ = torch.linalg.qr(projection, mode="reduced")
            random_matrix = random_matrix - projection @ (projection.transpose(0, 1) @ random_matrix)
        q_matrix, r_matrix = torch.linalg.qr(random_matrix, mode="reduced")
        if r_matrix.numel() == 0:
            continue
        keep = torch.diagonal(r_matrix, 0).abs() > 1e-5
        if keep.numel() == 0 or not bool(keep.any().cpu()):
            continue
        candidate = q_matrix[:, keep]
        take = min(needed, int(candidate.shape[1]))
        pieces.append(candidate[:, :take])
    if not pieces:
        return _random_orthogonal_basis(length, count, seed=seed, device=device)
    basis = torch.cat(pieces, dim=1)
    if basis.shape[1] < count:
        fallback = _random_orthogonal_basis(length, count - int(basis.shape[1]), seed=seed + 104729, device=device)
        projection_parts = [prefix, basis] if prefix.numel() else [basis]
        projection = torch.cat([item for item in projection_parts if item.numel() > 0], dim=1)
        projection, _ = torch.linalg.qr(projection, mode="reduced")
        fallback = fallback - projection @ (projection.transpose(0, 1) @ fallback)
        fallback = _column_normalize(fallback)
        basis = torch.cat([basis, fallback[:, : count - int(basis.shape[1])]], dim=1)
    return _column_normalize(basis[:, :count].to(dtype=torch.float32))

def _normalize_primitive_spec(
    primitive_spec: Sequence[dict[str, int | str]],
    *,
    count: int,
    source_name: str,
) -> list[dict[str, int | str]]:
    """Validate and normalize a group-tagged basis-bank primitive spec."""

    normalized_specs: list[dict[str, int | str]] = []
    total_count = 0
    for item in primitive_spec:
        if not isinstance(item, dict):
            raise ValueError(f"{source_name} primitive_spec entries must be objects")
        group = str(item.get("group", "")).strip().lower()
        if group not in _GROUP_NAME_TO_ID:
            raise ValueError(f"{source_name} has unsupported primitive group={group!r}")
        group_count = int(item.get("count", 0))
        if group_count < 0:
            raise ValueError(f"{source_name} primitive group={group!r} has negative count={group_count}")
        if group_count == 0:
            continue
        normalized_specs.append({"group": group, "count": group_count})
        total_count += group_count
    if total_count != int(count):
        raise ValueError(f"{source_name} primitive_spec total={total_count} does not match atom_count={int(count)}")
    if not normalized_specs and int(count) > 0:
        raise ValueError(f"{source_name} primitive_spec is empty for atom_count={int(count)}")
    return normalized_specs

def _basis_primitive_spec(basis_type: str, count: int, low_count: int) -> list[dict[str, int | str]]:
    """Resolve a compact compact basis identifier into primitive groups."""

    normalized = str(basis_type).strip().lower()
    count = int(count)
    if normalized in {"dct", "cosine"}:
        return [{"group": "dct", "count": count}]
    if normalized in {"haar", "wave", "wavelet", "haar_wavelet"}:
        return [{"group": "wave", "count": count}]
    if normalized in {"random_orthogonal", "orthogonal_random"}:
        return [{"group": "random", "count": count}]
    dct_only = re.fullmatch(r"dct_only_(\d+)", normalized)
    if dct_only:
        dct_count = int(dct_only.group(1))
        if dct_count != count:
            raise ValueError(f"basis_type={basis_type!r} requests {dct_count} atoms but atom_count={count}")
        return [{"group": "dct", "count": dct_count}]
    mixed = re.fullmatch(r"(?:mixed_)?dct(\d+)_wave(\d+)", normalized)
    if mixed:
        dct_count = int(mixed.group(1))
        wave_count = int(mixed.group(2))
        if dct_count + wave_count != count:
            raise ValueError(
                f"basis_type={basis_type!r} requests {dct_count + wave_count} atoms "
                f"but atom_count={count}"
            )
        return [{"group": "dct", "count": dct_count}, {"group": "wave", "count": wave_count}]
    mixed_random = re.fullmatch(r"(?:mixed_)?dct(\d+)_random(\d+)", normalized)
    if mixed_random:
        dct_count = int(mixed_random.group(1))
        random_count = int(mixed_random.group(2))
        if dct_count + random_count != count:
            raise ValueError(
                f"basis_type={basis_type!r} requests {dct_count + random_count} atoms "
                f"but atom_count={count}"
            )
        return [{"group": "dct", "count": dct_count}, {"group": "random", "count": random_count}]
    mixed_mask = re.fullmatch(r"(?:mixed_)?dct(\d+)_mask(?:ed)?(\d+)", normalized)
    if mixed_mask:
        dct_count = int(mixed_mask.group(1))
        masked_count = int(mixed_mask.group(2))
        if dct_count + masked_count != count:
            raise ValueError(
                f"basis_type={basis_type!r} requests {dct_count + masked_count} atoms "
                f"but atom_count={count}"
            )
        return [{"group": "dct", "count": dct_count}, {"group": "mask", "count": masked_count}]
    raise ValueError(
        f"Unknown basis_type={basis_type!r}; expected dct, dct_only_<K>, "
        "mixed_dct<N>_wave<M>, mixed_dct<N>_random<M>, mixed_dct<N>_mask<M>, haar_wavelet, or random_orthogonal, "
        "or define dictionary.basis_banks[<basis_type>].primitive_spec"
    )

def _basis_primitive_spec_from_config(
    dictionary_config: dict[str, Any],
    basis_type: str,
    count: int,
    low_count: int,
) -> list[dict[str, int | str]]:
    """Use dictionary.basis_banks as source of truth, falling back to compact basis names."""

    basis_banks = dictionary_config.get("basis_banks", {})
    if isinstance(basis_banks, dict) and str(basis_type) in basis_banks:
        bank_payload = basis_banks[str(basis_type)]
        if not isinstance(bank_payload, dict) or "primitive_spec" not in bank_payload:
            raise ValueError(f"basis bank {basis_type!r} must define primitive_spec")
        primitive_spec = bank_payload["primitive_spec"]
        if not isinstance(primitive_spec, list):
            raise ValueError(f"basis bank {basis_type!r} primitive_spec must be a list")
        return _normalize_primitive_spec(primitive_spec, count=int(count), source_name=f"basis bank {basis_type!r}")
    return _basis_primitive_spec(basis_type, count, low_count)

def _stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _basis_bank_hash(
    *,
    length: int,
    count: int,
    basis_type: str,
    low_count: int,
    seed: int,
    primitive_spec: Sequence[dict[str, int | str]],
) -> str:
    effective_seed = _basis_bank_effective_seed(seed, primitive_spec)
    return _stable_json_hash(
        {
            "basis_bank_version": "shared_mixed_residual_multiscale_wave_random_v4_seed_normalized",
            "length": int(length),
            "count": int(count),
            "basis_type": str(basis_type),
            "low_count": int(low_count),
            "seed": effective_seed,
            "seed_policy": "random_banks_only",
            "primitive_spec": list(primitive_spec),
        }
    )[:16]

def build_basis_bank(
    length: int,
    count: int,
    *,
    basis_type: str,
    low_count: int,
    seed: int,
    device: torch.device | None = None,
    primitive_spec: Sequence[dict[str, int | str]] | None = None,
    clone: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Build a fixed basis bank and atom group ids for Dictionary layers.

    Cached CPU banks are immutable. Public callers receive clones by default so
    accidental in-place edits cannot poison the process cache. Model
    construction can request ``clone=False`` because the returned bank is
    immediately copied into parameters or buffers, avoiding one large temporary
    clone per dictionary layer during frontWave source initialization.
    """

    specs = (
        _normalize_primitive_spec(primitive_spec, count=int(count), source_name=f"basis_type={basis_type!r}")
        if primitive_spec is not None
        else _basis_primitive_spec(basis_type, count, low_count)
    )
    key = _basis_bank_cache_key(length, count, basis_type=basis_type, low_count=low_count, seed=seed, primitive_spec=specs)
    cached = _BASIS_BANK_CACHE.get(key)
    if cached is None:
        columns: list[torch.Tensor] = []
        group_ids: list[int] = []
        selected_for_wave: list[torch.Tensor] = []
        used_dct_frequencies: set[int] = set()
        for spec in specs:
            group = str(spec["group"])
            group_count = int(spec["count"])
            if group_count <= 0:
                continue
            if group == "dct":
                dct_frequencies = _dct_frequency_indices(length, group_count, min(int(low_count), group_count), device=None)
                group_basis = _dct_basis_from_frequencies(length, dct_frequencies, device=None)
                used_dct_frequencies.update(int(item) for item in dct_frequencies.detach().cpu().tolist())
                selected_for_wave.extend([group_basis[:, index].detach().clone() for index in range(group_basis.shape[1])])
            elif group == "wave":
                group_basis = _select_wave_basis(length, group_count, selected_prefix=selected_for_wave, seed=seed, device=None)
                selected_for_wave.extend([group_basis[:, index].detach().clone() for index in range(group_basis.shape[1])])
            elif group == "random":
                group_basis = _random_residual_basis(length, group_count, selected_prefix=selected_for_wave, seed=seed, device=None)
                selected_for_wave.extend([group_basis[:, index].detach().clone() for index in range(group_basis.shape[1])])
            elif group == "mask":
                # Mask atoms represent DCT columns excluded from the active bank.
                # Selecting them from the full K-column DCT bank avoids duplicating
                # retained high-frequency columns in masked-control configurations.
                # Select unused frequencies from the full K-column DCT bank so
                # the masked control is numerically well-formed while projection
                # still disables the mask group during training/evaluation.
                full_dct_frequencies = _dct_frequency_indices(length, count, min(int(low_count), int(count)), device=None)
                selected_frequencies: list[int] = []
                for frequency in full_dct_frequencies.detach().cpu().tolist():
                    frequency_int = int(frequency)
                    if frequency_int in used_dct_frequencies:
                        continue
                    selected_frequencies.append(frequency_int)
                    if len(selected_frequencies) >= group_count:
                        break
                fallback_frequency = 0
                while len(selected_frequencies) < group_count:
                    if fallback_frequency not in used_dct_frequencies and fallback_frequency not in selected_frequencies:
                        selected_frequencies.append(fallback_frequency)
                    fallback_frequency += 1
                    if fallback_frequency > max(int(length) + int(count), int(group_count) * 4):
                        selected_frequencies.append(fallback_frequency % max(1, int(length)))
                dct_frequencies = torch.tensor(selected_frequencies[:group_count], dtype=torch.long)
                group_basis = _dct_basis_from_frequencies(length, dct_frequencies, device=None)
                used_dct_frequencies.update(int(item) for item in dct_frequencies.detach().cpu().tolist())
                selected_for_wave.extend([group_basis[:, index].detach().clone() for index in range(group_basis.shape[1])])
            else:
                raise ValueError(f"Unsupported basis primitive group={group!r}")
            columns.append(group_basis)
            group_ids.extend([_GROUP_NAME_TO_ID[group]] * group_basis.shape[1])
        if columns:
            basis = _column_normalize(torch.cat(columns, dim=1))
        else:
            basis = torch.empty((length, 0), dtype=torch.float32)
        if basis.shape != (int(length), int(count)):
            raise ValueError(f"basis_type={basis_type!r} produced shape={tuple(basis.shape)} expected={(int(length), int(count))}")
        group_tensor = torch.tensor(group_ids, dtype=torch.long)
        bank_hash = _basis_bank_hash(length=length, count=count, basis_type=basis_type, low_count=low_count, seed=seed, primitive_spec=specs)
        cached = (basis, group_tensor, bank_hash)
        _BASIS_BANK_CACHE[key] = cached
    basis, group_tensor, bank_hash = cached
    if device is not None:
        moved_basis = basis.to(device)
        moved_group_tensor = group_tensor.to(device)
        if bool(clone):
            return moved_basis.clone(), moved_group_tensor.clone(), bank_hash
        return moved_basis, moved_group_tensor, bank_hash
    if bool(clone):
        return basis.clone(), group_tensor.clone(), bank_hash
    return basis, group_tensor, bank_hash
