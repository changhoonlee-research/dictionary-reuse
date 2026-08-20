"""Matrix-level statistics for DiR functional correspondence."""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product
import random
from typing import Any, Sequence

import numpy as np


OBJECTIVES = {"maximize", "minimize"}


def _as_square(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"Expected a square matrix, got {value.shape}")
    return value


def _as_valid_mask(mask: Sequence[Sequence[bool]] | None, size: int) -> np.ndarray:
    if mask is None:
        return np.ones((size, size), dtype=bool)
    value = np.asarray(mask, dtype=bool)
    if value.shape != (size, size):
        raise ValueError(f"DiR validity mask shape mismatch: expected {(size, size)}, got {value.shape}")
    return value


def _validated_square_and_mask(
    matrix: Sequence[Sequence[float]],
    valid_mask: Sequence[Sequence[bool]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate only cells declared scientifically usable.

    Measurement matrices intentionally store NaN in invalid/low-signal cells.
    Those sentinels are legal only where the explicit validity mask is false.
    """

    value = _as_square(matrix)
    declared_valid = _as_valid_mask(valid_mask, int(value.shape[0]))
    # A measurement can legitimately become non-finite in a low-signal cell.
    # Statistics use only the explicit scientific validity mask intersected
    # with cells that are actually finite in this result.
    valid = declared_valid & np.isfinite(value)
    return value, valid


def _average_descending_rank(scores: np.ndarray, target_index: int) -> float:
    """Return a tie-aware one-based average rank for a descending score list."""

    target = float(scores[int(target_index)])
    greater = int(np.sum(scores > target))
    equal = int(np.sum(scores == target))
    return float(1 + greater + 0.5 * max(0, equal - 1))


def _objective_scores(value: np.ndarray, objective: str) -> np.ndarray:
    if str(objective) not in OBJECTIVES:
        raise ValueError(f"Unknown DiR matrix objective: {objective}")
    return value if str(objective) == "maximize" else -value


def _depth_bands(size: int) -> list[range]:
    if size == 12:
        return [range(0, 4), range(4, 8), range(8, 12)]
    boundaries = np.linspace(0, size, 4, dtype=int)
    return [range(int(boundaries[i]), int(boundaries[i + 1])) for i in range(3)]


def _valid_permutation_count_capped(
    valid: np.ndarray,
    rows: Sequence[int],
    columns: Sequence[int],
    *,
    cap: int = 2,
) -> int:
    """Count valid injective row-to-column assignments up to ``cap``.

    Only rows that contribute to the observed diagonal statistic need a valid
    mapped cell. Unevaluated rows can always fill the remaining permutation
    columns afterwards, so requiring them to pass the validity mask would make
    sparse masks falsely appear to have no usable null permutation.
    """

    row_values = [int(value) for value in rows]
    column_values = [int(value) for value in columns]
    if len(row_values) > len(column_values):
        return 0
    if not row_values:
        return 1
    dp = {0: 1}
    for row in row_values:
        next_dp: dict[int, int] = {}
        for mask, count in dp.items():
            for local_column, column in enumerate(column_values):
                bit = 1 << local_column
                if mask & bit or not bool(valid[row, column]):
                    continue
                new_mask = mask | bit
                next_dp[new_mask] = min(
                    int(cap), int(next_dp.get(new_mask, 0)) + int(count)
                )
        dp = next_dp
        if not dp:
            return 0
    return min(int(cap), int(sum(dp.values())))


def _has_alternative_valid_permutation(
    valid: np.ndarray,
    groups: Sequence[Sequence[int]],
    active_rows: Sequence[int],
) -> bool:
    """Return whether the scored rows admit a non-identity valid permutation."""

    active = {int(value) for value in active_rows}
    total = 1
    for group in groups:
        columns = [int(value) for value in group]
        rows = [value for value in columns if value in active]
        count = _valid_permutation_count_capped(valid, rows, columns, cap=2)
        if count == 0:
            return False
        total *= count
        if total > 1:
            return True
    return False


def _sample_valid_grouped_permutation(
    valid: np.ndarray,
    groups: Sequence[Sequence[int]],
    active_rows: Sequence[int],
    rng: random.Random,
) -> list[int] | None:
    """Sample a valid permutation directly instead of rejection sampling.

    For each group, scored rows are assigned injectively to valid columns. A
    dynamic-programming completion count makes the scored-row assignment
    uniform over all valid injective assignments. Unscored rows are then
    filled uniformly from the remaining columns; they never affect the null
    score but keep the returned mapping a genuine permutation.
    """

    size = int(valid.shape[0])
    permutation = list(range(size))
    active = {int(value) for value in active_rows}
    for raw_group in groups:
        columns = tuple(int(value) for value in raw_group)
        rows = tuple(value for value in columns if value in active)
        inactive_rows = [value for value in columns if value not in active]
        if not rows:
            shuffled = list(columns)
            rng.shuffle(shuffled)
            for row, column in zip(inactive_rows, shuffled):
                permutation[row] = column
            continue
        if all(bool(valid[row, column]) for row in rows for column in columns):
            shuffled = list(columns)
            rng.shuffle(shuffled)
            for row, column in zip(rows, shuffled[: len(rows)]):
                permutation[row] = column
            remaining_columns = shuffled[len(rows) :]
            for row, column in zip(inactive_rows, remaining_columns):
                permutation[row] = column
            continue

        @lru_cache(maxsize=None)
        def completion_count(position: int, used_mask: int) -> int:
            if position == len(rows):
                return 1
            row = rows[position]
            total = 0
            for local_column, column in enumerate(columns):
                bit = 1 << local_column
                if used_mask & bit or not bool(valid[row, column]):
                    continue
                total += completion_count(position + 1, used_mask | bit)
            return total

        total_count = completion_count(0, 0)
        if total_count <= 0:
            return None
        used_mask = 0
        for position, row in enumerate(rows):
            candidates: list[tuple[int, int, int]] = []
            candidate_total = 0
            for local_column, column in enumerate(columns):
                bit = 1 << local_column
                if used_mask & bit or not bool(valid[row, column]):
                    continue
                count = completion_count(position + 1, used_mask | bit)
                if count <= 0:
                    continue
                candidates.append((local_column, column, count))
                candidate_total += count
            if candidate_total <= 0:
                return None
            draw = rng.randrange(candidate_total)
            cumulative = 0
            selected_local = -1
            selected_column = -1
            for local_column, column, count in candidates:
                cumulative += count
                if draw < cumulative:
                    selected_local = local_column
                    selected_column = column
                    break
            if selected_local < 0:
                return None
            permutation[row] = selected_column
            used_mask |= 1 << selected_local

        remaining_columns = [
            column
            for local_column, column in enumerate(columns)
            if not (used_mask & (1 << local_column))
        ]
        rng.shuffle(remaining_columns)
        for row, column in zip(inactive_rows, remaining_columns):
            permutation[row] = column
    return permutation


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def same_index_advantage_vector(
    matrix: Sequence[Sequence[float]],
    *,
    objective: str = "maximize",
    valid_mask: Sequence[Sequence[bool]] | None = None,
    symmetric: bool = True,
) -> list[float]:
    """Depth-band matched same-index advantage for each block.

    The primary statistic is symmetric: it averages the row-wise Source→Target
    and column-wise Target→Source advantages. Invalid/low-signal cells are
    omitted rather than converted into arbitrary numeric scores.
    """

    value, valid = _validated_square_and_mask(matrix, valid_mask)
    scores = _objective_scores(value, objective)
    size = int(value.shape[0])
    index_to_band: dict[int, list[int]] = {}
    for band in _depth_bands(size):
        indices = list(band)
        for index in indices:
            index_to_band[index] = indices

    advantages: list[float] = []
    for index in range(size):
        if not bool(valid[index, index]):
            advantages.append(float("nan"))
            continue
        band = index_to_band.get(index, list(range(size)))
        row_candidates = [column for column in band if column != index and valid[index, column]]
        column_candidates = [row for row in band if row != index and valid[row, index]]
        row_advantage = (
            float(scores[index, index] - np.mean(scores[index, row_candidates]))
            if row_candidates
            else float("nan")
        )
        column_advantage = (
            float(scores[index, index] - np.mean(scores[column_candidates, index]))
            if column_candidates
            else float("nan")
        )
        if symmetric:
            advantages.append(_finite_mean([row_advantage, column_advantage]))
        else:
            advantages.append(row_advantage)
    return advantages


def matrix_summary(
    matrix: Sequence[Sequence[float]],
    *,
    objective: str = "maximize",
    valid_mask: Sequence[Sequence[bool]] | None = None,
) -> dict[str, Any]:
    value, valid = _validated_square_and_mask(matrix, valid_mask)
    scores = _objective_scores(value, objective)
    size = int(value.shape[0])
    diagonal_valid = np.asarray([valid[i, i] for i in range(size)], dtype=bool)
    diagonal = np.asarray([value[i, i] for i in range(size) if diagonal_valid[i]], dtype=np.float64)
    score_diagonal = np.asarray([scores[i, i] for i in range(size) if diagonal_valid[i]], dtype=np.float64)
    off_mask = valid & ~np.eye(size, dtype=bool)
    off_values = value[off_mask]
    off_scores = scores[off_mask]

    row_ranks: list[float] = []
    column_ranks: list[float] = []
    symmetric_ranks: list[float] = []
    symmetric_rank1_values: list[float] = []
    symmetric_top3_values: list[float] = []
    paired_two_direction_rank1_flags: list[bool] = []
    row_top3 = 0
    column_top3 = 0
    local_margins: list[float] = []
    for index in range(size):
        if not valid[index, index]:
            continue
        row_indices = np.flatnonzero(valid[index])
        column_indices = np.flatnonzero(valid[:, index])
        row_has_competitor = bool(np.any(row_indices != index))
        column_has_competitor = bool(np.any(column_indices != index))
        if row_has_competitor:
            row_scores = scores[index, row_indices]
            row_target = int(np.where(row_indices == index)[0][0])
            row_rank = _average_descending_rank(row_scores, row_target)
            row_ranks.append(row_rank)
            row_top3 += int(row_rank <= 3)
        else:
            row_rank = None
        if column_has_competitor:
            column_scores = scores[column_indices, index]
            column_target = int(np.where(column_indices == index)[0][0])
            column_rank = _average_descending_rank(column_scores, column_target)
            column_ranks.append(column_rank)
            column_top3 += int(column_rank <= 3)
        else:
            column_rank = None
        available_ranks = [
            float(rank) for rank in (row_rank, column_rank) if rank is not None
        ]
        if available_ranks:
            symmetric_ranks.append(float(np.mean(available_ranks)))
            symmetric_rank1_values.append(
                float(np.mean([float(rank == 1) for rank in available_ranks]))
            )
            symmetric_top3_values.append(
                float(np.mean([float(rank <= 3) for rank in available_ranks]))
            )
        if row_rank is not None and column_rank is not None:
            paired_two_direction_rank1_flags.append(
                bool(row_rank == 1 and column_rank == 1)
            )
        local_indices = [
            neighbour
            for neighbour in (index - 1, index + 1)
            if 0 <= neighbour < size and valid[index, neighbour]
        ]
        if local_indices:
            local_margins.append(
                float(scores[index, index] - max(scores[index, neighbour] for neighbour in local_indices))
            )

    absolute_distance_off_means: dict[str, float] = {}
    absolute_distance_margins: dict[str, float] = {}
    diagonal_score_mean = _finite_mean(score_diagonal.tolist())
    for distance in range(1, size):
        cells = [
            (row, column)
            for row in range(size)
            for column in range(size)
            if abs(row - column) == distance and valid[row, column]
        ]
        if not cells:
            continue
        raw_candidates = [float(value[row, column]) for row, column in cells]
        score_candidates = [float(scores[row, column]) for row, column in cells]
        absolute_distance_off_means[str(distance)] = float(np.mean(raw_candidates))
        absolute_distance_margins[str(distance)] = diagonal_score_mean - float(np.mean(score_candidates))

    cyclic: dict[str, float] = {}
    for shift in range(1, size):
        values = [
            float(value[row, (row + shift) % size])
            for row in range(size)
            if valid[row, (row + shift) % size]
        ]
        cyclic[str(shift)] = _finite_mean(values)

    band_summaries = []
    for band_index, band in enumerate(_depth_bands(size)):
        indices = list(band)
        if not indices:
            continue
        band_diagonal = [value[i, i] for i in indices if valid[i, i]]
        band_diagonal_scores = [scores[i, i] for i in indices if valid[i, i]]
        band_off = [value[i, j] for i in indices for j in indices if i != j and valid[i, j]]
        band_off_scores = [scores[i, j] for i in indices for j in indices if i != j and valid[i, j]]
        band_summaries.append(
            {
                "band_index": band_index,
                "start": min(indices),
                "end_exclusive": max(indices) + 1,
                "valid_diagonal_count": len(band_diagonal),
                "valid_off_index_count": len(band_off),
                "diagonal_mean": _finite_mean(band_diagonal),
                "within_band_off_mean": _finite_mean(band_off),
                "alignment_margin": (
                    _finite_mean(band_diagonal_scores) - _finite_mean(band_off_scores)
                    if band_diagonal_scores and band_off_scores
                    else float("nan")
                ),
            }
        )

    advantages = same_index_advantage_vector(
        value.tolist(), objective=objective, valid_mask=valid.tolist(), symmetric=True
    )
    row_advantages = same_index_advantage_vector(
        value.tolist(), objective=objective, valid_mask=valid.tolist(), symmetric=False
    )
    column_advantages = same_index_advantage_vector(
        value.T.tolist(), objective=objective, valid_mask=valid.T.tolist(), symmetric=False
    )
    finite_advantages = np.asarray([v for v in advantages if np.isfinite(v)], dtype=np.float64)
    symmetric_direction_count_by_block = [
        int(np.isfinite(row_value)) + int(np.isfinite(column_value))
        for row_value, column_value in zip(row_advantages, column_advantages)
    ]

    def summary_or_nan(array: np.ndarray, operation: str) -> float:
        if array.size == 0:
            return float("nan")
        return float(getattr(np, operation)(array))

    return {
        "size": size,
        "objective": str(objective),
        "valid_cell_count": int(valid.sum()),
        "valid_diagonal_count": int(diagonal_valid.sum()),
        "valid_off_index_count": int(off_mask.sum()),
        "diagonal_mean": summary_or_nan(diagonal, "mean"),
        "diagonal_min": summary_or_nan(diagonal, "min"),
        "diagonal_max": summary_or_nan(diagonal, "max"),
        "off_index_mean": summary_or_nan(off_values, "mean"),
        "off_index_min": summary_or_nan(off_values, "min"),
        "off_index_max": summary_or_nan(off_values, "max"),
        "same_index_margin": (
            _finite_mean(score_diagonal.tolist()) - _finite_mean(off_scores.tolist())
            if score_diagonal.size and off_scores.size
            else float("nan")
        ),
        "depth_band_matched_same_index_margin": _finite_mean(advantages),
        "depth_band_matched_same_index_advantage_by_block": advantages,
        "row_depth_band_matched_same_index_advantage_by_block": row_advantages,
        "column_depth_band_matched_same_index_advantage_by_block": column_advantages,
        "symmetric_direction_count_by_block": symmetric_direction_count_by_block,
        "two_direction_block_count": int(sum(value == 2 for value in symmetric_direction_count_by_block)),
        "one_direction_block_count": int(sum(value == 1 for value in symmetric_direction_count_by_block)),
        "zero_direction_block_count": int(sum(value == 0 for value in symmetric_direction_count_by_block)),
        "symmetric_direction_contract": (
            "use_both_row_and_column_when_available_use_the_single_available_direction_without_exclusion_and_record_direction_count"
        ),
        "same_index_rank_mean": _finite_mean(symmetric_ranks),
        "same_index_rank_median": float(np.median(symmetric_ranks)) if symmetric_ranks else float("nan"),
        "row_same_index_rank_mean": _finite_mean(row_ranks),
        "column_same_index_rank_mean": _finite_mean(column_ranks),
        "row_rank1_fraction": float(np.mean(np.asarray(row_ranks) == 1)) if row_ranks else float("nan"),
        "column_rank1_fraction": float(np.mean(np.asarray(column_ranks) == 1)) if column_ranks else float("nan"),
        "rank1_fraction": _finite_mean(symmetric_rank1_values),
        "paired_two_direction_rank1_fraction": (
            float(np.mean(paired_two_direction_rank1_flags))
            if paired_two_direction_rank1_flags
            else float("nan")
        ),
        "row_rank_defined_count": len(row_ranks),
        "column_rank_defined_count": len(column_ranks),
        "symmetric_rank_defined_count": len(symmetric_ranks),
        "paired_two_direction_rank_defined_count": len(paired_two_direction_rank1_flags),
        "rank_statistics_status": (
            "available" if symmetric_ranks else "inconclusive_no_valid_off_diagonal_competitor"
        ),
        "rank_contract": (
            "per_block_symmetric_rank_rank1_and_top3_average_all_available_row_and_column_directions_"
            "and_keep_the_single_available_direction_when_only_one_has_a_valid_off_diagonal_competitor;_"
            "paired_two_direction_rank1_is_reported_separately"
        ),
        "top3_fraction": _finite_mean(symmetric_top3_values),
        "local_neighbor_margin_mean": _finite_mean(local_margins),
        "distance_matched_margin_mean": _finite_mean(list(absolute_distance_margins.values())),
        "absolute_block_distance_off_means": absolute_distance_off_means,
        "absolute_block_distance_margins": absolute_distance_margins,
        "distance_matched_contract": "equal_weight_average_across_valid_absolute_off_index_block_distances",
        "depth_band_baseline_contract": (
            "average_row_and_column_depth_band_advantages_when_both_exist_otherwise_keep_the_single_available_direction_and_record_coverage"
        ),
        "low_signal_contract": "invalid_cells_are_excluded_not_zero_filled",
        "cyclic_shift_diagonal_means": cyclic,
        "depth_bands": band_summaries,
        "finite_advantage_count": int(finite_advantages.size),
        "diagonal_statistics_status": (
            "available" if int(diagonal_valid.sum()) > 0 else "inconclusive_no_valid_diagonal"
        ),
        "advantage_statistics_status": (
            "available" if int(finite_advantages.size) > 0 else "inconclusive_no_valid_depth_band_competitor"
        ),
    }


def block_axis_permutation_test(
    matrix: Sequence[Sequence[float]],
    *,
    global_permutations: int,
    depth_band_permutations: int,
    seed: int,
    objective: str = "maximize",
    valid_mask: Sequence[Sequence[bool]] | None = None,
) -> dict[str, Any]:
    value, valid = _validated_square_and_mask(matrix, valid_mask)
    scores = _objective_scores(value, objective)
    size = int(value.shape[0])
    diagonal_indices = [index for index in range(size) if valid[index, index]]
    if not diagonal_indices:
        unavailable = {
            "status": "inconclusive_no_valid_diagonal",
            "null_alignment_score_mean": float("nan"),
            "null_alignment_score_std": float("nan"),
            "p_alignment_at_least_observed": float("nan"),
            "z_score": float("nan"),
        }
        return {
            "status": "inconclusive_no_valid_diagonal",
            "objective": str(objective),
            "observed_diagonal_mean": float("nan"),
            "observed_alignment_score_mean": float("nan"),
            "valid_diagonal_count": 0,
            "global": dict(unavailable),
            "depth_band": dict(unavailable),
            "global_permutation_count": 0,
            "global_permutation_method": "inconclusive_no_valid_diagonal",
            "global_has_alternative_valid_permutation": False,
            "depth_band_permutation_count": 0,
            "depth_band_has_alternative_valid_permutation": False,
            "depth_band_permutation_method": "inconclusive_no_valid_diagonal",
            "seed": int(seed),
        }
    observed_raw = float(np.mean([value[i, i] for i in diagonal_indices]))
    observed_score = float(np.mean([scores[i, i] for i in diagonal_indices]))
    rng = random.Random(int(seed))
    indices = list(range(size))
    global_has_alternative = _has_alternative_valid_permutation(valid, [indices], diagonal_indices)

    def permutation_score(permutation: Sequence[int]) -> float | None:
        selected = [scores[row, permutation[row]] for row in diagonal_indices if valid[row, permutation[row]]]
        if len(selected) != len(diagonal_indices):
            return None
        return float(np.mean(selected))

    global_null: list[float] = []
    if global_has_alternative:
        for _ in range(int(global_permutations)):
            permutation = _sample_valid_grouped_permutation(
                valid, [indices], diagonal_indices, rng
            )
            if permutation is None:
                break
            score = permutation_score(permutation)
            if score is None:
                raise RuntimeError("DiR direct valid global permutation sampler returned an invalid assignment")
            global_null.append(score)

    bands = [list(band) for band in _depth_bands(size)]
    depth_band_has_alternative = _has_alternative_valid_permutation(valid, bands, diagonal_indices)
    exact_band_possible = (
        size == 12
        and all(len(band) == 4 for band in bands)
        and all(valid[row, column] for band in bands for row in band for column in band)
    )
    band_null: list[float] = []
    if exact_band_possible:
        band_permutations = [list(permutations(band)) for band in bands]
        for selected in product(*band_permutations):
            permutation = indices[:]
            for band, permuted in zip(bands, selected):
                for source, target in zip(band, permuted):
                    permutation[source] = target
            score = permutation_score(permutation)
            if score is not None:
                band_null.append(score)
        depth_band_method = "exact_all_24_cubed_within_band_permutations"
    else:
        if depth_band_has_alternative:
            for _ in range(int(depth_band_permutations)):
                permutation = _sample_valid_grouped_permutation(
                    valid, bands, diagonal_indices, rng
                )
                if permutation is None:
                    break
                score = permutation_score(permutation)
                if score is None:
                    raise RuntimeError("DiR direct valid depth-band permutation sampler returned an invalid assignment")
                band_null.append(score)
        depth_band_method = (
            "monte_carlo_direct_valid_within_band_permutations"
            if depth_band_has_alternative
            else "inconclusive_no_alternative_valid_within_band_permutation"
        )

    def summarize(null: list[float], *, exact: bool) -> dict[str, Any]:
        if not null:
            return {
                "status": "inconclusive_no_valid_permutations",
                "null_alignment_score_mean": float("nan"),
                "null_alignment_score_std": float("nan"),
                "p_alignment_at_least_observed": float("nan"),
                "z_score": float("nan"),
            }
        array = np.asarray(null, dtype=np.float64)
        exceed = int(np.count_nonzero(array >= observed_score - 1e-15))
        p_value = exceed / len(array) if exact else (1 + exceed) / (len(array) + 1)
        return {
            "status": "available",
            "null_alignment_score_mean": float(array.mean()),
            "null_alignment_score_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "p_alignment_at_least_observed": float(p_value),
            "z_score": (
                float((observed_score - array.mean()) / max(1e-12, array.std(ddof=1)))
                if len(array) > 1
                else float("nan")
            ),
        }

    global_summary = summarize(global_null, exact=False)
    depth_band_summary = summarize(band_null, exact=exact_band_possible)
    available_null_count = int(global_summary["status"] == "available") + int(
        depth_band_summary["status"] == "available"
    )
    return {
        "status": (
            "available"
            if available_null_count == 2
            else ("partial" if available_null_count == 1 else "inconclusive_no_valid_permutations")
        ),
        "objective": str(objective),
        "observed_diagonal_mean": observed_raw,
        "observed_alignment_score_mean": observed_score,
        "valid_diagonal_count": len(diagonal_indices),
        "global": global_summary,
        "depth_band": depth_band_summary,
        "global_permutation_count": len(global_null),
        "global_permutation_method": (
            "monte_carlo_direct_valid_global_permutations"
            if global_has_alternative
            else "inconclusive_no_alternative_valid_global_permutation"
        ),
        "global_has_alternative_valid_permutation": bool(global_has_alternative),
        "depth_band_permutation_count": len(band_null),
        "depth_band_has_alternative_valid_permutation": bool(depth_band_has_alternative),
        "depth_band_permutation_method": depth_band_method,
        "seed": int(seed),
    }


def paired_bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("DiR paired bootstrap inputs must be equal-length vectors")
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    left_array = left_array[finite]
    right_array = right_array[finite]
    if len(left_array) == 0:
        return {
            "status": "inconclusive_no_common_finite_pairs",
            "paired_mean_difference": float("nan"),
            "bootstrap_mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "paired_block_count": 0,
            "bootstrap_iterations": int(iterations),
            "seed": int(seed),
        }
    rng = np.random.default_rng(int(seed))
    differences = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sample = rng.integers(0, len(left_array), size=len(left_array))
        differences[index] = np.mean(left_array[sample] - right_array[sample])
    return {
        "status": "available",
        "paired_mean_difference": float(np.mean(left_array - right_array)),
        "bootstrap_mean": float(np.mean(differences)),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "paired_block_count": int(len(left_array)),
        "bootstrap_iterations": int(iterations),
        "seed": int(seed),
    }
