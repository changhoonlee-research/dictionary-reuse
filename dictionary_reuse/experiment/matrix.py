"""Canonical final-paper experiment matrix.

Only this module defines which trained endpoints are compared. Training and
measurement orchestration consume these declarations instead of duplicating
condition-specific branches throughout the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementPairSpec:
    condition: str
    left_model: str
    right_model: str
    task_key: str
    same_task: bool
    family: str


# Final paper matrix:
#   same-task / different-seed: DiR vs Dense
#   different-task: DiR Dictionary-Fixed vs Dictionary-Trainable vs Dense Full-Transfer
# "Dictionary-Fixed" means Source-active D slices/D-owned scales are anchored;
# Source-inactive D remains Target-trainable only on phase-allowed dictionary
# coordinates: internal-facing block D plus included head D; residual-facing/endpoint
# D is still fixed by ``internal_only``.
MEASUREMENT_PAIRS: tuple[MeasurementPairSpec, ...] = (
    MeasurementPairSpec(
        condition="dir_same_task",
        left_model="dir_source",
        right_model="dir_same_task",
        task_key="task1",
        same_task=True,
        family="same_task",
    ),
    MeasurementPairSpec(
        condition="dense_same_task",
        left_model="dense_source",
        right_model="dense_same_task",
        task_key="task1",
        same_task=True,
        family="same_task",
    ),
    MeasurementPairSpec(
        condition="dir_dictionary_fixed",
        left_model="dir_source",
        right_model="dir_dictionary_fixed",
        task_key="task2",
        same_task=False,
        family="different_task",
    ),
    MeasurementPairSpec(
        condition="dir_dictionary_trainable",
        left_model="dir_source",
        right_model="dir_dictionary_trainable",
        task_key="task2",
        same_task=False,
        family="different_task",
    ),
    MeasurementPairSpec(
        condition="dense_different_task",
        left_model="dense_source",
        right_model="dense_different_task",
        task_key="task2",
        same_task=False,
        family="different_task",
    ),
)

CONDITION_ORDER: tuple[str, ...] = tuple(spec.condition for spec in MEASUREMENT_PAIRS)
PAIR_BY_CONDITION = {spec.condition: spec for spec in MEASUREMENT_PAIRS}

# Comparisons reported in the paper-facing summary. Positive differences favor
# the left condition after each metric's maximize/minimize orientation is applied.
SUMMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("same_task_dir_minus_dense", "dir_same_task", "dense_same_task"),
    ("different_task_dictionary_fixed_minus_dense", "dir_dictionary_fixed", "dense_different_task"),
    ("different_task_dictionary_fixed_minus_dictionary_trainable", "dir_dictionary_fixed", "dir_dictionary_trainable"),
)

# Training endpoints. Source runs are intentionally outside CONDITION_ORDER
# because they are reference models rather than Source-vs-Target comparisons.
TARGET_RUNS: tuple[str, ...] = CONDITION_ORDER

SOURCE_RUNS: tuple[str, ...] = ("dir_source_a", "dense_source_a")
