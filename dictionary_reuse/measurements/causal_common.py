"""Shared primitives for causal correspondence measurements."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .representation_similarity import _feature_view

def _true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    true_value = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return true_value - masked.max(dim=1).values


def _response_signature(
    baseline_logits: torch.Tensor,
    baseline_taps: Mapping[str, torch.Tensor],
    intervention_logits: torch.Tensor,
    intervention_taps: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    intervention_block: int,
    include_logit_comparison: bool,
    include_label_metrics: bool,
) -> torch.Tensor:
    """Compact causal profile with task semantics separated explicitly.

    Prediction-index changes require only a shared head semantic. True-class
    margins additionally require that the current dataset labels are native to
    that shared head. This prevents task2 local labels from being interpreted
    as task1 classes in the same-task/OOD diagnostic.
    """

    depth = sum(1 for key in baseline_taps if key.endswith("_update"))
    values: list[torch.Tensor] = []
    for receiver in range(intervention_block + 1, depth):
        baseline = baseline_taps[f"block_{receiver:02d}_update"]
        changed = intervention_taps[f"block_{receiver:02d}_update"]
        values.append(
            (changed.float() - baseline.float())
            .reshape(baseline.shape[0], -1)
            .square()
            .mean(dim=1)
            .sqrt()
        )
    baseline_rep = baseline_taps["pre_classifier"]
    changed_rep = intervention_taps["pre_classifier"]
    post_layernorm_delta = changed_rep.float() - baseline_rep.float()
    values.append(
        post_layernorm_delta.reshape(baseline_rep.shape[0], -1)
        .square()
        .mean(dim=1)
        .sqrt()
    )
    values.append(_feature_view(post_layernorm_delta, "cls").square().mean(dim=1).sqrt())
    values.append(_feature_view(post_layernorm_delta, "patch").square().mean(dim=1).sqrt())
    if include_label_metrics:
        values.append(
            _true_class_margin(intervention_logits, labels)
            - _true_class_margin(baseline_logits, labels)
        )
    if include_logit_comparison:
        values.append(
            (intervention_logits.argmax(dim=1) != baseline_logits.argmax(dim=1)).float()
        )
    return torch.stack(values, dim=1).cpu()


def _structural_response_profiles(
    values: Sequence[torch.Tensor],
    *,
    appended_task_columns: int,
) -> list[torch.Tensor]:
    """Remove task-semantic columns independently for each intervention depth."""

    count = int(appended_task_columns)
    if count < 0:
        raise ValueError("DiR appended_task_columns cannot be negative")
    structural: list[torch.Tensor] = []
    for value in values:
        if value.ndim != 2 or int(value.shape[1]) < count:
            raise ValueError("DiR scalar response profile has an invalid column count")
        stop = int(value.shape[1]) - count
        structural.append(value[:, :stop] if count else value)
    return structural


def _normalize_causal_intervention_points(
    intervention_points: Sequence[str],
) -> tuple[str, ...]:
    allowed = (
        "block_update",
        "post_o_attention_output",
        "post_w2_mlp_output",
    )
    normalized = tuple(str(value) for value in intervention_points)
    if not normalized:
        raise ValueError("DiR causal intervention point set must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("DiR causal intervention points must be unique")
    unknown = [value for value in normalized if value not in allowed]
    if unknown:
        raise ValueError(f"Unsupported DiR causal intervention points: {unknown}")
    return normalized


def _causal_point_exception_result(
    *,
    intervention_point: str,
    stage: str,
    errors: Sequence[Mapping[str, Any]],
    cleanup_status: Mapping[str, Any],
    corruption: str | None = None,
) -> dict[str, Any]:
    """Return a point-local failure without invalidating sibling suite points."""

    payload: dict[str, Any] = {
        "measurement_status": "warning_point_exception",
        "intervention_point": str(intervention_point),
        "point_failure_stage": str(stage),
        "point_errors": [dict(value) for value in errors],
        "primary_metrics": [],
        "validity_masks": {},
        # Keep the shared mutable status object so the context-manager cleanup
        # result (completed or warning_cleanup_failed) is visible even when
        # this point failed before suite exit.
        "cache_cleanup": cleanup_status,
        "shared_suite_point_isolation_contract": (
            "one_intervention_point_failure_does_not_discard_sibling_point_results"
        ),
    }
    if corruption is not None:
        payload["corruption"] = str(corruption)
    return payload
