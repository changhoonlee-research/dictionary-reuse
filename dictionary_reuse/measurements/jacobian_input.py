"""Input-response JVP and randomized Jacobian-SVD alignment."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import forward_with_capture_and_interventions
from .representation_similarity import _degenerate_aware_subspace_overlap, _jacobian_gram_similarity_matrix, _padded_singular_spectrum_cosine, centered_gram_cka
from .representation_cache import gram_variation_strength, combined_signal_variation_validity, pairwise_paired_output_metric_matrices
from .jacobian_common import (
    _finalize_randomized_svd_descriptor,
    _measurement_status_from_primary_validity,
    _numerical_svd_rank,
    _flat_rank_correlation,
    _rademacher,
    _shared_input_probe_direction,
    _shared_probe_projection_seed,
)



class _InputJacobianRuntime:
    """Bound runtime state for input-Jacobian helper operations.

    Keeping execution state here makes the public measurement function read as
    a linear scientific pipeline while preserving the original tensor order.
    """

    def __init__(
        self,
        *,
        sample_count,
        half,
        device,
        depth,
        probe_count,
        microbatch_size,
        all_images_cpu,
        probe_seed,
        normalization_std,
        raw_probe_columns,
        randomized_svd_rank,
        range_holdout_relative_residual_maximum,
        minimum_signal_rms_absolute,
        cache_sample_key,
        model_descriptor_cache,
        split_half_spearman_minimum,
        split_half_diagonal_difference_maximum,
        split_half_norm_relative_difference_maximum,
    ) -> None:
        self.sample_count = sample_count
        self.half = half
        self.device = device
        self.depth = depth
        self.probe_count = probe_count
        self.microbatch_size = microbatch_size
        self.all_images_cpu = all_images_cpu
        self.probe_seed = probe_seed
        self.normalization_std = normalization_std
        self.raw_probe_columns = raw_probe_columns
        self.randomized_svd_rank = randomized_svd_rank
        self.range_holdout_relative_residual_maximum = range_holdout_relative_residual_maximum
        self.minimum_signal_rms_absolute = minimum_signal_rms_absolute
        self.cache_sample_key = cache_sample_key
        self.model_descriptor_cache = model_descriptor_cache
        self.split_half_spearman_minimum = split_half_spearman_minimum
        self.split_half_diagonal_difference_maximum = split_half_diagonal_difference_maximum
        self.split_half_norm_relative_difference_maximum = split_half_norm_relative_difference_maximum


    def empty_accumulator(self, *, retain_range: bool) -> dict[str, Any]:
        return {
            "gram": torch.zeros(self.sample_count, self.sample_count, dtype=torch.float64),
            "first_gram": torch.zeros(self.sample_count, self.sample_count, dtype=torch.float64),
            "second_gram": torch.zeros(self.sample_count, self.sample_count, dtype=torch.float64),
            "square_sum": 0.0,
            "element_count": 0,
            "first_square_sum": 0.0,
            "first_element_count": 0,
            "second_square_sum": 0.0,
            "second_element_count": 0,
            "probe_projection_rows": [],
            "mean_response_rows": [] if retain_range else None,
        }


    def update_accumulator(
        self,
        accumulator: dict[str, Any],
        flattened: torch.Tensor,
        gram: torch.Tensor,
        *,
        probe_index: int,
        projection_seed: int,
    ) -> None:
        if flattened.ndim != 2 or int(flattened.shape[0]) != self.sample_count:
            raise ValueError("DiR JVP flattened response has an invalid shape")
        accumulator["gram"].add_(gram)
        square_sum = float(flattened.square().sum())
        element_count = int(flattened.numel())
        if int(probe_index) < self.half:
            accumulator["first_gram"].add_(gram)
            accumulator["first_square_sum"] += square_sum
            accumulator["first_element_count"] += element_count
        else:
            accumulator["second_gram"].add_(gram)
            accumulator["second_square_sum"] += square_sum
            accumulator["second_element_count"] += element_count
        accumulator["square_sum"] += square_sum
        accumulator["element_count"] += element_count
        output_direction = _rademacher(
            (1, int(flattened.shape[1])),
            seed=int(projection_seed),
            device=torch.device("cpu"),
            dtype=flattened.dtype,
        )
        accumulator["probe_projection_rows"].append(
            (flattened * output_direction).sum(dim=1)
        )
        if accumulator["mean_response_rows"] is not None:
            accumulator["mean_response_rows"].append(flattened.mean(dim=0))


    def finalize_accumulator(self, accumulator: dict[str, Any]) -> dict[str, Any]:
        projected = torch.stack(accumulator.pop("probe_projection_rows"), dim=0).float()
        accumulator["rms"] = math.sqrt(
            float(accumulator["square_sum"]) / max(1, int(accumulator["element_count"]))
        )
        accumulator["first_rms"] = math.sqrt(
            float(accumulator["first_square_sum"])
            / max(1, int(accumulator["first_element_count"]))
        )
        accumulator["second_rms"] = math.sqrt(
            float(accumulator["second_square_sum"])
            / max(1, int(accumulator["second_element_count"]))
        )
        accumulator["projection_features"] = projected.T.contiguous()
        accumulator["first_projection_features"] = projected[:self.half].T.contiguous()
        accumulator["second_projection_features"] = projected[self.half:].T.contiguous()
        mean_rows = accumulator.pop("mean_response_rows")
        if mean_rows is not None:
            accumulator["range_matrix"] = torch.stack(mean_rows, dim=1).float()
        return accumulator


    def response_gram(self, flattened: torch.Tensor) -> torch.Tensor:
        runtime = flattened.to(device=self.device, dtype=torch.float32)
        value = (runtime @ runtime.T).double().cpu()
        del runtime
        return value


    def model_jvp_sketch(
        self,
        model: nn.Module,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        model.eval().to(self.device)
        block_full_accumulators = [self.empty_accumulator(retain_range=False) for _ in range(self.depth)]
        block_cls_accumulators = [self.empty_accumulator(retain_range=True) for _ in range(self.depth)]
        block_patch_accumulators = [self.empty_accumulator(retain_range=True) for _ in range(self.depth)]
        class_token_accumulators = [self.empty_accumulator(retain_range=False) for _ in range(self.depth)]
        capture_points = [
            *[f"block_{i:02d}_update" for i in range(self.depth)],
            *[f"block_{i:02d}_output" for i in range(self.depth)],
        ]
        for probe_index in range(int(self.probe_count)):
            block_cls_chunks: list[list[torch.Tensor]] = [[] for _ in range(self.depth)]
            block_patch_chunks: list[list[torch.Tensor]] = [[] for _ in range(self.depth)]
            class_token_chunks: list[list[torch.Tensor]] = [[] for _ in range(self.depth)]
            for start_index in range(0, self.sample_count, self.microbatch_size):
                stop_index = min(self.sample_count, start_index + self.microbatch_size)
                images = (
                    self.all_images_cpu[start_index:stop_index]
                    .to(self.device)
                    .float()
                    .requires_grad_(True)
                )
                direction = _shared_input_probe_direction(
                    images,
                    seed=int(self.probe_seed) + 100003 * probe_index,
                    normalization_std=self.normalization_std,
                )

                def function(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
                    _logits, taps = forward_with_capture_and_interventions(
                        model, value, capture_points=capture_points
                    )
                    return tuple(
                        [taps[f"block_{index:02d}_update"] for index in range(self.depth)]
                        + [
                            taps[f"block_{index:02d}_output"][:, 0]
                            for index in range(self.depth)
                        ]
                    )

                _primal, tangent = torch.autograd.functional.jvp(
                    function,
                    images,
                    direction,
                    create_graph=False,
                    strict=False,
                )
                for index in range(self.depth):
                    update_tangent = tangent[index].detach().cpu()
                    block_cls_chunks[index].append(update_tangent[:, 0])
                    block_patch_chunks[index].append(update_tangent[:, 1:])
                    class_token_chunks[index].append(tangent[self.depth + index].detach().cpu())
                del tangent, _primal, direction, images
            for index in range(self.depth):
                cls_flat = torch.cat(block_cls_chunks[index], dim=0).float().reshape(self.sample_count, -1)
                patch_flat = torch.cat(block_patch_chunks[index], dim=0).float().reshape(self.sample_count, -1)
                class_flat = torch.cat(class_token_chunks[index], dim=0).float().reshape(self.sample_count, -1)
                cls_gram = self.response_gram(cls_flat)
                patch_gram = self.response_gram(patch_flat)
                full_flat = torch.cat((cls_flat, patch_flat), dim=1)
                self.update_accumulator(
                    block_full_accumulators[index],
                    full_flat,
                    cls_gram + patch_gram,
                    probe_index=probe_index,
                    projection_seed=_shared_probe_projection_seed(
                        self.probe_seed, probe_index, family_offset=7000003
                    ),
                )
                self.update_accumulator(
                    block_cls_accumulators[index],
                    cls_flat,
                    cls_gram,
                    probe_index=probe_index,
                    projection_seed=_shared_probe_projection_seed(
                        self.probe_seed, probe_index, family_offset=8000009
                    ),
                )
                self.update_accumulator(
                    block_patch_accumulators[index],
                    patch_flat,
                    patch_gram,
                    probe_index=probe_index,
                    projection_seed=_shared_probe_projection_seed(
                        self.probe_seed, probe_index, family_offset=8500007
                    ),
                )
                self.update_accumulator(
                    class_token_accumulators[index],
                    class_flat,
                    self.response_gram(class_flat),
                    probe_index=probe_index,
                    projection_seed=_shared_probe_projection_seed(
                        self.probe_seed, probe_index, family_offset=9000011
                    ),
                )
                del cls_flat, patch_flat, class_flat, full_flat, cls_gram, patch_gram
            del block_cls_chunks, block_patch_chunks, class_token_chunks
        return (
            [self.finalize_accumulator(value) for value in block_full_accumulators],
            [self.finalize_accumulator(value) for value in block_cls_accumulators],
            [self.finalize_accumulator(value) for value in block_patch_accumulators],
            [self.finalize_accumulator(value) for value in class_token_accumulators],
        )


    def orthonormal_range_bases(self, values: Sequence[dict[str, Any]]) -> list[torch.Tensor]:
        bases: list[torch.Tensor] = []
        for value in values:
            matrix = value["range_matrix"].double()
            if not torch.isfinite(matrix).all():
                raise ValueError("DiR randomized Jacobian range sketch is non-finite")
            response_rms = float(matrix.square().mean().sqrt().cpu())
            u, singular, _vh = torch.linalg.svd(matrix, full_matrices=False)
            rank, tolerance = _numerical_svd_rank(
                singular,
                row_count=int(matrix.shape[0]),
                column_count=int(matrix.shape[1]),
            )
            if response_rms <= float(self.minimum_signal_rms_absolute):
                rank = 0
            value["range_response_rms"] = float(response_rms)
            value["below_detection_signal_rms_threshold"] = float(self.minimum_signal_rms_absolute)
            value["range_rank_tolerance"] = float(tolerance)
            value["numerical_range_rank"] = int(rank)
            bases.append(u[:, :rank].contiguous().cpu())
        return bases


    def range_holdout_diagnostic(self, range_matrix: torch.Tensor) -> dict[str, Any]:
        """Use the last eight fixed probes to audit the range found by the first 32."""

        discovery_count = min(int(self.randomized_svd_rank), int(range_matrix.shape[1]))
        holdout_count = int(range_matrix.shape[1]) - discovery_count
        if holdout_count < 1:
            return {
                "status": "unavailable_no_holdout_probes",
                "discovery_probe_count": discovery_count,
                "holdout_probe_count": 0,
                "relative_residual": float("nan"),
            }
        discovery = range_matrix[:, :discovery_count].double()
        holdout = range_matrix[:, discovery_count:].double()
        u, singular, _vh = torch.linalg.svd(discovery, full_matrices=False)
        rank, tolerance = _numerical_svd_rank(
            singular,
            row_count=int(discovery.shape[0]),
            column_count=int(discovery.shape[1]),
        )
        discovery_rms = float(discovery.square().mean().sqrt().cpu())
        if discovery_rms <= float(self.minimum_signal_rms_absolute):
            rank = 0
        holdout_norm = float(holdout.norm().cpu())
        holdout_rms = float(holdout.square().mean().sqrt().cpu())
        if holdout_rms <= float(self.minimum_signal_rms_absolute):
            return {
                "status": "completed_below_detection_holdout_response",
                "discovery_probe_count": discovery_count,
                "holdout_probe_count": holdout_count,
                "discovery_numerical_rank": int(rank),
                "discovery_rank_tolerance": float(tolerance),
                "discovery_response_rms": float(discovery_rms),
                "relative_residual": 0.0,
                "holdout_response_rms": float(holdout_rms),
                "below_detection_signal_rms_threshold": float(self.minimum_signal_rms_absolute),
                "role": "below_detection_response_diagnostic",
            }
        if rank == 0:
            residual = holdout
        else:
            basis = u[:, :rank]
            residual = holdout - basis @ (basis.T @ holdout)
        return {
            "status": "completed",
            "discovery_probe_count": discovery_count,
            "holdout_probe_count": holdout_count,
            "discovery_numerical_rank": int(rank),
            "discovery_rank_tolerance": float(tolerance),
            "discovery_response_rms": float(discovery_rms),
            "relative_residual": float((residual.norm() / holdout.norm()).cpu()),
            "holdout_response_rms": float(holdout_rms),
            "below_detection_signal_rms_threshold": float(self.minimum_signal_rms_absolute),
            "role": "advisory_range_capture_diagnostic_no_probe_fallback",
        }


    def model_vjp_rows(
        self,
        model: nn.Module,
        bases: Sequence[torch.Tensor],
        *,
        view: str,
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        model.eval().to(self.device)
        rows = [
            torch.zeros(int(base.shape[1]), int(self.raw_probe_columns.shape[0]), dtype=torch.float64)
            for base in bases
        ]
        diagnostics_by_block: list[dict[str, Any]] = [
            {
                "scope": "block",
                "view": str(view),
                "block_index": int(block_index),
                "batched_vjp_attempt_count": 0,
                "batched_vjp_success_count": 0,
                "sequential_fallback_count": 0,
                "sequential_fallback_reason_counts": {},
            }
            for block_index in range(self.depth)
        ]
        std_tensor = torch.tensor(
            self.normalization_std, device=self.device, dtype=torch.float32
        ).view(1, 1, int(self.all_images_cpu.shape[1]), 1, 1)
        capture_points = [f"block_{index:02d}_update" for index in range(self.depth)]
        for start_index in range(0, self.sample_count, self.microbatch_size):
            stop_index = min(self.sample_count, start_index + self.microbatch_size)
            images = (
                self.all_images_cpu[start_index:stop_index]
                .to(self.device)
                .float()
                .requires_grad_(True)
            )
            _logits, taps = forward_with_capture_and_interventions(
                model, images, capture_points=capture_points
            )
            batch_size = int(images.shape[0])
            for block_index in range(self.depth):
                update = taps[f"block_{block_index:02d}_update"]
                output = update[:, 0] if view == "cls" else update[:, 1:]
                basis_width = int(bases[block_index].shape[1])
                if basis_width == 0:
                    continue
                q = bases[block_index].T.reshape(
                    basis_width, *output.shape[1:]
                ).to(device=self.device, dtype=output.dtype)
                grad_outputs = q[:, None].expand(
                    int(q.shape[0]), batch_size, *q.shape[1:]
                ) / float(self.sample_count)
                block_diagnostics = diagnostics_by_block[block_index]
                try:
                    block_diagnostics["batched_vjp_attempt_count"] += 1
                    gradient = torch.autograd.grad(
                        output,
                        images,
                        grad_outputs=grad_outputs,
                        retain_graph=True,
                        create_graph=False,
                        is_grads_batched=True,
                    )[0]
                    block_diagnostics["batched_vjp_success_count"] += 1
                except (RuntimeError, TypeError) as error:
                    block_diagnostics["sequential_fallback_count"] += 1
                    reason = f"{type(error).__name__}:{str(error).splitlines()[0][:240]}"
                    reason_counts = block_diagnostics["sequential_fallback_reason_counts"]
                    reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                    gradients: list[torch.Tensor] = []
                    for row_index in range(int(q.shape[0])):
                        gradients.append(
                            torch.autograd.grad(
                                output,
                                images,
                                grad_outputs=grad_outputs[row_index],
                                retain_graph=True,
                                create_graph=False,
                            )[0]
                        )
                    gradient = torch.stack(gradients, dim=0)
                raw_gradient = gradient / std_tensor
                rows[block_index].add_(
                    raw_gradient.sum(dim=1).reshape(int(q.shape[0]), -1).double().cpu()
                )
                del q, grad_outputs, gradient, raw_gradient
            del taps, _logits, images
        for block_diagnostics in diagnostics_by_block:
            block_diagnostics["used_sequential_fallback"] = bool(
                block_diagnostics["sequential_fallback_count"]
            )
        return rows, diagnostics_by_block


    def attach_randomized_svd(
        self,
        model: nn.Module,
        cls_values: Sequence[dict[str, Any]],
        patch_values: Sequence[dict[str, Any]],
    ) -> None:
        for view, values in (("cls", cls_values), ("patch", patch_values)):
            bases = self.orthonormal_range_bases(values)
            vjp_rows, vjp_diagnostics_by_block = self.model_vjp_rows(
                model, bases, view=view
            )
            for block_index, (value, basis, rows) in enumerate(
                zip(values, bases, vjp_rows)
            ):
                holdout = self.range_holdout_diagnostic(value["range_matrix"])
                descriptor = _finalize_randomized_svd_descriptor(
                    value["range_matrix"],
                    rows,
                    target_rank=int(self.randomized_svd_rank),
                    range_basis=basis,
                    zero_signal_rms_tolerance=float(self.minimum_signal_rms_absolute),
                )
                residual = float(holdout.get("relative_residual", float("nan")))
                holdout_valid = bool(
                    str(holdout.get("status", "")).startswith("completed")
                    and math.isfinite(residual)
                    and residual <= float(self.range_holdout_relative_residual_maximum)
                )
                descriptor["range_basis_width"] = int(basis.shape[1])
                descriptor["range_holdout"] = {
                    **holdout,
                    "relative_residual_maximum": float(
                        self.range_holdout_relative_residual_maximum
                    ),
                    "quality_passed": holdout_valid,
                    "quality_status": (
                        "passed" if holdout_valid else "warning_range_capture_above_declared_residual"
                    ),
                    "role": "advisory_approximation_quality_diagnostic_not_a_validity_gate",
                }
                descriptor["vjp_execution"] = dict(
                    vjp_diagnostics_by_block[block_index]
                )
                if bool(descriptor.get("valid", False)) and not holdout_valid:
                    descriptor["approximation_quality_warning"] = (
                        "range_holdout_relative_residual_above_declared_maximum"
                    )
                value["randomized_svd"] = descriptor
                del value["range_matrix"]


    def model_runtime_cache_signature(self, model: nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict(keep_vars=True).items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(int(value) for value in tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tensor.device).encode("utf-8"))
            digest.update(str(int(tensor.data_ptr())).encode("utf-8"))
            digest.update(str(int(getattr(tensor, "_version", -1))).encode("utf-8"))
        return digest.hexdigest()


    def model_descriptor(self, model: nn.Module) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        cache_key = (
            id(model),
            self.model_runtime_cache_signature(model),
            self.cache_sample_key,
            int(self.probe_count),
            int(self.probe_seed),
            int(self.randomized_svd_rank),
            int(self.microbatch_size),
            tuple(float(value) for value in self.normalization_std),
            float(self.range_holdout_relative_residual_maximum),
            float(self.minimum_signal_rms_absolute),
        )
        if self.model_descriptor_cache is not None and cache_key in self.model_descriptor_cache:
            return self.model_descriptor_cache[cache_key]
        descriptor = self.model_jvp_sketch(model)
        self.attach_randomized_svd(model, descriptor[1], descriptor[2])
        if self.model_descriptor_cache is not None:
            self.model_descriptor_cache[cache_key] = descriptor
        return descriptor


    def gram_matrix(
        self,
        left_values: Sequence[dict[str, Any]],
        right_values: Sequence[dict[str, Any]],
        key: str = "gram",
    ) -> list[list[float]]:
        scores, _classes, _auxiliary = _jacobian_gram_similarity_matrix(
            left_values,
            right_values,
            key=key,
            minimum_signal_rms_absolute=float(self.minimum_signal_rms_absolute),
        )
        return scores


    def randomized_svd_matrices(
        self,
        left_values: Sequence[dict[str, Any]],
        right_values: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        spectrum: list[list[float]] = []
        input_subspace: list[list[float]] = []
        output_subspace: list[list[float]] = []
        operator: list[list[float]] = []
        validity: list[list[bool]] = []
        pair_classification: list[list[str]] = []
        for left_item in left_values:
            spectrum_row: list[float] = []
            input_row: list[float] = []
            output_row: list[float] = []
            operator_row: list[float] = []
            validity_row: list[bool] = []
            classification_row: list[str] = []
            left_svd = left_item["randomized_svd"]
            for right_item in right_values:
                right_svd = right_item["randomized_svd"]
                if not bool(left_svd.get("valid", False) and right_svd.get("valid", False)):
                    raise ValueError("DiR randomized Jacobian descriptor is invalid after computation")
                left_s = left_svd["singular_values"].double()
                right_s = right_svd["singular_values"].double()
                left_u = left_svd["output_subspace"].double()
                right_u = right_svd["output_subspace"].double()
                left_v = left_svd["input_subspace"].double()
                right_v = right_svd["input_subspace"].double()
                common_left = min(
                    int(left_svd["rank_used"]), int(left_u.shape[1]), int(left_v.shape[1])
                )
                common_right = min(
                    int(right_svd["rank_used"]), int(right_u.shape[1]), int(right_v.shape[1])
                )
                lu = left_u[:, :common_left]
                lv = left_v[:, :common_left]
                ls = left_s[:common_left]
                ru = right_u[:, :common_right]
                rv = right_v[:, :common_right]
                rs = right_s[:common_right]

                left_zero = common_left == 0
                right_zero = common_right == 0
                if left_zero and right_zero:
                    spectrum_score = float("nan")
                    input_score = float("nan")
                    output_score = float("nan")
                    operator_score = float("nan")
                    classification = "both_rank_zero_inconclusive"
                    pair_valid = False
                elif left_zero or right_zero:
                    spectrum_score = float("nan")
                    input_score = float("nan")
                    output_score = float("nan")
                    operator_score = float("nan")
                    classification = "one_rank_zero_inconclusive"
                    pair_valid = False
                else:
                    spectrum_score = _padded_singular_spectrum_cosine(left_s, right_s)
                    input_score, _input_class = _degenerate_aware_subspace_overlap(lv, rv)
                    output_score, _output_class = _degenerate_aware_subspace_overlap(lu, ru)
                    u_cross = lu.T @ ru
                    v_cross = lv.T @ rv
                    inner = (ls[:, None] * u_cross * rs[None, :] * v_cross).sum()
                    operator_score = float(
                        (
                            inner
                            / (
                                ls.square().sum().sqrt()
                                * rs.square().sum().sqrt()
                            ).clamp_min(1e-12)
                        ).cpu()
                    )
                    classification = (
                        "positive_rank"
                        if common_left == int(self.randomized_svd_rank)
                        and common_right == int(self.randomized_svd_rank)
                        else "valid_low_rank"
                    )
                    pair_valid = True
                validity_row.append(bool(pair_valid))
                spectrum_row.append(float(spectrum_score))
                input_row.append(float(input_score))
                output_row.append(float(output_score))
                operator_row.append(float(operator_score))
                classification_row.append(classification)
            spectrum.append(spectrum_row)
            input_subspace.append(input_row)
            output_subspace.append(output_row)
            operator.append(operator_row)
            validity.append(validity_row)
            pair_classification.append(classification_row)
        masks = {
            "singular_spectrum_cosine_12x12": validity,
            "input_singular_subspace_overlap_12x12": validity,
            "output_singular_subspace_overlap_12x12": validity,
            "low_rank_operator_cosine_12x12": validity,
        }
        return {
            "singular_spectrum_cosine_12x12": spectrum,
            "input_singular_subspace_overlap_12x12": input_subspace,
            "output_singular_subspace_overlap_12x12": output_subspace,
            "low_rank_operator_cosine_12x12": operator,
            "validity_masks": masks,
            "pair_rank_classification_12x12": pair_classification,
            "degenerate_similarity_contract": (
                "below_detection_or_rank_zero_pairs_have_no_numeric_alignment_score_and_are_invalid_for_similarity_statistics_"
                "positive_rank_uses_standard_similarity_low_rank_uses_all_numerically_valid_basis_vectors"
            ),
            "left_valid_by_block": [True for _value in left_values],
            "right_valid_by_block": [True for _value in right_values],
        }


    def split_half_for_view(
        self,
        left_values: Sequence[dict[str, Any]],
        right_values: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        first = np.asarray(self.gram_matrix(left_values, right_values, "first_gram"))
        second = np.asarray(self.gram_matrix(left_values, right_values, "second_gram"))
        correlation = _flat_rank_correlation(first, second)
        correlation_defined = bool(np.isfinite(correlation))
        diagonal_difference = float(abs(np.diag(first).mean() - np.diag(second).mean()))
        first_norm = 0.5 * (
            np.mean([item["first_rms"] for item in left_values])
            + np.mean([item["first_rms"] for item in right_values])
        )
        second_norm = 0.5 * (
            np.mean([item["second_rms"] for item in left_values])
            + np.mean([item["second_rms"] for item in right_values])
        )
        norm_relative_difference = abs(first_norm - second_norm) / max(
            1e-12, 0.5 * (first_norm + second_norm)
        )
        return {
            "spearman_matrix_correlation": correlation,
            "spearman_defined": correlation_defined,
            "diagonal_mean_difference": diagonal_difference,
            "rms_sensitivity_relative_difference": float(norm_relative_difference),
            "stable": bool(
                correlation_defined
                and correlation >= float(self.split_half_spearman_minimum)
                and diagonal_difference <= float(self.split_half_diagonal_difference_maximum)
                and norm_relative_difference <= float(self.split_half_norm_relative_difference_maximum)
            ),
        }


    def debiased_cka_validity_mask(self, classes: Sequence[Sequence[str]]) -> list[list[bool]]:
        return [
            [str(classification) == "nondegenerate_debiased_cka" for classification in row]
            for row in classes
        ]



def _build_input_jacobian_runtime(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    probe_count: int,
    probe_seed: int,
    normalization_std: Sequence[float],
    split_half_spearman_minimum: float,
    split_half_diagonal_difference_maximum: float,
    split_half_norm_relative_difference_maximum: float,
    randomized_svd_rank: int,
    range_holdout_relative_residual_maximum: float,
    minimum_signal_rms_absolute: float,
    microbatch_size: int,
    model_descriptor_cache: dict[tuple[Any, ...], Any] | None,
) -> _InputJacobianRuntime:
    depth = len(left_model.transformer_blocks)
    if depth != len(right_model.transformer_blocks):
        raise ValueError("DiR input JVP comparison requires equal depth")
    if int(probe_count) <= int(randomized_svd_rank):
        raise ValueError("DiR randomized Jacobian SVD requires probe_count > rank")
    if not 0.0 <= float(range_holdout_relative_residual_maximum) <= 1.0:
        raise ValueError(
            "DiR range holdout relative residual maximum must be in [0, 1]"
        )

    all_images_cpu = torch.cat(
        [images for images, _labels, _ids in batches], dim=0
    )
    all_sample_ids = torch.cat(
        [ids for _images, _labels, ids in batches], dim=0
    )
    sample_count = int(all_images_cpu.shape[0])
    microbatch_size = int(microbatch_size)
    if microbatch_size < 1 or microbatch_size > sample_count:
        raise ValueError(
            f"DiR JVP microbatch_size must be in [1, {sample_count}], "
            f"got {microbatch_size}"
        )

    raw_probe_columns = torch.stack(
        [
            _rademacher(
                (1, *all_images_cpu.shape[1:]),
                seed=int(probe_seed) + 100003 * probe_index,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).reshape(-1)
            for probe_index in range(int(probe_count))
        ],
        dim=1,
    ).double()
    cache_sample_key = hashlib.sha256(
        all_sample_ids.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    return _InputJacobianRuntime(
        sample_count=sample_count,
        half=int(probe_count) // 2,
        device=device,
        depth=depth,
        probe_count=probe_count,
        microbatch_size=microbatch_size,
        all_images_cpu=all_images_cpu,
        probe_seed=probe_seed,
        normalization_std=normalization_std,
        raw_probe_columns=raw_probe_columns,
        randomized_svd_rank=randomized_svd_rank,
        range_holdout_relative_residual_maximum=(
            range_holdout_relative_residual_maximum
        ),
        minimum_signal_rms_absolute=minimum_signal_rms_absolute,
        cache_sample_key=cache_sample_key,
        model_descriptor_cache=model_descriptor_cache,
        split_half_spearman_minimum=split_half_spearman_minimum,
        split_half_diagonal_difference_maximum=(
            split_half_diagonal_difference_maximum
        ),
        split_half_norm_relative_difference_maximum=(
            split_half_norm_relative_difference_maximum
        ),
    )


def _jacobian_view_payload(
    left_values: Sequence[dict[str, Any]],
    right_values: Sequence[dict[str, Any]],
    *,
    minimum_signal_rms_absolute: float,
) -> dict[str, Any]:
    debiased, degenerate_class, constant_projection_cosine = (
        _jacobian_gram_similarity_matrix(
            left_values,
            right_values,
            minimum_signal_rms_absolute=float(minimum_signal_rms_absolute),
        )
    )
    biased = [
        [
            centered_gram_cka(left_item["gram"], right_item["gram"])
            for right_item in right_values
        ]
        for left_item in left_values
    ]
    paired = pairwise_paired_output_metric_matrices(
        [item["projection_features"] for item in left_values],
        [item["projection_features"] for item in right_values],
    )
    return {
        "debiased": debiased,
        "degenerate_class": degenerate_class,
        "constant_projection_cosine": constant_projection_cosine,
        "biased": biased,
        "paired": paired,
    }


def _jacobian_family_validity(
    left_values: Sequence[dict[str, Any]],
    right_values: Sequence[dict[str, Any]],
    *,
    minimum_signal_rms_absolute: float,
    minimum_signal_rms_relative_to_median: float,
) -> dict[str, Any]:
    left_validity = combined_signal_variation_validity(
        [float(value["rms"]) for value in left_values],
        [gram_variation_strength(value["gram"]) for value in left_values],
        absolute_minimum=float(minimum_signal_rms_absolute),
        relative_to_median=float(minimum_signal_rms_relative_to_median),
    )
    right_validity = combined_signal_variation_validity(
        [float(value["rms"]) for value in right_values],
        [gram_variation_strength(value["gram"]) for value in right_values],
        absolute_minimum=float(minimum_signal_rms_absolute),
        relative_to_median=float(minimum_signal_rms_relative_to_median),
    )
    return {"left": left_validity, "right": right_validity}


def _rank32_jacobian_view_payload(
    svd_payload: Mapping[str, Any],
    left_values: Sequence[dict[str, Any]],
    right_values: Sequence[dict[str, Any]],
    *,
    primary_valid: bool,
    inconclusive_status: str,
) -> dict[str, Any]:
    return {
        **svd_payload,
        "measurement_status": "completed" if primary_valid else inconclusive_status,
        "left_leading_singular_values_by_block": [
            value["randomized_svd"]["singular_values"].tolist()
            for value in left_values
        ],
        "right_leading_singular_values_by_block": [
            value["randomized_svd"]["singular_values"].tolist()
            for value in right_values
        ],
        "left_rank_used_by_block": [
            int(value["randomized_svd"]["rank_used"]) for value in left_values
        ],
        "right_rank_used_by_block": [
            int(value["randomized_svd"]["rank_used"]) for value in right_values
        ],
        "left_descriptor_status_by_block": [
            str(value["randomized_svd"]["status"]) for value in left_values
        ],
        "right_descriptor_status_by_block": [
            str(value["randomized_svd"]["status"]) for value in right_values
        ],
        "left_holdout_relative_residual_by_block": [
            float(
                value["randomized_svd"]["range_holdout"].get(
                    "relative_residual", float("nan")
                )
            )
            for value in left_values
        ],
        "right_holdout_relative_residual_by_block": [
            float(
                value["randomized_svd"]["range_holdout"].get(
                    "relative_residual", float("nan")
                )
            )
            for value in right_values
        ],
        "left_holdout_quality_passed_by_block": [
            bool(
                value["randomized_svd"]["range_holdout"].get(
                    "quality_passed", False
                )
            )
            for value in left_values
        ],
        "right_holdout_quality_passed_by_block": [
            bool(
                value["randomized_svd"]["range_holdout"].get(
                    "quality_passed", False
                )
            )
            for value in right_values
        ],
        "left_vjp_execution_by_block": [
            dict(value["randomized_svd"].get("vjp_execution", {}))
            for value in left_values
        ],
        "right_vjp_execution_by_block": [
            dict(value["randomized_svd"].get("vjp_execution", {}))
            for value in right_values
        ],
    }


def _build_jacobian_alignment_result(
    *,
    measurement_status: str,
    primary_measurement_status: dict[str, str],
    paired_projection_quality_passed: bool,
    probe_count: int,
    probe_seed: int,
    microbatch_size: int,
    randomized_svd_rank: int,
    range_holdout_relative_residual_maximum: float,
    view_payloads: dict[str, dict[str, Any]],
    family_values: dict[str, tuple[Any, Any]],
    cls_svd: dict[str, Any],
    patch_svd: dict[str, Any],
    svd_primary_valid: Sequence[bool],
    split_half: dict[str, Any],
    validity_masks: dict[str, Any],
    family_validity: dict[str, Any],
) -> dict[str, Any]:
    block_full_view = view_payloads["block_update_full"]
    block_cls_view = view_payloads["block_update_cls"]
    block_patch_view = view_payloads["block_update_patch"]
    class_token_view = view_payloads["class_token"]
    left_block_full, right_block_full = family_values["block_update_full"]
    left_block_cls, right_block_cls = family_values["block_update_cls"]
    left_block_patch, right_block_patch = family_values["block_update_patch"]
    left_class_token, right_class_token = family_values["class_token"]

    block_full_matrix = block_full_view["debiased"]
    block_cls_matrix = block_cls_view["debiased"]
    block_patch_matrix = block_patch_view["debiased"]
    class_token_matrix = class_token_view["debiased"]
    block_full_degenerate_class = block_full_view["degenerate_class"]
    block_cls_degenerate_class = block_cls_view["degenerate_class"]
    block_patch_degenerate_class = block_patch_view["degenerate_class"]
    class_token_degenerate_class = class_token_view["degenerate_class"]
    block_full_constant_projection_cosine = block_full_view[
        "constant_projection_cosine"
    ]
    block_cls_constant_projection_cosine = block_cls_view[
        "constant_projection_cosine"
    ]
    block_patch_constant_projection_cosine = block_patch_view[
        "constant_projection_cosine"
    ]
    class_token_constant_projection_cosine = class_token_view[
        "constant_projection_cosine"
    ]
    biased_block_full_matrix = block_full_view["biased"]
    biased_block_cls_matrix = block_cls_view["biased"]
    biased_block_patch_matrix = block_patch_view["biased"]
    biased_class_token_matrix = class_token_view["biased"]
    block_full_paired_metrics = block_full_view["paired"]
    block_cls_paired_metrics = block_cls_view["paired"]
    block_patch_paired_metrics = block_patch_view["paired"]
    class_token_paired_metrics = class_token_view["paired"]
    return {
        "measurement_status": measurement_status,
        "primary_measurement_status": primary_measurement_status,
        "paired_projection_quality_passed": paired_projection_quality_passed,
        "probe_count": int(probe_count),
        "probe_seed": int(probe_seed),
        "microbatch_size": int(microbatch_size),
        "input_to_block_update_full_debiased_cka_12x12": block_full_matrix,
        "input_to_block_update_cls_debiased_cka_12x12": block_cls_matrix,
        "input_to_block_update_patch_debiased_cka_12x12": block_patch_matrix,
        "input_to_class_token_debiased_cka_12x12": class_token_matrix,
        "auxiliary_biased_cka": {
            "input_to_block_update_full_biased_cka_12x12": biased_block_full_matrix,
            "input_to_block_update_cls_biased_cka_12x12": biased_block_cls_matrix,
            "input_to_block_update_patch_biased_cka_12x12": biased_block_patch_matrix,
            "input_to_class_token_biased_cka_12x12": biased_class_token_matrix,
        },
        "auxiliary_constant_response_shared_projection_mean_cosine": {
            "input_to_block_update_full_12x12": block_full_constant_projection_cosine,
            "input_to_block_update_cls_12x12": block_cls_constant_projection_cosine,
            "input_to_block_update_patch_12x12": block_patch_constant_projection_cosine,
            "input_to_class_token_12x12": class_token_constant_projection_cosine,
            "contract": "reported_only_for_detected_sample_constant_Jacobian_responses_and_never_substituted_into_debiased_CKA",
        },
        "paired_shared_projection_metrics": {
            "block_update_full": block_full_paired_metrics,
            "block_update_cls": block_cls_paired_metrics,
            "block_update_patch": block_patch_paired_metrics,
            "class_token": class_token_paired_metrics,
        },
        "rank32_sample_mean_jacobian": {
            "block_update_cls": _rank32_jacobian_view_payload(
                cls_svd,
                left_block_cls,
                right_block_cls,
                primary_valid=svd_primary_valid[0],
                inconclusive_status=(
                    "inconclusive_no_detectable_rank32_cls_jacobian"
                ),
            ),
            "block_update_patch": _rank32_jacobian_view_payload(
                patch_svd,
                left_block_patch,
                right_block_patch,
                primary_valid=svd_primary_valid[1],
                inconclusive_status=(
                    "inconclusive_no_detectable_rank32_patch_jacobian"
                ),
            ),
            "rank": int(randomized_svd_rank),
            "range_probe_count": int(probe_count),
            "oversampling": int(probe_count) - int(randomized_svd_rank),
            "operator": (
                "Jacobian_of_fixed_sample_mean_block_update_with_respect_to_one_"
                "shared_raw_pixel_perturbation"
            ),
            "algorithm": (
                "standard_randomized_SVD_Y_equals_JOmega_Q_then_B_equals_Q_transpose_"
                "J_via_batched_VJP"
            ),
            "explicit_full_jacobian_materialized": False,
            "range_holdout_relative_residual_maximum": float(
                range_holdout_relative_residual_maximum
            ),
            "range_holdout_contract": (
                "first_32_probes_discover_range_last_8_report_approximation_quality_"
                "without_invalidating_the_fixed_rank_descriptor_final_descriptor_uses_"
                "all_40"
            ),
            "numerical_rank_contract": (
                "uncentered_range_response_RMS_at_or_below_minimum_signal_threshold_is_"
                "below_detection_and_not_evidence_of_a_true_zero_operator; detected_"
                "responses_with_zero_numerical_rank_are_recorded_separately; otherwise_"
                "only_singular_values_above_numerical_tolerance_enter_spectrum_and_"
                "subspaces"
            ),
        },
        "split_half": split_half,
        "left_block_rms_jvp_sensitivity": [float(value["rms"]) for value in left_block_full],
        "right_block_rms_jvp_sensitivity": [float(value["rms"]) for value in right_block_full],
        "left_class_token_rms_jvp_sensitivity": [float(value["rms"]) for value in left_class_token],
        "right_class_token_rms_jvp_sensitivity": [float(value["rms"]) for value in right_class_token],
        "validity_masks": validity_masks,
        "low_signal": family_validity,
        "jacobian_response_degenerate_classification": {
            "block_update_full": block_full_degenerate_class,
            "block_update_cls": block_cls_degenerate_class,
            "block_update_patch": block_patch_degenerate_class,
            "class_token": class_token_degenerate_class,
        },
        "jacobian_degenerate_similarity_contract": (
            "U_centered_CKA_only_when_sample_varying; below_detection_and_detected_constant_responses_are_inconclusive_for_primary_CKA; constant_response_shared_projection_mean_direction_cosine_is_auxiliary_only"
        ),
        "primary_metrics": [
            "input_to_block_update_cls_debiased_cka_12x12",
            "input_to_block_update_patch_debiased_cka_12x12",
        ],
        "full_token_role": "auxiliary_only",
        "token_stage_contract": "native_residual_block_update_JVP_space_before_next_block_pre_norm",
        "probe_direction_contract": "one_shared_unit_l2_raw_pixel_Rademacher_direction_broadcast_to_all_samples_and_models_then_divide_by_channel_std",
        "paired_metric_contract": "shared_output_projection_is_auxiliary_only_and_never_substituted_into_primary_debiased_CKA",
        "jacobian_svd_contract": "rank32_standard_randomized_SVD_of_fixed_sample_mean_CLS_and_patch_block_update_Jacobians_using_40_range_probes_batched_VJP_uncentered_RMS_detection_threshold_and_advisory_holdout_quality",
        "cka_contract": "U_centered_debiased_primary_biased_auxiliary",
        "absolute_norm_interpretation": "rms_jvp_sensitivity_not_unscaled_frobenius_norm",
        "memory_policy": "full_Jacobian_never_materialized_exact_sample_grams_use_one_transfer_per_view_and_randomized_SVD_uses_bounded_JVP_VJP",
    }


def jacobian_input_response_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    probe_count: int,
    probe_seed: int,
    normalization_std: Sequence[float],
    split_half_spearman_minimum: float = 0.80,
    split_half_diagonal_difference_maximum: float = 0.05,
    split_half_norm_relative_difference_maximum: float = 0.15,
    randomized_svd_rank: int = 32,
    range_holdout_relative_residual_maximum: float = 0.50,
    minimum_signal_rms_absolute: float = 1e-8,
    minimum_signal_rms_relative_to_median: float = 0.05,
    microbatch_size: int = 8,
    model_descriptor_cache: dict[tuple[Any, ...], Any] | None = None,
) -> dict[str, Any]:
    """Measure JVP response alignment and a true rank-bounded Jacobian SVD.

    JVP CKA retains the sample-by-sample response contract. The spectrum and
    singular subspaces are computed separately from a standard randomized SVD
    of the Jacobian of the fixed-sample mean block-update function with respect
    to one shared raw-pixel perturbation. Forty Rademacher range probes and
    VJP rows construct ``B=QᵀJ`` without materializing the full Jacobian.
    """

    runtime = _build_input_jacobian_runtime(
        left_model,
        right_model,
        batches,
        device=device,
        probe_count=probe_count,
        probe_seed=probe_seed,
        normalization_std=normalization_std,
        split_half_spearman_minimum=split_half_spearman_minimum,
        split_half_diagonal_difference_maximum=split_half_diagonal_difference_maximum,
        split_half_norm_relative_difference_maximum=(
            split_half_norm_relative_difference_maximum
        ),
        randomized_svd_rank=randomized_svd_rank,
        range_holdout_relative_residual_maximum=range_holdout_relative_residual_maximum,
        minimum_signal_rms_absolute=minimum_signal_rms_absolute,
        microbatch_size=microbatch_size,
        model_descriptor_cache=model_descriptor_cache,
    )
    depth = runtime.depth
    sample_count = runtime.sample_count
    microbatch_size = runtime.microbatch_size

    (
        left_block_full,
        left_block_cls,
        left_block_patch,
        left_class_token,
    ) = runtime.model_descriptor(left_model)
    (
        right_block_full,
        right_block_cls,
        right_block_patch,
        right_class_token,
    ) = runtime.model_descriptor(right_model)

    family_values = {
        "block_update_full": (left_block_full, right_block_full),
        "block_update_cls": (left_block_cls, right_block_cls),
        "block_update_patch": (left_block_patch, right_block_patch),
        "class_token": (left_class_token, right_class_token),
    }
    view_payloads = {
        family_name: _jacobian_view_payload(
            left_values,
            right_values,
            minimum_signal_rms_absolute=minimum_signal_rms_absolute,
        )
        for family_name, (left_values, right_values) in family_values.items()
    }
    block_full_view = view_payloads["block_update_full"]
    block_cls_view = view_payloads["block_update_cls"]
    block_patch_view = view_payloads["block_update_patch"]
    class_token_view = view_payloads["class_token"]

    block_full_matrix = block_full_view["debiased"]
    block_cls_matrix = block_cls_view["debiased"]
    block_patch_matrix = block_patch_view["debiased"]
    class_token_matrix = class_token_view["debiased"]
    block_full_degenerate_class = block_full_view["degenerate_class"]
    block_cls_degenerate_class = block_cls_view["degenerate_class"]
    block_patch_degenerate_class = block_patch_view["degenerate_class"]
    class_token_degenerate_class = class_token_view["degenerate_class"]
    block_full_constant_projection_cosine = block_full_view[
        "constant_projection_cosine"
    ]
    block_cls_constant_projection_cosine = block_cls_view[
        "constant_projection_cosine"
    ]
    block_patch_constant_projection_cosine = block_patch_view[
        "constant_projection_cosine"
    ]
    class_token_constant_projection_cosine = class_token_view[
        "constant_projection_cosine"
    ]
    biased_block_full_matrix = block_full_view["biased"]
    biased_block_cls_matrix = block_cls_view["biased"]
    biased_block_patch_matrix = block_patch_view["biased"]
    biased_class_token_matrix = class_token_view["biased"]
    block_full_paired_metrics = block_full_view["paired"]
    block_cls_paired_metrics = block_cls_view["paired"]
    block_patch_paired_metrics = block_patch_view["paired"]
    class_token_paired_metrics = class_token_view["paired"]

    split_half_by_view = {
        "block_update_cls": runtime.split_half_for_view(
            left_block_cls, right_block_cls
        ),
        "block_update_patch": runtime.split_half_for_view(
            left_block_patch, right_block_patch
        ),
    }
    split_half = {
        "stable": bool(all(value["stable"] for value in split_half_by_view.values())),
        "advisory_only": True,
        "probe_count_is_fixed": True,
        "by_view": split_half_by_view,
        "thresholds": {
            "spearman_minimum": float(split_half_spearman_minimum),
            "diagonal_difference_maximum": float(
                split_half_diagonal_difference_maximum
            ),
            "norm_relative_difference_maximum": float(
                split_half_norm_relative_difference_maximum
            ),
        },
    }

    family_validity = {
        family_name: _jacobian_family_validity(
            left_values,
            right_values,
            minimum_signal_rms_absolute=minimum_signal_rms_absolute,
            minimum_signal_rms_relative_to_median=(
                minimum_signal_rms_relative_to_median
            ),
        )
        for family_name, (left_values, right_values) in family_values.items()
    }
    validity_masks = {
        "input_to_block_update_full_debiased_cka_12x12": (
            runtime.debiased_cka_validity_mask(block_full_degenerate_class)
        ),
        "input_to_block_update_cls_debiased_cka_12x12": (
            runtime.debiased_cka_validity_mask(block_cls_degenerate_class)
        ),
        "input_to_block_update_patch_debiased_cka_12x12": (
            runtime.debiased_cka_validity_mask(block_patch_degenerate_class)
        ),
        "input_to_class_token_debiased_cka_12x12": (
            runtime.debiased_cka_validity_mask(class_token_degenerate_class)
        ),
    }
    for family_payload in family_validity.values():
        family_payload["role"] = (
            "diagnostic_signal_strength_and_variation; absolute_below_detection_"
            "responses_are_inconclusive_for_Jacobian_similarity"
        )

    jvp_primary_valid = [
        bool(np.diag(np.asarray(validity_masks[key], dtype=bool)).any())
        for key in (
            "input_to_block_update_cls_debiased_cka_12x12",
            "input_to_block_update_patch_debiased_cka_12x12",
        )
    ]
    cls_svd = runtime.randomized_svd_matrices(left_block_cls, right_block_cls)
    patch_svd = runtime.randomized_svd_matrices(left_block_patch, right_block_patch)
    svd_primary_valid = []
    for payload in (cls_svd, patch_svd):
        mask = np.asarray(
            payload["validity_masks"]["low_rank_operator_cosine_12x12"],
            dtype=bool,
        )
        svd_primary_valid.append(bool(np.diag(mask).any()))

    paired_projection_payloads = tuple(
        view_payloads[name]["paired"]
        for name in (
            "block_update_full",
            "block_update_cls",
            "block_update_patch",
            "class_token",
        )
    )
    paired_projection_quality_passed = bool(
        all(
            bool(payload.get("quality_passed", False))
            for payload in paired_projection_payloads
        )
    )
    if not paired_projection_quality_passed:
        raise ValueError(
            "DiR Jacobian paired projection diagnostics contain non-finite outputs"
        )

    primary_measurement_status = {
        "input_to_block_update_cls_debiased_cka_12x12": (
            "valid" if jvp_primary_valid[0] else "inconclusive_no_detectable_cls_JVP"
        ),
        "input_to_block_update_patch_debiased_cka_12x12": (
            "valid"
            if jvp_primary_valid[1]
            else "inconclusive_no_detectable_patch_JVP"
        ),
        "rank32_sample_mean_jacobian.block_update_cls": (
            "valid"
            if svd_primary_valid[0]
            else "inconclusive_no_detectable_rank32_cls_jacobian"
        ),
        "rank32_sample_mean_jacobian.block_update_patch": (
            "valid"
            if svd_primary_valid[1]
            else "inconclusive_no_detectable_rank32_patch_jacobian"
        ),
    }
    measurement_status = _measurement_status_from_primary_validity(
        [*jvp_primary_valid, *svd_primary_valid],
        no_valid_status="inconclusive_no_valid_primary_jacobian_view",
    )
    return _build_jacobian_alignment_result(
        measurement_status=measurement_status,
        primary_measurement_status=primary_measurement_status,
        paired_projection_quality_passed=paired_projection_quality_passed,
        probe_count=probe_count,
        probe_seed=probe_seed,
        microbatch_size=microbatch_size,
        randomized_svd_rank=randomized_svd_rank,
        range_holdout_relative_residual_maximum=(
            range_holdout_relative_residual_maximum
        ),
        view_payloads=view_payloads,
        family_values=family_values,
        cls_svd=cls_svd,
        patch_svd=patch_svd,
        svd_primary_valid=svd_primary_valid,
        split_half=split_half,
        validity_masks=validity_masks,
        family_validity=family_validity,
    )
