"""Supplementary attention-transport and spectral correspondence measurements."""

from __future__ import annotations


# Attention transport
from functools import lru_cache
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import forward_with_capture_and_interventions
from ..measurements.representation_similarity import pairwise_cka_matrix

def _exact_assignment_mean(matrix: np.ndarray) -> float:
    """Maximum finite one-to-one assignment mean or NaN if none exists."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("attention head assignment requires one square matrix")
    size = int(values.shape[0])

    @lru_cache(maxsize=None)
    def solve(row: int, used: int) -> float:
        if row == size:
            return 0.0
        best = -float("inf")
        for column in range(size):
            if used & (1 << column):
                continue
            edge = float(values[row, column])
            if not math.isfinite(edge):
                continue
            suffix = solve(row + 1, used | (1 << column))
            if not math.isfinite(suffix):
                continue
            best = max(best, edge + suffix)
        return best

    total = solve(0, 0)
    return float(total / max(1, size)) if math.isfinite(total) else float("nan")


def _collect_attention(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]]]:
    depth = len(model.transformer_blocks)
    aggregate_heads: list[list[torch.Tensor]] = [[] for _ in range(depth)]
    post_o: list[list[torch.Tensor]] = [[] for _ in range(depth)]
    head_features: list[list[list[torch.Tensor]]] = []
    model.eval().to(device)
    with torch.no_grad():
        for images_cpu, _labels, _ids in batches:
            points = [
                *[f"block_{i:02d}_value_weighted_heads" for i in range(depth)],
                *[f"block_{i:02d}_attention_residual_contribution" for i in range(depth)],
            ]
            _logits, taps = forward_with_capture_and_interventions(
                model, images_cpu.to(device), capture_points=points
            )
            if not head_features:
                head_count = int(taps["block_00_value_weighted_heads"].shape[1])
                head_features = [[[] for _ in range(head_count)] for _ in range(depth)]
            for index in range(depth):
                heads = taps[f"block_{index:02d}_value_weighted_heads"]
                aggregate_heads[index].append(heads.transpose(1, 2).reshape(heads.shape[0], heads.shape[2], -1).cpu())
                post_o[index].append(taps[f"block_{index:02d}_attention_residual_contribution"].cpu())
                for head in range(int(heads.shape[1])):
                    value = heads[:, head]
                    feature = torch.cat([value[:, 0], value[:, 1:].mean(dim=1)], dim=1)
                    head_features[index][head].append(feature.cpu())
    return (
        [torch.cat(values, dim=0) for values in aggregate_heads],
        [torch.cat(values, dim=0) for values in post_o],
        [[torch.cat(parts, dim=0) for parts in block] for block in head_features],
    )


def attention_transport_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    left_aggregate, left_post_o, left_heads = _collect_attention(left_model, batches, device=device)
    right_aggregate, right_post_o, right_heads = _collect_attention(right_model, batches, device=device)
    hungarian: list[list[float]] = []
    raw_same_head: list[list[float]] = []
    hungarian_validity: list[list[bool]] = []
    same_head_validity: list[list[bool]] = []
    finite_head_pair_counts: list[list[int]] = []
    for left_block in left_heads:
        hungarian_row: list[float] = []
        same_row: list[float] = []
        hungarian_valid_row: list[bool] = []
        same_valid_row: list[bool] = []
        finite_count_row: list[int] = []
        for right_block in right_heads:
            matrix = np.asarray(pairwise_cka_matrix(left_block, right_block), dtype=np.float64)
            assignment = _exact_assignment_mean(matrix)
            diagonal = np.diag(matrix)
            same_value = float(diagonal.mean()) if bool(np.isfinite(diagonal).all()) else float("nan")
            hungarian_row.append(assignment)
            same_row.append(same_value)
            hungarian_valid_row.append(bool(math.isfinite(assignment)))
            same_valid_row.append(bool(math.isfinite(same_value)))
            finite_count_row.append(int(np.isfinite(matrix).sum()))
        hungarian.append(hungarian_row)
        raw_same_head.append(same_row)
        hungarian_validity.append(hungarian_valid_row)
        same_head_validity.append(same_valid_row)
        finite_head_pair_counts.append(finite_count_row)
    aggregate = np.asarray(pairwise_cka_matrix(left_aggregate, right_aggregate), dtype=np.float64)
    post_o = np.asarray(pairwise_cka_matrix(left_post_o, right_post_o), dtype=np.float64)
    return {
        "value_weighted_transport_cka_12x12": aggregate.tolist(),
        "post_o_residual_contribution_cka_12x12": post_o.tolist(),
        "hungarian_head_matching_cka_12x12": hungarian,
        "raw_same_head_index_cka_12x12": raw_same_head,
        "validity_masks": {
            "value_weighted_transport_cka_12x12": np.isfinite(aggregate).tolist(),
            "post_o_residual_contribution_cka_12x12": np.isfinite(post_o).tolist(),
            "hungarian_head_matching_cka_12x12": hungarian_validity,
            "raw_same_head_index_cka_12x12": same_head_validity,
        },
        "finite_head_pair_count_12x12": finite_head_pair_counts,
        "head_matching_contract": (
            "Hungarian_score_requires_one_complete_finite_one_to_one_head_assignment;_otherwise_NaN_inconclusive."
        ),
        "same_head_contract": (
            "same-index_head_mean_requires_every_same-index_head_pair_to_be_finite;_otherwise_NaN_inconclusive"
        ),
    }


# Spectral perturbations
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..interventions import forward_with_capture_and_interventions
from ..model.dictionary_operator import iter_dictionary_layers
from ..measurements.representation_similarity import _feature_view, pairwise_cka_matrix

def _spectral_radial_masks(
    height: int,
    width: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return conjugate-symmetric fftshift radial bands for even or odd images."""

    yy, xx = torch.meshgrid(
        torch.arange(int(height), device=device),
        torch.arange(int(width), device=device),
        indexing="ij",
    )
    # After fftshift, the DC bin is exactly floor(N/2), including 32x32 CIFAR.
    center_y = int(height) // 2
    center_x = int(width) // 2
    radius = torch.sqrt(
        (yy - center_y).float().square() + (xx - center_x).float().square()
    )
    low = radius <= 6.0
    middle = (radius > 6.0) & (radius <= 12.0)
    high = radius > 12.0
    return low, middle, high


def _enforce_shifted_hermitian_symmetry(spectrum: torch.Tensor) -> torch.Tensor:
    """Project an fftshifted 2-D spectrum onto the real-image Hermitian subspace."""

    height, width = spectrum.shape[-2:]
    center_y = int(height) // 2
    center_x = int(width) // 2
    partner_y = (2 * center_y - torch.arange(int(height), device=spectrum.device)) % int(height)
    partner_x = (2 * center_x - torch.arange(int(width), device=spectrum.device)) % int(width)
    conjugate_partner = spectrum.index_select(-2, partner_y).index_select(-1, partner_x).conj()
    return 0.5 * (spectrum + conjugate_partner)


def _shifted_self_conjugate_mask(
    height: int, width: int, *, device: torch.device
) -> torch.Tensor:
    """Return fftshifted bins that are their own Hermitian partners."""

    center_y = int(height) // 2
    center_x = int(width) // 2
    y = torch.arange(int(height), device=device)
    x = torch.arange(int(width), device=device)
    partner_y = (2 * center_y - y) % int(height)
    partner_x = (2 * center_x - x) % int(width)
    return (y == partner_y)[:, None] & (x == partner_x)[None, :]


def _spectral_variant(raw: torch.Tensor, variant: str) -> torch.Tensor:
    spectrum = torch.fft.fftshift(torch.fft.fft2(raw.float(), norm="ortho"), dim=(-2, -1))
    height, width = raw.shape[-2:]
    low, middle, high = _spectral_radial_masks(
        int(height), int(width), device=raw.device
    )
    if variant == "low_remove":
        changed = spectrum * (~low)
    elif variant == "high_remove":
        changed = spectrum * (~high)
    elif variant == "band_remove":
        changed = spectrum * (~middle)
    elif variant == "amplitude_flatten":
        amplitude = spectrum.abs()
        changed = amplitude.mean(dim=(-2, -1), keepdim=True) * torch.exp(1j * torch.angle(spectrum))
    elif variant == "phase_attenuate":
        changed = spectrum.abs() * torch.exp(0.5j * torch.angle(spectrum))
        self_conjugate = _shifted_self_conjugate_mask(
            int(height), int(width), device=spectrum.device
        )
        changed = torch.where(self_conjugate, spectrum, changed)
    else:
        raise ValueError(variant)
    # Frequency-domain perturbations of a real image must remain Hermitian.
    # This is especially important for phase attenuation at self-conjugate
    # Nyquist bins, where naive phase scaling can create an imaginary image.
    changed = _enforce_shifted_hermitian_symmetry(changed)
    reconstructed = torch.fft.ifft2(
        torch.fft.ifftshift(changed, dim=(-2, -1)), norm="ortho"
    )
    if float(reconstructed.imag.detach().abs().max()) > 1e-5:
        raise RuntimeError("spectral perturbation violated the real-image Hermitian contract")
    return reconstructed.real.clamp(0, 1)


def _spectral_energy_audit(raw: torch.Tensor, changed_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    clean_spectrum = torch.fft.fftshift(
        torch.fft.fft2(raw.float(), norm="ortho"), dim=(-2, -1)
    )
    changed_spectrum = torch.fft.fftshift(
        torch.fft.fft2(changed_raw.float(), norm="ortho"), dim=(-2, -1)
    )
    clean_energy = clean_spectrum.abs().square()
    changed_energy = changed_spectrum.abs().square()
    low, middle, high = _spectral_radial_masks(
        int(raw.shape[-2]), int(raw.shape[-1]), device=raw.device
    )

    def total(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        selected = value if mask is None else value * mask.to(value.dtype)
        return selected.sum(dim=(1, 2, 3))

    clean_total = total(clean_energy)
    changed_total = total(changed_energy)
    output: dict[str, torch.Tensor] = {
        "fourier_total_energy_relative_change": (
            (changed_total - clean_total) / clean_total.clamp_min(1e-12)
        ),
        "fourier_total_energy_retention": changed_total / clean_total.clamp_min(1e-12),
    }
    for label, mask in (("low", low), ("middle", middle), ("high", high)):
        clean_band = total(clean_energy, mask)
        changed_band = total(changed_energy, mask)
        output[f"fourier_{label}_energy_retention"] = (
            changed_band / clean_band.clamp_min(1e-12)
        )
        output[f"fourier_{label}_energy_fraction_change"] = (
            (changed_band - clean_band) / clean_total.clamp_min(1e-12)
        )
    return output


def _begin_dictionary_atom_response(model: nn.Module) -> list[tuple[str, nn.Module]]:
    layers = list(iter_dictionary_layers(model))
    begin_functions = [
        getattr(layer, "begin_activation_contribution_measurement_", None)
        for _name, layer in layers
    ]
    if any(not callable(begin) for begin in begin_functions):
        return []
    for begin in begin_functions:
        begin(threshold=0.0, collect_sample_hard_counts=False)
    return layers


def _end_dictionary_atom_response(
    layers: Sequence[tuple[str, nn.Module]],
) -> dict[str, tuple[torch.Tensor, int]]:
    output: dict[str, tuple[torch.Tensor, int]] = {}
    for name, layer in layers:
        finish = getattr(layer, "end_activation_contribution_measurement_", None)
        state = finish() if callable(finish) else None
        if not isinstance(state, dict):
            continue
        mass_sum = state.get("mass_sum")
        sample_count = int(state.get("sample_count", 0) or 0)
        if isinstance(mass_sum, torch.Tensor) and sample_count > 0:
            output[str(name)] = (mass_sum.detach().float().cpu(), sample_count)
    return output


def _accumulate_atom_response(
    accumulator: dict[str, tuple[torch.Tensor, int]],
    batch_state: Mapping[str, tuple[torch.Tensor, int]],
) -> None:
    for name, (mass_sum, sample_count) in batch_state.items():
        if name not in accumulator:
            accumulator[name] = (mass_sum.clone(), int(sample_count))
        else:
            previous_mass, previous_count = accumulator[name]
            accumulator[name] = (
                previous_mass + mass_sum,
                int(previous_count) + int(sample_count),
            )


def _atom_response_by_block(
    clean_state: Mapping[str, tuple[torch.Tensor, int]],
    changed_state: Mapping[str, tuple[torch.Tensor, int]],
    *,
    depth: int,
) -> tuple[list[torch.Tensor] | None, list[float] | None]:
    common_names = sorted(set(clean_state) & set(changed_state))
    if not common_names:
        return None, None
    by_block: list[list[torch.Tensor]] = [[] for _ in range(int(depth))]
    clean_by_block: list[list[torch.Tensor]] = [[] for _ in range(int(depth))]
    for name in common_names:
        match = re.search(r"transformer_blocks[._](\d+)", str(name))
        if match is None:
            continue
        block_index = int(match.group(1))
        if not 0 <= block_index < int(depth):
            continue
        clean_mass, clean_count = clean_state[name]
        changed_mass, changed_count = changed_state[name]
        if tuple(clean_mass.shape) != tuple(changed_mass.shape):
            continue
        clean_mean = clean_mass / float(max(1, int(clean_count)))
        changed_mean = changed_mass / float(max(1, int(changed_count)))
        clean_by_block[block_index].append(clean_mean)
        by_block[block_index].append(changed_mean - clean_mean)
    if any(not values for values in by_block):
        return None, None
    deltas = [torch.cat(values).float() for values in by_block]
    clean_vectors = [torch.cat(values).float() for values in clean_by_block]
    relative = [
        float(delta.norm() / clean.norm().clamp_min(1e-12))
        for delta, clean in zip(deltas, clean_vectors)
    ]
    return deltas, relative


def _pairwise_signed_cosine_vectors(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> list[list[float]]:
    output: list[list[float]] = []
    for left_value in left:
        row: list[float] = []
        left_flat = left_value.reshape(-1).double()
        for right_value in right:
            right_flat = right_value.reshape(-1).double()
            if int(left_flat.numel()) != int(right_flat.numel()):
                row.append(float("nan"))
                continue
            denominator = left_flat.norm() * right_flat.norm()
            if float(denominator) <= 1e-12:
                row.append(float("nan"))
            else:
                row.append(float(torch.dot(left_flat, right_flat) / denominator))
        output.append(row)
    return output


def spectral_perturbation_alignment(
    left_model: nn.Module,
    right_model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    mean: Sequence[float],
    std: Sequence[float],
) -> dict[str, Any]:
    depth = len(left_model.transformer_blocks)
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def signatures(
        model: nn.Module, variant: str
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float], list[torch.Tensor] | None, list[float] | None]:
        model.eval().to(device)
        rows: list[list[torch.Tensor]] = [[] for _ in range(depth)]
        attention_rows: list[list[torch.Tensor]] = [[] for _ in range(depth)]
        pixel_rms: list[torch.Tensor] = []
        pixel_l2: list[torch.Tensor] = []
        retention: list[torch.Tensor] = []
        energy_rows: dict[str, list[torch.Tensor]] = {}
        clean_atom_accumulator: dict[str, tuple[torch.Tensor, int]] = {}
        changed_atom_accumulator: dict[str, tuple[torch.Tensor, int]] = {}
        with torch.no_grad():
            for images_cpu, _labels, _ids in batches:
                images = images_cpu.to(device)
                raw = (images * std_tensor + mean_tensor).clamp(0, 1)
                changed_raw = _spectral_variant(raw, variant)
                changed = (changed_raw - mean_tensor) / std_tensor
                points = [
                    "pre_classifier",
                    *[f"block_{i:02d}_update" for i in range(depth)],
                    *[f"block_{i:02d}_attention_residual_contribution" for i in range(depth)],
                ]

                clean_layers = _begin_dictionary_atom_response(model)
                try:
                    clean_logits, clean = forward_with_capture_and_interventions(
                        model, images, capture_points=points
                    )
                finally:
                    clean_atom_state = _end_dictionary_atom_response(clean_layers)
                changed_layers = _begin_dictionary_atom_response(model)
                try:
                    changed_logits, changed_taps = forward_with_capture_and_interventions(
                        model, changed, capture_points=points
                    )
                finally:
                    changed_atom_state = _end_dictionary_atom_response(changed_layers)
                _accumulate_atom_response(clean_atom_accumulator, clean_atom_state)
                _accumulate_atom_response(changed_atom_accumulator, changed_atom_state)

                difference = changed_raw - raw
                flat_difference = difference.reshape(raw.shape[0], -1)
                pixel_rms.append(flat_difference.square().mean(dim=1).sqrt().cpu())
                pixel_l2.append(flat_difference.norm(dim=1).cpu())
                retention.append((changed_logits.argmax(1) == clean_logits.argmax(1)).float().cpu())
                energy = _spectral_energy_audit(raw, changed_raw)
                for key, value in energy.items():
                    energy_rows.setdefault(key, []).append(value.detach().cpu())

                final_delta = (
                    changed_taps["pre_classifier"] - clean["pre_classifier"]
                ).reshape(raw.shape[0], -1).square().mean(1).sqrt()
                for index in range(depth):
                    delta = changed_taps[f"block_{index:02d}_update"] - clean[f"block_{index:02d}_update"]
                    block_delta = delta.reshape(raw.shape[0], -1).square().mean(1).sqrt()
                    rows[index].append(torch.stack([block_delta, final_delta], dim=1).cpu())

                    attention_delta = (
                        changed_taps[f"block_{index:02d}_attention_residual_contribution"]
                        - clean[f"block_{index:02d}_attention_residual_contribution"]
                    )
                    attention_rows[index].append(
                        torch.stack(
                            [
                                _feature_view(attention_delta, "full_token").square().mean(1).sqrt(),
                                _feature_view(attention_delta, "cls").square().mean(1).sqrt(),
                                _feature_view(attention_delta, "patch").square().mean(1).sqrt(),
                            ],
                            dim=1,
                        ).cpu()
                    )
        atom_delta, atom_relative = _atom_response_by_block(
            clean_atom_accumulator, changed_atom_accumulator, depth=depth
        )
        audit = {
            "pixel_rms": float(torch.cat(pixel_rms).mean()),
            "pixel_l2": float(torch.cat(pixel_l2).mean()),
            "prediction_retention": float(torch.cat(retention).mean()),
            **{
                key: float(torch.cat(values).mean())
                for key, values in sorted(energy_rows.items())
            },
            "atom_response_available": bool(atom_delta is not None),
        }
        if atom_relative is not None:
            audit["atom_contribution_mass_relative_change_mean"] = float(np.mean(atom_relative))
        return (
            [torch.cat(values) for values in rows],
            [torch.cat(values) for values in attention_rows],
            audit,
            atom_delta,
            atom_relative,
        )

    output: dict[str, Any] = {}
    for variant in ("low_remove", "high_remove", "band_remove", "amplitude_flatten", "phase_attenuate"):
        left, left_attention, left_audit, left_atom, left_atom_relative = signatures(left_model, variant)
        right, right_attention, right_audit, right_atom, right_atom_relative = signatures(right_model, variant)
        record: dict[str, Any] = {
            "response_cka_12x12": pairwise_cka_matrix(left, right),
            "attention_response_cka_12x12": pairwise_cka_matrix(left_attention, right_attention),
            "left_audit": left_audit,
            "right_audit": right_audit,
            "fourier_contract": (
                "fftshift_DC_center_uses_floor_half_indices_and_energy_is_recomputed_after_real_image_reconstruction_and_clipping"
            ),
        }
        if left_atom is not None and right_atom is not None:
            atom_matrix = _pairwise_signed_cosine_vectors(left_atom, right_atom)
            record["atom_contribution_response_signed_cosine_12x12"] = atom_matrix
            record["atom_response_validity_mask_12x12"] = np.isfinite(
                np.asarray(atom_matrix, dtype=np.float64)
            ).tolist()
            record["left_atom_contribution_mass_relative_change_by_block"] = left_atom_relative
            record["right_atom_contribution_mass_relative_change_by_block"] = right_atom_relative
            record["atom_response_contract"] = (
                "post-route_per-atom_contribution-mass_change_aggregated_over_the_fixed_spectral_sample_set_and_compared_in_block-local_dictionary_coordinates"
            )
        else:
            record["atom_contribution_response_signed_cosine_12x12"] = None
            record["atom_response_status"] = "not_applicable_no_dictionary_atoms_on_one_or_both_models"
        output[variant] = record
    output["spectral_measurement_contract"] = (
        "pixel_change_Fourier_energy_prediction_retention_block_response_attention_response_and_dictionary_atom_response_when_available"
    )
    return output
