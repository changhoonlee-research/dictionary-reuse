"""Deterministic weak corruptions and patching-validity audits."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..interventions import forward_with_capture_and_interventions
from .representation_similarity import _feature_view

def _gaussian_kernel(
    kernel_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("DiR blur kernel size must be a positive odd integer")
    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel.to(dtype=dtype).reshape(1, 1, kernel_size, kernel_size)


def _sample_hash_index(sample_id: int, modulus: int, *, namespace: str) -> int:
    digest = hashlib.sha256(f"{namespace}/{int(sample_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, int(modulus))


def apply_weak_corruption(
    normalized_images: torch.Tensor,
    *,
    corruption: str,
    sample_ids: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    noise_sigma: float = 0.03,
    noise_seed: int = 2026080602,
    blur_sigma: float = 0.8,
    blur_kernel_size: int = 3,
    blur_padding: str = "reflect",
    mask_size: int = 8,
    mask_positions: Sequence[int] = (4, 12, 20),
    mask_fill: str = "channel_mean",
) -> torch.Tensor:
    mean_tensor = torch.tensor(
        mean, device=normalized_images.device, dtype=normalized_images.dtype
    ).view(1, 3, 1, 1)
    std_tensor = torch.tensor(
        std, device=normalized_images.device, dtype=normalized_images.dtype
    ).view(1, 3, 1, 1)
    raw = (normalized_images * std_tensor + mean_tensor).clamp(0.0, 1.0)
    if corruption == "mask":
        changed = raw.clone()
        positions = tuple(int(value) for value in mask_positions)
        if not positions:
            raise ValueError("DiR mask_positions must not be empty")
        if str(mask_fill) != "channel_mean":
            raise ValueError("DiR mask_fill must be channel_mean")
        maximum = int(raw.shape[-1]) - int(mask_size)
        if any(value < 0 or value > maximum for value in positions):
            raise ValueError("DiR mask position exceeds image bounds")
        coordinate_pairs = [(top, left) for top in positions for left in positions]
        for row, sample_id in enumerate(sample_ids.detach().cpu().tolist()):
            position_index = _sample_hash_index(
                int(sample_id), len(coordinate_pairs), namespace="dir-mask"
            )
            top, left = coordinate_pairs[position_index]
            changed[
                row : row + 1,
                :,
                top : top + int(mask_size),
                left : left + int(mask_size),
            ] = mean_tensor
    elif corruption == "blur":
        kernel_size = int(blur_kernel_size)
        kernel = _gaussian_kernel(
            kernel_size, float(blur_sigma), device=raw.device, dtype=raw.dtype
        ).expand(3, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        if str(blur_padding) not in {"reflect", "replicate"}:
            raise ValueError("DiR blur_padding must be reflect or replicate")
        changed = F.conv2d(
            F.pad(raw, (padding, padding, padding, padding), mode=str(blur_padding)),
            kernel,
            groups=3,
        )
    elif corruption == "noise":
        noise_parts = []
        for sample_id in sample_ids.detach().cpu().tolist():
            generator = torch.Generator(device=raw.device)
            generator.manual_seed(int(noise_seed) + int(sample_id))
            noise_parts.append(
                torch.randn(
                    raw.shape[1:],
                    generator=generator,
                    device=raw.device,
                    dtype=raw.dtype,
                )
            )
        noise = torch.stack(noise_parts, dim=0)
        changed = (raw + float(noise_sigma) * noise).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown DiR corruption: {corruption}")
    return (changed - mean_tensor) / std_tensor


def _patching_corruption_validity_audit(
    model: nn.Module,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    corruption: str,
    mean: Sequence[float],
    std: Sequence[float],
    same_head: bool,
    minimum_relative_effect: float,
    minimum_prediction_retention: float,
    noise_sigma: float,
    noise_seed: int,
    blur_sigma: float,
    blur_kernel_size: int,
    blur_padding: str,
    mask_size: int,
    mask_positions: Sequence[int],
    mask_fill: str,
) -> dict[str, Any]:
    """Compute model-specific patching corruption validity once for reuse.

    The returned per-sample masks are deterministic for a fixed model, batch
    manifest and corruption. Family-level DiR/Dense comparisons may intersect
    these masks before either pair computes recovery CKA so both families use
    exactly the same samples.
    """

    depth = len(model.transformer_blocks)
    final_output_key = f"block_{depth - 1:02d}_output"
    audit_feature_keys = [
        "post_layernorm_full",
        "post_layernorm_cls",
        "post_layernorm_patch",
        "pre_layernorm_full",
        "pre_layernorm_cls",
        "pre_layernorm_patch",
    ]
    if same_head:
        audit_feature_keys.append("logits")
    relative_effect_values = {key: [] for key in audit_feature_keys}
    absolute_effect_values = {key: [] for key in audit_feature_keys}
    valid_values = {key: [] for key in audit_feature_keys}
    prediction_retained: list[torch.Tensor] = []
    audit_forward_count = 0
    model.eval().to(device)
    with torch.no_grad():
        for images_cpu, _labels_cpu, ids_cpu in batches:
            images = images_cpu.to(device)
            ids = ids_cpu.to(device)
            corrupted = apply_weak_corruption(
                images,
                corruption=corruption,
                sample_ids=ids,
                mean=mean,
                std=std,
                noise_sigma=float(noise_sigma),
                noise_seed=int(noise_seed),
                blur_sigma=float(blur_sigma),
                blur_kernel_size=int(blur_kernel_size),
                blur_padding=str(blur_padding),
                mask_size=int(mask_size),
                mask_positions=mask_positions,
                mask_fill=str(mask_fill),
            )
            capture_points = ["pre_classifier", final_output_key]
            clean_logits, clean = forward_with_capture_and_interventions(
                model, images, capture_points=capture_points
            )
            corrupted_logits, corrupted_taps = forward_with_capture_and_interventions(
                model, corrupted, capture_points=capture_points
            )
            audit_forward_count += 2
            clean_post = clean["pre_classifier"].float()
            corrupted_post = corrupted_taps["pre_classifier"].float()
            clean_pre = clean[final_output_key].float()
            corrupted_pre = corrupted_taps[final_output_key].float()
            post_target = clean_post - corrupted_post
            pre_target = clean_pre - corrupted_pre
            targets = {
                "post_layernorm_full": _feature_view(post_target, "full_token"),
                "post_layernorm_cls": _feature_view(post_target, "cls"),
                "post_layernorm_patch": _feature_view(post_target, "patch"),
                "pre_layernorm_full": _feature_view(pre_target, "full_token"),
                "pre_layernorm_cls": _feature_view(pre_target, "cls"),
                "pre_layernorm_patch": _feature_view(pre_target, "patch"),
            }
            clean_views = {
                "post_layernorm_full": _feature_view(clean_post, "full_token"),
                "post_layernorm_cls": _feature_view(clean_post, "cls"),
                "post_layernorm_patch": _feature_view(clean_post, "patch"),
                "pre_layernorm_full": _feature_view(clean_pre, "full_token"),
                "pre_layernorm_cls": _feature_view(clean_pre, "cls"),
                "pre_layernorm_patch": _feature_view(clean_pre, "patch"),
            }
            if same_head:
                targets["logits"] = clean_logits.float() - corrupted_logits.float()
                clean_views["logits"] = clean_logits.float()
            retained = corrupted_logits.argmax(dim=1) == clean_logits.argmax(dim=1)
            prediction_retained.append(retained.float().cpu())
            for key in audit_feature_keys:
                target_flat = targets[key].reshape(images.shape[0], -1)
                clean_flat = clean_views[key].reshape(images.shape[0], -1)
                absolute_effect = target_flat.norm(dim=1)
                relative_effect = absolute_effect / clean_flat.norm(dim=1).clamp_min(1e-12)
                valid = (relative_effect >= float(minimum_relative_effect)) & retained
                absolute_effect_values[key].append(absolute_effect.cpu())
                relative_effect_values[key].append(relative_effect.cpu())
                valid_values[key].append(valid.cpu())

    retained_values = torch.cat(prediction_retained)
    audits: dict[str, Any] = {}
    valid_masks: dict[str, torch.Tensor] = {}
    for key in audit_feature_keys:
        relative_effects = torch.cat(relative_effect_values[key])
        absolute_effects = torch.cat(absolute_effect_values[key])
        valid_mask = torch.cat(valid_values[key]).bool()
        valid_masks[key] = valid_mask
        audits[key] = {
            "median_relative_corruption_effect": float(relative_effects.median()),
            "mean_relative_corruption_effect": float(relative_effects.mean()),
            "median_absolute_corruption_effect": float(absolute_effects.median()),
            "mean_absolute_corruption_effect": float(absolute_effects.mean()),
            "prediction_retention": float(retained_values.mean()),
            "valid_fraction": float(valid_mask.float().mean()),
            "minimum_relative_effect": float(minimum_relative_effect),
            "minimum_prediction_retention": float(minimum_prediction_retention),
            "validity_passed": bool(
                float(relative_effects.median()) >= float(minimum_relative_effect)
                and float(retained_values.mean()) >= float(minimum_prediction_retention)
            ),
        }
    return {
        "audits": audits,
        "valid_masks": valid_masks,
        "audit_forward_count": int(audit_forward_count),
        "sample_count": int(retained_values.numel()),
        "same_head": bool(same_head),
        "corruption": str(corruption),
        "contract": "model_specific_validity_reusable_before_family_level_sample_intersection",
    }
