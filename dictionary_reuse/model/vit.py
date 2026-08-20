"""Vision Transformer backbone used by the DiR release.

The module exposes the layer and tap tensors required by training and
measurement while preserving stable measurement parameter names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

import torch
from torch import nn


@dataclass
class VisionTransformerModelConfiguration:
    image_size: int
    patch_size: int
    patch_embedding_kernel_size: int
    patch_embedding_stride: int
    patch_embedding_padding: int
    number_of_input_channels: int
    number_of_classes: int
    embedding_dimension: int
    transformer_depth: int
    number_of_attention_heads: int
    mlp_hidden_dimension: int


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        patch_embedding_kernel_size: int,
        patch_embedding_stride: int,
        patch_embedding_padding: int,
        number_of_input_channels: int,
        embedding_dimension: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embedding_kernel_size = patch_embedding_kernel_size
        self.patch_embedding_stride = patch_embedding_stride
        self.patch_embedding_padding = patch_embedding_padding
        self.number_of_input_channels = number_of_input_channels
        self.embedding_dimension = embedding_dimension
        self.projection = nn.Conv2d(number_of_input_channels, embedding_dimension, kernel_size=patch_embedding_kernel_size, stride=patch_embedding_stride, padding=patch_embedding_padding)
        projected_size = (image_size + 2 * patch_embedding_padding - patch_embedding_kernel_size) // patch_embedding_stride + 1
        self.number_of_patches_per_side = int(projected_size)
        self.number_of_patches = int(projected_size * projected_size)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        projected_patches = self.projection(input_images)
        return projected_patches.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        number_of_attention_heads: int,
    ):
        super().__init__()
        if embedding_dimension % number_of_attention_heads != 0:
            raise ValueError(f"embedding_dimension ({embedding_dimension}) must be divisible by number_of_attention_heads ({number_of_attention_heads})")
        self.embedding_dimension = embedding_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.attention_head_dimension = embedding_dimension // number_of_attention_heads
        # Default dense QKV path. Full-DiR runs replace
        # this with explicit query/key/value DiR projections and set the
        # combined projection to None, without changing attention dimensions.
        self.query_key_value_projection = nn.Linear(embedding_dimension, 3 * embedding_dimension)
        self.query_projection: nn.Module | None = None
        self.key_projection: nn.Module | None = None
        self.value_projection: nn.Module | None = None
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension)
        # DiR installs one trainable QK logit scale and one trainable VO
        # output scale after explicit Q/K/V/O dictionary replacement. Matching
        # non-trainable coordinate-gauge buffers preserve the represented
        # function when a projection's relative-C support anchor changes.
        # Activation-RMS measurement is opt-in and used only during bounded
        # report passes, so normal training forwards carry no extra reductions.
        self._dictionary_attention_rms_measurement_enabled = False
        self._dictionary_attention_rms_measurement_state: dict[str, object] | None = None


    @torch.no_grad()
    def begin_dictionary_attention_rms_measurement_(self, *, include_structure: bool = False) -> None:
        self._dictionary_attention_rms_measurement_enabled = True
        self._dictionary_attention_rms_measurement_state = {
            "include_structure": bool(include_structure),
            "qk_logits_pre_scale_sum_sq": None,
            "qk_logits_pre_scale_count": 0,
            "qk_logits_post_scale_sum_sq": None,
            "qk_logits_post_scale_count": 0,
            "vo_output_pre_scale_sum_sq": None,
            "vo_output_pre_scale_count": 0,
            "vo_output_post_scale_sum_sq": None,
            "vo_output_post_scale_count": 0,
            "qk_logits_post_scale_cls": None,
            "attention_cls_map": None,
            "vo_output_post_scale_cls": None,
        }

    @torch.no_grad()
    def _record_dictionary_attention_rms_(self, key: str, value: torch.Tensor) -> None:
        if not bool(self._dictionary_attention_rms_measurement_enabled):
            return
        state = self._dictionary_attention_rms_measurement_state
        if not isinstance(state, dict):
            return
        detached = value.detach().float()
        sum_sq_key = f"{key}_sum_sq"
        count_key = f"{key}_count"
        batch_sum_sq = detached.pow(2).sum()
        previous = state.get(sum_sq_key)
        state[sum_sq_key] = batch_sum_sq if not isinstance(previous, torch.Tensor) else previous + batch_sum_sq
        state[count_key] = int(state.get(count_key, 0) or 0) + int(detached.numel())

    @torch.no_grad()
    def end_dictionary_attention_rms_measurement_(self) -> dict[str, torch.Tensor]:
        self._dictionary_attention_rms_measurement_enabled = False
        state = self._dictionary_attention_rms_measurement_state
        self._dictionary_attention_rms_measurement_state = None
        if not isinstance(state, dict):
            return {}
        results: dict[str, torch.Tensor] = {}
        for key in (
            "qk_logits_pre_scale",
            "qk_logits_post_scale",
            "vo_output_pre_scale",
            "vo_output_post_scale",
        ):
            sum_sq = state.get(f"{key}_sum_sq")
            count = int(state.get(f"{key}_count", 0) or 0)
            if isinstance(sum_sq, torch.Tensor) and count > 0:
                results[f"{key}_rms"] = (sum_sq / float(count)).clamp_min(0.0).sqrt()
        if bool(state.get("include_structure", False)):
            for key in ("qk_logits_post_scale_cls", "attention_cls_map", "vo_output_post_scale_cls"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    results[key] = value
        return results

    @staticmethod
    @torch.no_grad()
    def _commit_projection_coordinate_corrections_(
        projections: tuple[nn.Module | None, ...],
        coordinate_log_scale: torch.Tensor,
    ) -> None:
        correction = coordinate_log_scale.new_zeros(())
        for projection in projections:
            commit = getattr(projection, "commit_pending_relative_support_transition_", None)
            if callable(commit):
                correction.add_(commit(apply_local_scale=False).to(
                    device=coordinate_log_scale.device,
                    dtype=coordinate_log_scale.dtype,
                ))
            consume_direct = getattr(projection, "consume_relative_support_direct_log_correction_", None)
            if callable(consume_direct):
                correction.add_(consume_direct().to(
                    device=coordinate_log_scale.device,
                    dtype=coordinate_log_scale.dtype,
                ))
        coordinate_log_scale.add_(correction)

    @torch.no_grad()
    def flush_dictionary_coordinate_corrections_(self) -> None:
        qk_coordinate = getattr(self, "dictionary_qk_coordinate_log_scale", None)
        if isinstance(qk_coordinate, torch.Tensor):
            self._commit_projection_coordinate_corrections_(
                (self.query_projection, self.key_projection), qk_coordinate
            )
        vo_coordinate = getattr(self, "dictionary_vo_coordinate_log_scale", None)
        if isinstance(vo_coordinate, torch.Tensor):
            self._commit_projection_coordinate_corrections_(
                (self.value_projection, self.output_projection), vo_coordinate
            )

    def forward_with_measurement_tensors(
        self,
        input_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the attention residual contribution and explicit DiR taps."""
        batch_size, sequence_length, embedding_dimension = input_sequence.shape
        if self.query_projection is not None and self.key_projection is not None and self.value_projection is not None:
            query_tensor = self.query_projection(input_sequence)
            key_tensor = self.key_projection(input_sequence)
            value_tensor = self.value_projection(input_sequence)
            query_tensor = query_tensor.reshape(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension).permute(0, 2, 1, 3)
            key_tensor = key_tensor.reshape(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension).permute(0, 2, 1, 3)
            value_tensor = value_tensor.reshape(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension).permute(0, 2, 1, 3)
        else:
            if self.query_key_value_projection is None:
                raise RuntimeError("MultiHeadSelfAttention has no QKV projection path configured")
            query_key_value = self.query_key_value_projection(input_sequence)
            query_key_value = query_key_value.reshape(batch_size, sequence_length, 3, self.number_of_attention_heads, self.attention_head_dimension)
            query_key_value = query_key_value.permute(2, 0, 3, 1, 4)
            query_tensor, key_tensor, value_tensor = query_key_value[0], query_key_value[1], query_key_value[2]
        attention_scores = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        attention_scores = attention_scores / (self.attention_head_dimension ** 0.5)
        qk_log_scale = getattr(self, "dictionary_qk_log_scale", None)
        if qk_log_scale is not None:
            qk_coordinate_log_scale = getattr(self, "dictionary_qk_coordinate_log_scale", None)
            if qk_coordinate_log_scale is not None:
                qk_log_scale = qk_log_scale + qk_coordinate_log_scale
            attention_scores = attention_scores * qk_log_scale.exp().to(device=attention_scores.device, dtype=attention_scores.dtype)
        attention_probabilities = attention_scores.softmax(dim=-1)
        value_weighted_heads = torch.matmul(attention_probabilities, value_tensor)
        value_weighted_output = value_weighted_heads.transpose(1, 2).reshape(batch_size, sequence_length, embedding_dimension)
        projected_output = self.output_projection(value_weighted_output)
        vo_log_scale = getattr(self, "dictionary_vo_log_scale", None)
        if vo_log_scale is not None:
            vo_coordinate_log_scale = getattr(self, "dictionary_vo_coordinate_log_scale", None)
            if vo_coordinate_log_scale is not None:
                vo_log_scale = vo_log_scale + vo_coordinate_log_scale
            projected_output = projected_output * vo_log_scale.exp().to(device=projected_output.device, dtype=projected_output.dtype)
        output = projected_output
        return output, {
            "attention_probability": attention_probabilities,
            "value_tensor": value_tensor,
            "value_weighted_heads": value_weighted_heads,
            "value_weighted_output": value_weighted_output,
            "post_o_attention_output": output,
        }

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dimension = input_sequence.shape
        if self.query_projection is not None and self.key_projection is not None and self.value_projection is not None:
            query_tensor = self.query_projection(input_sequence)
            key_tensor = self.key_projection(input_sequence)
            value_tensor = self.value_projection(input_sequence)
            query_tensor = query_tensor.reshape(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension).permute(0, 2, 1, 3)
            key_tensor = key_tensor.reshape(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension).permute(0, 2, 1, 3)
            value_tensor = value_tensor.reshape(batch_size, sequence_length, self.number_of_attention_heads, self.attention_head_dimension).permute(0, 2, 1, 3)
        else:
            if self.query_key_value_projection is None:
                raise RuntimeError("MultiHeadSelfAttention has no QKV projection path configured")
            query_key_value = self.query_key_value_projection(input_sequence)
            query_key_value = query_key_value.reshape(batch_size, sequence_length, 3, self.number_of_attention_heads, self.attention_head_dimension)
            query_key_value = query_key_value.permute(2, 0, 3, 1, 4)
            query_tensor, key_tensor, value_tensor = query_key_value[0], query_key_value[1], query_key_value[2]
        attention_scores = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        attention_scores = attention_scores / (self.attention_head_dimension ** 0.5)
        self._record_dictionary_attention_rms_("qk_logits_pre_scale", attention_scores)
        qk_log_scale = getattr(self, "dictionary_qk_log_scale", None)
        if qk_log_scale is not None:
            qk_coordinate_log_scale = getattr(self, "dictionary_qk_coordinate_log_scale", None)
            if qk_coordinate_log_scale is not None:
                qk_log_scale = qk_log_scale + qk_coordinate_log_scale
            attention_scores = attention_scores * qk_log_scale.exp().to(
                device=attention_scores.device, dtype=attention_scores.dtype
            )
        self._record_dictionary_attention_rms_("qk_logits_post_scale", attention_scores)
        if (
            bool(self._dictionary_attention_rms_measurement_enabled)
            and isinstance(self._dictionary_attention_rms_measurement_state, dict)
            and bool(self._dictionary_attention_rms_measurement_state.get("include_structure", False))
        ):
            self._dictionary_attention_rms_measurement_state["qk_logits_post_scale_cls"] = attention_scores[:, :, 0, :].detach()
        attention_probabilities = attention_scores.softmax(dim=-1)
        if (
            bool(self._dictionary_attention_rms_measurement_enabled)
            and isinstance(self._dictionary_attention_rms_measurement_state, dict)
            and bool(self._dictionary_attention_rms_measurement_state.get("include_structure", False))
        ):
            self._dictionary_attention_rms_measurement_state["attention_cls_map"] = attention_probabilities[:, :, 0, :].detach()
        attended_values = torch.matmul(attention_probabilities, value_tensor)
        attended_values = attended_values.transpose(1, 2).reshape(batch_size, sequence_length, embedding_dimension)
        projected_output = self.output_projection(attended_values)
        self._record_dictionary_attention_rms_("vo_output_pre_scale", projected_output)
        vo_log_scale = getattr(self, "dictionary_vo_log_scale", None)
        if vo_log_scale is not None:
            vo_coordinate_log_scale = getattr(self, "dictionary_vo_coordinate_log_scale", None)
            if vo_coordinate_log_scale is not None:
                vo_log_scale = vo_log_scale + vo_coordinate_log_scale
            projected_output = projected_output * vo_log_scale.exp().to(
                device=projected_output.device, dtype=projected_output.dtype
            )
        self._record_dictionary_attention_rms_("vo_output_post_scale", projected_output)
        if (
            bool(self._dictionary_attention_rms_measurement_enabled)
            and isinstance(self._dictionary_attention_rms_measurement_state, dict)
            and bool(self._dictionary_attention_rms_measurement_state.get("include_structure", False))
        ):
            self._dictionary_attention_rms_measurement_state["vo_output_post_scale_cls"] = projected_output[:, 0, :].detach()
        return projected_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int):
        super().__init__()
        self.first_linear_layer = nn.Linear(embedding_dimension, hidden_dimension)
        self.activation = nn.GELU()
        self.second_linear_layer = nn.Linear(hidden_dimension, embedding_dimension)

    def forward_with_measurement_tensors(
        self,
        input_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_pre_activation = self.first_linear_layer(input_sequence)
        hidden_representation = self.activation(hidden_pre_activation)
        output_representation = self.second_linear_layer(hidden_representation)
        return output_representation, {
            "mlp_hidden_pre_activation": hidden_pre_activation,
            "mlp_hidden": hidden_representation,
            "post_w2_mlp_output": output_representation,
        }

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        hidden_representation = self.first_linear_layer(input_sequence)
        hidden_representation = self.activation(hidden_representation)
        output_representation = self.second_linear_layer(hidden_representation)
        return output_representation


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        number_of_attention_heads: int,
        mlp_hidden_dimension: int,
    ):
        super().__init__()
        self.first_layer_normalization = nn.LayerNorm(embedding_dimension)
        self.multi_head_self_attention = MultiHeadSelfAttention(embedding_dimension, number_of_attention_heads)
        self.second_layer_normalization = nn.LayerNorm(embedding_dimension)
        self.feed_forward_network = FeedForwardNetwork(embedding_dimension, mlp_hidden_dimension)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        attention_input = self.first_layer_normalization(input_sequence)
        attention_output = self.multi_head_self_attention(attention_input)
        residual_after_attention = input_sequence + attention_output
        feed_forward_input = self.second_layer_normalization(residual_after_attention)
        feed_forward_output = self.feed_forward_network(feed_forward_input)
        return residual_after_attention + feed_forward_output

    def forward_with_measurement_intermediates(
        self,
        input_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward one block and expose the explicit DiR capture contract."""
        attention_input = self.first_layer_normalization(input_sequence)
        attention_output, attention_taps = self.multi_head_self_attention.forward_with_measurement_tensors(attention_input)
        attention_contribution = attention_output
        residual_after_attention = input_sequence + attention_contribution
        feed_forward_input = self.second_layer_normalization(residual_after_attention)
        feed_forward_output, mlp_taps = self.feed_forward_network.forward_with_measurement_tensors(feed_forward_input)
        mlp_contribution = feed_forward_output
        block_output = residual_after_attention + mlp_contribution
        return block_output, {
            "block_input": input_sequence,
            "pre_attention_norm": attention_input,
            **attention_taps,
            "attention_residual_contribution": attention_contribution,
            "post_attention_residual": residual_after_attention,
            "pre_mlp_norm": feed_forward_input,
            **mlp_taps,
            "mlp_output": feed_forward_output,
            "mlp_residual_contribution": mlp_contribution,
            "post_mlp_residual": block_output,
            "block_output": block_output,
            "block_update": block_output - input_sequence,
        }


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        model_configuration: VisionTransformerModelConfiguration,
        *,
        initialize_parameters: bool = True,
    ):
        super().__init__()
        self.model_configuration = model_configuration
        self.image_size = model_configuration.image_size
        self.patch_size = model_configuration.patch_size
        self.patch_embedding_kernel_size = model_configuration.patch_embedding_kernel_size
        self.patch_embedding_stride = model_configuration.patch_embedding_stride
        self.patch_embedding_padding = model_configuration.patch_embedding_padding
        self.number_of_input_channels = model_configuration.number_of_input_channels
        self.number_of_classes = model_configuration.number_of_classes
        self.embedding_dimension = model_configuration.embedding_dimension
        self.transformer_depth = model_configuration.transformer_depth
        self.number_of_attention_heads = model_configuration.number_of_attention_heads
        self.mlp_hidden_dimension = model_configuration.mlp_hidden_dimension
        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, self.patch_embedding_kernel_size, self.patch_embedding_stride, self.patch_embedding_padding, self.number_of_input_channels, self.embedding_dimension)
        number_of_patches = self.patch_embedding.number_of_patches
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dimension))
        self.position_embedding = nn.Parameter(torch.zeros(1, number_of_patches + 1, self.embedding_dimension))
        self.transformer_blocks = nn.ModuleList([TransformerEncoderBlock(self.embedding_dimension, self.number_of_attention_heads, self.mlp_hidden_dimension) for _ in range(self.transformer_depth)])
        self.pre_classifier_normalization = nn.LayerNorm(self.embedding_dimension)
        self.classification_head = nn.Linear(self.embedding_dimension, self.number_of_classes)
        if initialize_parameters:
            self._initialize_parameters()
        else:
            self._initialize_deterministic_normalization_parameters()


    def _initialize_deterministic_normalization_parameters(self) -> None:
        """Initialize only deterministic non-carrier parameters for full DiR builds."""
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_with_measurement_tensors(
        self,
        input_images: torch.Tensor,
        include_internal_block_taps: bool = False,
        requested_tap_names: Set[str] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return logits plus requested token-sequence tensors for DiR measurements.

        ``requested_tap_names`` limits stored activations without changing the
        forward computation. ``None`` preserves the all-taps behavior.
        """
        patch_tokens = self.patch_embedding(input_images)
        batch_size = patch_tokens.shape[0]
        class_token_parameter = self.class_token() if isinstance(self.class_token, nn.Module) else self.class_token
        position_embedding_parameter = self.position_embedding() if isinstance(self.position_embedding, nn.Module) else self.position_embedding
        repeated_class_token = class_token_parameter.expand(batch_size, -1, -1)
        token_sequence = torch.cat([repeated_class_token, patch_tokens], dim=1)
        token_sequence = token_sequence + position_embedding_parameter

        def should_store(tap_name: str) -> bool:
            return requested_tap_names is None or str(tap_name) in requested_tap_names

        measurement_tensor_dictionary: Dict[str, torch.Tensor] = {}
        if should_store("patch_embedding_out"):
            measurement_tensor_dictionary["patch_embedding_out"] = patch_tokens
        if should_store("embedding_sequence_out"):
            measurement_tensor_dictionary["embedding_sequence_out"] = token_sequence

        for transformer_block_index, transformer_block in enumerate(self.transformer_blocks):
            if include_internal_block_taps:
                token_sequence, intermediate_dictionary = transformer_block.forward_with_measurement_intermediates(token_sequence)
                block_output_tap_name = f"block_{transformer_block_index:02d}_output"
                if should_store(block_output_tap_name):
                    measurement_tensor_dictionary[block_output_tap_name] = token_sequence
                for intermediate_name, intermediate_tensor in intermediate_dictionary.items():
                    intermediate_tap_name = f"block_{transformer_block_index:02d}_{intermediate_name}"
                    if should_store(intermediate_tap_name):
                        measurement_tensor_dictionary[intermediate_tap_name] = intermediate_tensor
            else:
                token_sequence = transformer_block(token_sequence)
                block_output_tap_name = f"block_{transformer_block_index:02d}_output"
                if should_store(block_output_tap_name):
                    measurement_tensor_dictionary[block_output_tap_name] = token_sequence

        normalized_token_sequence = self.pre_classifier_normalization(token_sequence)
        if should_store("pre_classifier"):
            measurement_tensor_dictionary["pre_classifier"] = normalized_token_sequence
        class_token_representation = normalized_token_sequence[:, 0]
        logits = self.classification_head(class_token_representation)
        return logits, measurement_tensor_dictionary

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        logits, _measurement_tensors = self.forward_with_measurement_tensors(
            input_images=input_images,
            include_internal_block_taps=False,
            requested_tap_names=set(),
        )
        return logits



def create_vision_transformer_small_patch4_for_cifar100(
    number_of_classes: int,
    *,
    initialize_parameters: bool = True,
) -> VisionTransformerClassifier:
    model_configuration = VisionTransformerModelConfiguration(
        image_size=32,
        patch_size=4,
        patch_embedding_kernel_size=7,
        patch_embedding_stride=4,
        patch_embedding_padding=3,
        number_of_input_channels=3,
        number_of_classes=number_of_classes,
        embedding_dimension=384,
        transformer_depth=12,
        number_of_attention_heads=6,
        mlp_hidden_dimension=1536,
    )
    return VisionTransformerClassifier(model_configuration, initialize_parameters=initialize_parameters)
