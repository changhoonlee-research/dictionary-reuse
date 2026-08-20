"""Model-side DiR primitives: basis, routing, dictionary operator, and ViT."""

from .basis import build_basis_bank
from .dictionary_operator import SeparableDictionaryLinear, iter_dictionary_layers
from .vit import create_vision_transformer_small_patch4_for_cifar100

__all__ = [
    "SeparableDictionaryLinear",
    "build_basis_bank",
    "create_vision_transformer_small_patch4_for_cifar100",
    "iter_dictionary_layers",
]
