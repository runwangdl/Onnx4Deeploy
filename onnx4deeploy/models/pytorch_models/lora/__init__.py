# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""LoRA building blocks (loralib-compatible semantics)."""

from .conv_lora import Conv1d, Conv2d, ConvLoRA, lora_parameter_names, mark_only_lora_as_trainable

__all__ = [
    "ConvLoRA",
    "Conv1d",
    "Conv2d",
    "lora_parameter_names",
    "mark_only_lora_as_trainable",
]
