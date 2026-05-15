# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""BC-ResNet (Kim 2021) models for ONNX export."""

from .bc_resnet import BCResNet, NormalBlock, TransitionBlock, bc_resnet1, bc_resnet3

try:
    from .bc_resnet_quant import (
        QuantBCResNet,
        QuantNormalBlock,
        QuantTransitionBlock,
        quant_bc_resnet1,
        quant_bc_resnet3,
    )

    __all__ = [
        "BCResNet",
        "NormalBlock",
        "TransitionBlock",
        "bc_resnet1",
        "bc_resnet3",
        "QuantBCResNet",
        "QuantNormalBlock",
        "QuantTransitionBlock",
        "quant_bc_resnet1",
        "quant_bc_resnet3",
    ]
except ImportError:
    __all__ = ["BCResNet", "NormalBlock", "TransitionBlock", "bc_resnet1", "bc_resnet3"]
