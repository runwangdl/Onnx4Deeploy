# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TC-ResNet (temporal-conv ResNet) models for ONNX export."""

from .tc_resnet import TCResBlock, TCResNet, tc_resnet8, tc_resnet14

try:
    from .tc_resnet_quant import QuantTCResBlock, QuantTCResNet, quant_tc_resnet8, quant_tc_resnet14

    __all__ = [
        "TCResBlock",
        "TCResNet",
        "tc_resnet8",
        "tc_resnet14",
        "QuantTCResBlock",
        "QuantTCResNet",
        "quant_tc_resnet8",
        "quant_tc_resnet14",
    ]
except ImportError:
    __all__ = ["TCResBlock", "TCResNet", "tc_resnet8", "tc_resnet14"]
