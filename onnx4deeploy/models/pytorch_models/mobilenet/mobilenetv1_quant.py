# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized MobileNetV1 for embedded vision deployments.

Mirrors the FP32 MobileNetV1 in ``mobilenetv1.py`` but with Brevitas
QuantConv2d / QuantLinear / QuantReLU substitutions. No residual adds
— purely feed-forward depthwise-separable blocks. Designed to be
``DeepQuant.exportBrevitas``-compatible and to lower to Deeploy's
RequantizedConv / RequantizedGemm via ``qcdq_to_deeploy``.
"""

import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int32Bias

_QUANT_KW = dict(
    weight_quant=Int8WeightPerTensorFloat,
    bias_quant=Int32Bias,
    output_quant=Int8ActPerTensorFloat,
    return_quant_tensor=True,
)


class QuantDepthwiseSeparableConv(nn.Module):
    """Brevitas-quantized depthwise-separable block (DW 3×3 + PW 1×1)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        """Initialize the DW+PW pair with per-tensor int8 quant."""
        super().__init__()
        self.dw = qnn.QuantConv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=True,
            **_QUANT_KW,
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.relu_dw = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.pw = qnn.QuantConv2d(in_ch, out_ch, kernel_size=1, bias=True, **_QUANT_KW)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu_pw = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DW → BN → ReLU → PW → BN → ReLU."""
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x


class QuantMobileNetV1(nn.Module):
    """Brevitas-quantized MobileNetV1.

    Functionally identical to ``mobilenetv1.MobileNetV1`` modulo int8
    quantization of weights/activations.
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, input_channels: int = 3):
        """Build the quantized MobileNetV1."""
        super().__init__()

        def c(ch: int) -> int:
            return int(ch * width_mult)

        cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        self.stem_conv = qnn.QuantConv2d(
            input_channels, c(32), kernel_size=3, stride=2, padding=1, bias=True, **_QUANT_KW
        )
        self.stem_bn = nn.BatchNorm2d(c(32))
        self.stem_relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        in_ch = c(32)
        blocks = []
        for out_ch, stride in cfg:
            blocks.append(QuantDepthwiseSeparableConv(in_ch, c(out_ch), stride=stride))
            in_ch = c(out_ch)
        self.ds_blocks = nn.Sequential(*blocks)
        self.last_channel = in_ch

        self.pool_dq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_iq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.classifier = qnn.QuantLinear(
            self.last_channel,
            num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through QuantMobileNetV1."""
        x = self.input_quant(x)
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.ds_blocks(x)
        x = self.pool_dq(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        x = self.fc_iq(x)
        x = self.classifier(x)
        return x


def quant_mobilenet_v1(
    num_classes: int = 1000, width_mult: float = 1.0, input_channels: int = 3
) -> QuantMobileNetV1:
    """Factory for Brevitas-quantized MobileNetV1."""
    return QuantMobileNetV1(
        num_classes=num_classes, width_mult=width_mult, input_channels=input_channels
    )
