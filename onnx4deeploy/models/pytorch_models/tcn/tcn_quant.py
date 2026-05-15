# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized TCN.

Mirrors ``tcn.TCN`` with int8 per-tensor quant. The residual add uses the
same dq/q wrapping as QuantResNet8.
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


class QuantTemporalBlock(nn.Module):
    """Brevitas-quantized TCN residual block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
    ):
        """Build the quantized temporal block."""
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = qnn.QuantConv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            dilation=(1, dilation),
            bias=True,
            **_QUANT_KW,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = qnn.QuantConv2d(
            out_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, pad),
            dilation=(1, dilation),
            bias=True,
            **_QUANT_KW,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.use_skip = in_ch != out_ch or stride != 1
        if self.use_skip:
            self.skip_conv = qnn.QuantConv2d(
                in_ch, out_ch, kernel_size=1, stride=(1, stride), bias=True, **_QUANT_KW
            )
            self.skip_bn = nn.BatchNorm2d(out_ch)
        else:
            self.skip_conv = None
            self.skip_bn = None

        self.dq_main = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.dq_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.add_q = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two dilated convs + residual."""
        identity = self.dq_identity(x if not self.use_skip else self.skip_bn(self.skip_conv(x)))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dq_main(out)
        return self.relu(self.add_q(out + identity))


class QuantTCN(nn.Module):
    """Brevitas-quantized TCN."""

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 9,
        n_time: int = 128,
        channels=None,
        kernel_size: int = 3,
    ):
        """Build the quantized TCN."""
        super().__init__()
        if channels is None:
            channels = [16, 32, 64]
        self.n_time = n_time
        self.in_channels = in_channels

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        blocks = []
        prev = in_channels
        for i, ch in enumerate(channels):
            dilation = 2**i
            blocks.append(QuantTemporalBlock(prev, ch, kernel_size=kernel_size, dilation=dilation))
            prev = ch
        self.blocks = nn.Sequential(*blocks)

        self.head_conv = qnn.QuantConv2d(
            prev,
            num_classes,
            kernel_size=1,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.pool_dq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, 1, n_time) → (B, num_classes)."""
        x = self.input_quant(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool_dq(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return self.flatten(x)


def quant_tcn_har(num_classes: int = 6, in_channels: int = 9, n_time: int = 128) -> QuantTCN:
    """Brevitas-quantized TCN-HAR (UCI-HAR/WISDM-sized)."""
    return QuantTCN(
        num_classes=num_classes,
        in_channels=in_channels,
        n_time=n_time,
        channels=[16, 32, 64],
        kernel_size=3,
    )


def quant_tcn_ecg(num_classes: int = 5, in_channels: int = 1, n_time: int = 256) -> QuantTCN:
    """Brevitas-quantized TCN-ECG."""
    return QuantTCN(
        num_classes=num_classes,
        in_channels=in_channels,
        n_time=n_time,
        channels=[8, 16, 32, 64],
        kernel_size=7,
    )
