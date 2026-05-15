# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized TC-ResNet for keyword spotting.

Mirrors ``tc_resnet.TCResNet`` with Brevitas int8 quant and the same
dq/q wrapping around the residual add that QuantResNet8 uses.
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


class QuantTCResBlock(nn.Module):
    """Brevitas-quantized counterpart of ``tc_resnet.TCResBlock``."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 9,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        """Build the residual block."""
        super().__init__()
        pad = kernel // 2
        self.conv1 = qnn.QuantConv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel),
            stride=(1, stride),
            padding=(0, pad),
            bias=True,
            **_QUANT_KW,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.conv2 = qnn.QuantConv2d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel),
            stride=(1, 1),
            padding=(0, pad),
            bias=True,
            **_QUANT_KW,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.dq_main = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.dq_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.add_q = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two 1×k convs with residual skip."""
        identity = self.dq_identity(x if self.downsample is None else self.downsample(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dq_main(out)
        out = self.add_q(out + identity)
        out = self.relu(out)
        return out


class _QuantTCDownsample(nn.Module):
    """Brevitas-quantized 1×1 stride-S downsample."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """Build the 1×1 projection."""
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=(1, stride),
            bias=True,
            **_QUANT_KW,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1×1 conv + BN."""
        return self.bn(self.conv(x))


class QuantTCResNet(nn.Module):
    """Brevitas-quantized TC-ResNet."""

    def __init__(
        self,
        num_classes: int = 12,
        n_mfcc: int = 40,
        n_time: int = 49,
        base_channels: int = 16,
        stage_channels=None,
        kernel: int = 9,
    ):
        """Build the quantized TC-ResNet."""
        super().__init__()
        if stage_channels is None:
            stage_channels = [24, 32, 48]
        self.n_mfcc = n_mfcc
        self.n_time = n_time

        pad = kernel // 2
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        self.stem_conv = qnn.QuantConv2d(
            n_mfcc,
            base_channels,
            kernel_size=(1, kernel),
            stride=(1, 1),
            padding=(0, pad),
            bias=True,
            **_QUANT_KW,
        )
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        stages = []
        in_ch = base_channels
        for out_ch in stage_channels:
            down = _QuantTCDownsample(in_ch, out_ch, stride=2)
            stages.append(QuantTCResBlock(in_ch, out_ch, kernel, stride=2, downsample=down))
            stages.append(QuantTCResBlock(out_ch, out_ch, kernel, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool_dq = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_iq = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )
        self.fc = qnn.QuantLinear(
            in_ch,
            num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_mfcc, 1, n_time) → (B, num_classes)."""
        x = self.input_quant(x)
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.stages(x)
        x = self.pool_dq(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        x = self.fc_iq(x)
        x = self.fc(x)
        return x


def quant_tc_resnet8(
    num_classes: int = 12, n_mfcc: int = 40, n_time: int = 49
) -> QuantTCResNet:
    """Brevitas-quantized TC-ResNet8."""
    return QuantTCResNet(
        num_classes=num_classes,
        n_mfcc=n_mfcc,
        n_time=n_time,
        base_channels=16,
        stage_channels=[24, 32, 48],
        kernel=9,
    )


def quant_tc_resnet14(
    num_classes: int = 12, n_mfcc: int = 40, n_time: int = 49
) -> QuantTCResNet:
    """Brevitas-quantized TC-ResNet14."""
    return QuantTCResNet(
        num_classes=num_classes,
        n_mfcc=n_mfcc,
        n_time=n_time,
        base_channels=16,
        stage_channels=[24, 24, 32, 32, 48, 48],
        kernel=9,
    )
