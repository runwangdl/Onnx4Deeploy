# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized BC-ResNet for KWS.

Mirrors ``bc_resnet.BCResNet`` with int8 per-tensor quant. Same dq/q
wrapping around the residual add as QuantResNet8 / QuantTCResNet.
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


def _qconv_freq(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
    """Frequency-only Brevitas QuantConv2d on (kernel, 1)."""
    pad = kernel // 2
    return qnn.QuantConv2d(
        in_ch,
        out_ch,
        kernel_size=(kernel, 1),
        stride=(stride, 1),
        padding=(pad, 0),
        groups=in_ch if in_ch == out_ch else 1,
        bias=True,
        **_QUANT_KW,
    )


def _qconv_time(in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
    """Temporal Brevitas QuantConv2d on (1, kernel)."""
    pad = (kernel // 2) * dilation
    return qnn.QuantConv2d(
        in_ch,
        out_ch,
        kernel_size=(1, kernel),
        stride=(1, 1),
        padding=(0, pad),
        dilation=(1, dilation),
        groups=in_ch if in_ch == out_ch else 1,
        bias=True,
        **_QUANT_KW,
    )


class QuantTransitionBlock(nn.Module):
    """Quantized stride-2 transition block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, dilation: int = 1):
        """Build the quantized transition block."""
        super().__init__()
        self.expand = qnn.QuantConv2d(in_ch, out_ch, kernel_size=1, bias=True, **_QUANT_KW)
        self.bn_expand = nn.BatchNorm2d(out_ch)
        self.f_conv = _qconv_freq(out_ch, out_ch, kernel=3, stride=stride)
        self.bn_f = nn.BatchNorm2d(out_ch)
        self.t_conv = _qconv_time(out_ch, out_ch, kernel=3, dilation=dilation)
        self.bn_t = nn.BatchNorm2d(out_ch)
        self.proj = qnn.QuantConv2d(
            in_ch, out_ch, kernel_size=1, stride=(stride, 1), bias=True, **_QUANT_KW
        )
        self.bn_proj = nn.BatchNorm2d(out_ch)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.dq_main = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.dq_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.add_q = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand → freq-conv → temporal-conv + skip → ReLU."""
        skip = self.dq_identity(self.bn_proj(self.proj(x)))
        out = self.relu(self.bn_expand(self.expand(x)))
        out = self.relu(self.bn_f(self.f_conv(out)))
        out = self.bn_t(self.t_conv(out))
        out = self.dq_main(out)
        return self.relu(self.add_q(out + skip))


class QuantNormalBlock(nn.Module):
    """Quantized stride-1 residual block."""

    def __init__(self, channels: int, dilation: int = 1):
        """Build the quantized normal block."""
        super().__init__()
        self.f_conv = _qconv_freq(channels, channels, kernel=3, stride=1)
        self.bn_f = nn.BatchNorm2d(channels)
        self.t_conv = _qconv_time(channels, channels, kernel=3, dilation=dilation)
        self.bn_t = nn.BatchNorm2d(channels)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.dq_main = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.dq_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.add_q = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """freq-conv → temporal-conv + residual skip."""
        identity = self.dq_identity(x)
        out = self.relu(self.bn_f(self.f_conv(x)))
        out = self.bn_t(self.t_conv(out))
        out = self.dq_main(out)
        return self.relu(self.add_q(out + identity))


class QuantBCResNet(nn.Module):
    """Brevitas-quantized BC-ResNet."""

    def __init__(
        self,
        num_classes: int = 12,
        n_mel: int = 40,
        n_time: int = 49,
        base_channels: int = 8,
        block_counts=None,
        block_dilations=None,
    ):
        """Build the quantized BC-ResNet."""
        super().__init__()
        if block_counts is None:
            block_counts = [2, 2, 4, 4]
        if block_dilations is None:
            block_dilations = [1, 2, 4, 8]

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        self.stem_conv = qnn.QuantConv2d(
            1, base_channels, kernel_size=5, stride=(2, 1), padding=2, bias=True, **_QUANT_KW
        )
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        stages = []
        prev = base_channels
        for stage_idx, (n_blocks, dilation) in enumerate(zip(block_counts, block_dilations)):
            stage_ch = base_channels * (stage_idx + 1)
            stride = 2 if stage_idx > 0 else 1
            stages.append(QuantTransitionBlock(prev, stage_ch, stride=stride, dilation=dilation))
            for _ in range(n_blocks - 1):
                stages.append(QuantNormalBlock(stage_ch, dilation=dilation))
            prev = stage_ch
        self.stages = nn.Sequential(*stages)
        self.last_channel = prev

        self.head_conv = qnn.QuantConv2d(
            self.last_channel,
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
        """(B, 1, n_mel, n_time) → (B, num_classes)."""
        x = self.input_quant(x)
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.pool_dq(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return self.flatten(x)


def quant_bc_resnet1(num_classes: int = 12, n_mel: int = 40, n_time: int = 49) -> QuantBCResNet:
    """Brevitas-quantized BC-ResNet-1."""
    return QuantBCResNet(
        num_classes=num_classes,
        n_mel=n_mel,
        n_time=n_time,
        base_channels=8,
        block_counts=[2, 2, 4, 4],
        block_dilations=[1, 2, 4, 8],
    )


def quant_bc_resnet3(num_classes: int = 12, n_mel: int = 40, n_time: int = 49) -> QuantBCResNet:
    """Brevitas-quantized BC-ResNet-3 (KWS sota < 100K params)."""
    return QuantBCResNet(
        num_classes=num_classes,
        n_mel=n_mel,
        n_time=n_time,
        base_channels=24,
        block_counts=[2, 2, 4, 4],
        block_dilations=[1, 2, 4, 8],
    )
