# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized MatchboxNet for embedded KWS.

Mirrors ``matchboxnet.MatchboxNet`` with int8 per-tensor quant. Uses the
same dq/q wrapping around the residual add as QuantResNet8 / QuantTCResNet.
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


def _qconv1xk(in_ch: int, out_ch: int, k: int, stride: int = 1, groups: int = 1):
    """Brevitas QuantConv2d with kernel (1, k)."""
    pad = k // 2
    return qnn.QuantConv2d(
        in_ch,
        out_ch,
        kernel_size=(1, k),
        stride=(1, stride),
        padding=(0, pad),
        groups=groups,
        bias=True,
        **_QUANT_KW,
    )


class QuantTimeChannelSepConv(nn.Module):
    """Quantized DW 1×k + PW 1×1 + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int):
        """Build the quantized DW + PW pair."""
        super().__init__()
        self.dw = _qconv1xk(in_ch, in_ch, k, groups=in_ch)
        self.pw = _qconv1xk(in_ch, out_ch, k=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DW → PW → BN → ReLU."""
        return self.relu(self.bn(self.pw(self.dw(x))))


class QuantMatchboxBlock(nn.Module):
    """Quantized MatchboxNet block (R sub-blocks + 1×1 residual)."""

    def __init__(self, in_ch: int, out_ch: int, k: int, R: int = 2):
        """Build the quantized block."""
        super().__init__()
        layers = []
        prev = in_ch
        for _ in range(R):
            layers.append(QuantTimeChannelSepConv(prev, out_ch, k))
            prev = out_ch
        self.body = nn.Sequential(*layers)
        self.skip_conv = _qconv1xk(in_ch, out_ch, k=1)
        self.skip_bn = nn.BatchNorm2d(out_ch)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        # Strip QuantTensors around the residual add.
        self.dq_main = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.dq_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )
        self.add_q = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Body branch + skip branch, then ReLU."""
        body = self.dq_main(self.body(x))
        skip = self.dq_identity(self.skip_bn(self.skip_conv(x)))
        return self.relu(self.add_q(body + skip))


class QuantMatchboxNet(nn.Module):
    """Brevitas-quantized MatchboxNet."""

    def __init__(
        self,
        num_classes: int = 12,
        in_features: int = 64,
        n_time: int = 128,
        channels: int = 64,
        R: int = 2,
        block_kernels=None,
    ):
        """Build the quantized MatchboxNet."""
        super().__init__()
        if block_kernels is None:
            block_kernels = [13, 15, 17]
        self.in_features = in_features
        self.n_time = n_time

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        self.stem_conv = _qconv1xk(in_features, channels, k=11, stride=2)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.relu = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        body = []
        prev = channels
        for k in block_kernels:
            body.append(QuantMatchboxBlock(prev, channels, k, R=R))
            prev = channels
        self.body = nn.Sequential(*body)

        self.tail_conv1 = _qconv1xk(channels, 128, k=29)
        self.tail_bn1 = nn.BatchNorm2d(128)
        self.tail_conv2 = _qconv1xk(128, 128, k=1)
        self.tail_bn2 = nn.BatchNorm2d(128)
        self.tail_conv3 = qnn.QuantConv2d(
            128,
            num_classes,
            kernel_size=(1, 1),
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        self.pool_dq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=False)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_features, 1, n_time) → (B, num_classes)."""
        x = self.input_quant(x)
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.body(x)
        x = self.relu(self.tail_bn1(self.tail_conv1(x)))
        x = self.relu(self.tail_bn2(self.tail_conv2(x)))
        x = self.tail_conv3(x)
        x = self.pool_dq(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        return x


def quant_matchboxnet(
    num_classes: int = 12,
    in_features: int = 64,
    n_time: int = 128,
    channels: int = 64,
    R: int = 2,
) -> QuantMatchboxNet:
    """Factory for Brevitas-quantized MatchboxNet."""
    return QuantMatchboxNet(
        num_classes=num_classes,
        in_features=in_features,
        n_time=n_time,
        channels=channels,
        R=R,
    )
