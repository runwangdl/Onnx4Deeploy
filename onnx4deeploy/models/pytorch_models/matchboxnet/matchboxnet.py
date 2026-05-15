# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MatchboxNet: 1D conv keyword spotter for edge devices.

Reference: Majumdar & Ginsburg (2020), "MatchboxNet: 1D Time-Channel
Separable Convolutional Neural Network Architecture for Speech Commands
Recognition" — INTERSPEECH 2020. NVIDIA's compact KWS architecture.

Deeploy-friendly layout: (B, MFCC_channels, 1, Time). All convs are 2D
with kernel ``(1, k)`` — the height axis stays fixed at 1 so PULPConv2D
handles them as if they were ordinary 1D temporal convs.

Architecture (B=3 blocks × R=2 sub-blocks per block, 64 channels):
  Stem  : Conv(in=in_features, out=64, k=11, stride=2)
  Body  : 3 blocks of {(DW(k_b) + PW + BN + ReLU) × 2 + residual 1×1 PW}
          with k_b ∈ [13, 15, 17]
  Tail  : Conv(64, 128, k=29) → Conv(128, 128, k=1) → Conv(128, n_cls, k=1)
  Head  : Global average over time → output logits
"""

import torch
import torch.nn as nn


def _conv1xk(in_ch: int, out_ch: int, k: int, stride: int = 1, groups: int = 1):
    """Standard Conv2d with kernel (1, k) and same-style time padding."""
    pad = k // 2
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=(1, k),
        stride=(1, stride),
        padding=(0, pad),
        groups=groups,
        bias=False,
    )


class TimeChannelSepConv(nn.Module):
    """Time-channel separable conv: depthwise 1×k → pointwise 1×1 → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int):
        """Build the DW + PW pair."""
        super().__init__()
        self.dw = _conv1xk(in_ch, in_ch, k, groups=in_ch)
        self.pw = _conv1xk(in_ch, out_ch, k=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DW → PW → BN → ReLU."""
        return self.relu(self.bn(self.pw(self.dw(x))))


class MatchboxBlock(nn.Module):
    """One MatchboxNet block: R sub-blocks + 1×1 residual."""

    def __init__(self, in_ch: int, out_ch: int, k: int, R: int = 2):
        """Build the R-sub-block stack."""
        super().__init__()
        layers = []
        prev = in_ch
        for _ in range(R):
            layers.append(TimeChannelSepConv(prev, out_ch, k))
            prev = out_ch
        self.body = nn.Sequential(*layers)
        # Residual: 1×1 conv adapts in_ch → out_ch (also stride=1 here).
        self.skip_conv = _conv1xk(in_ch, out_ch, k=1)
        self.skip_bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """body(x) + skip(x), then ReLU."""
        return self.relu(self.body(x) + self.skip_bn(self.skip_conv(x)))


class MatchboxNet(nn.Module):
    """MatchboxNet 3×R×C — embedded KWS.

    Parameters
    ----------
    num_classes : int
        Number of command classes (12 = MLperf-style: 10 commands + silence + unknown).
    in_features : int
        Number of MFCC bins (64 = paper default; 40 also common).
    n_time : int
        Time frames (paper default ~256 for 1s @ 10ms shift).
    channels : int
        Body channel width (64 = paper default).
    R : int
        Sub-blocks per body block (paper: 2).
    block_kernels : list of int
        Per-body-block kernel sizes (paper: [13, 15, 17] for 3×R×64).
    """

    def __init__(
        self,
        num_classes: int = 12,
        in_features: int = 64,
        n_time: int = 128,
        channels: int = 64,
        R: int = 2,
        block_kernels=None,
    ):
        """Build MatchboxNet."""
        super().__init__()
        if block_kernels is None:
            block_kernels = [13, 15, 17]
        self.in_features = in_features
        self.n_time = n_time

        # Stem
        self.stem_conv = _conv1xk(in_features, channels, k=11, stride=2)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

        # Body
        body = []
        prev = channels
        for k in block_kernels:
            body.append(MatchboxBlock(prev, channels, k, R=R))
            prev = channels
        self.body = nn.Sequential(*body)

        # Tail: 1×29 → 1×1 → 1×1 (classifier head conv)
        self.tail_conv1 = _conv1xk(channels, 128, k=29)
        self.tail_bn1 = nn.BatchNorm2d(128)
        self.tail_conv2 = _conv1xk(128, 128, k=1)
        self.tail_bn2 = nn.BatchNorm2d(128)
        self.tail_conv3 = _conv1xk(128, num_classes, k=1)

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_features, 1, n_time) → (B, num_classes)."""
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.body(x)
        x = self.relu(self.tail_bn1(self.tail_conv1(x)))
        x = self.relu(self.tail_bn2(self.tail_conv2(x)))
        x = self.tail_conv3(x)
        # Global average along time → (B, num_classes, 1, 1).
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        return x


def matchboxnet(
    num_classes: int = 12,
    in_features: int = 64,
    n_time: int = 128,
    channels: int = 64,
    R: int = 2,
) -> MatchboxNet:
    """Factory for MatchboxNet 3×R×C (paper-default 3×2×64)."""
    return MatchboxNet(
        num_classes=num_classes,
        in_features=in_features,
        n_time=n_time,
        channels=channels,
        R=R,
    )
