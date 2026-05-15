# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TC-ResNet: Temporal Convolutional ResNet for Keyword Spotting.

Reference: Choi et al. (2019), "Temporal Convolution for Real-time Keyword
Spotting on Mobile Devices" — INTERSPEECH 2019. The canonical embedded-KWS
baseline that replaces DS-CNN with 1D temporal residual blocks.

Implementation note (Deeploy-friendly layout):
  Input is shaped (B, C, 1, T) — features go down the channel axis, time
  on the last axis, height fixed to 1. All convs are 2D with kernel
  ``(1, k)`` so they lower cleanly through Deeploy's PULPConv2D parser
  rather than the rarely-supported Conv1D path.
"""

import torch
import torch.nn as nn


class TCResBlock(nn.Module):
    """TC-ResNet basic residual block (two 1×k temporal convs + skip add)."""

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
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel),
            stride=(1, stride),
            padding=(0, pad),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel),
            stride=(1, 1),
            padding=(0, pad),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two 1×k convs with residual skip."""
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class _TCDownsample(nn.Module):
    """1×1 stride-S downsample on the time axis."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """Build the 1×1 projection."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=(1, stride), bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1×1 conv + BN."""
        return self.bn(self.conv(x))


class TCResNet(nn.Module):
    """
    TC-ResNet for keyword spotting.

    Parameters
    ----------
    num_classes : int
        Number of output classes (12 for MLperf Tiny KWS).
    n_mfcc : int
        Number of MFCC feature bins per time step (40 = standard).
    n_time : int
        Time dimension (49 for 1s @ 25ms shift, 98 for 50ms shift).
    base_channels : int
        Stem-conv channel width (16 for TC-ResNet8, 24 for larger variants).
    stage_channels : list of int
        Per-stage out_channel widths (3 stages = TC-ResNet8, 6 stages = 14).
    kernel : int
        Temporal kernel size (9 = paper default).
    """

    def __init__(
        self,
        num_classes: int = 12,
        n_mfcc: int = 40,
        n_time: int = 49,
        base_channels: int = 16,
        stage_channels=None,
        kernel: int = 9,
    ):
        """Build TC-ResNet."""
        super().__init__()
        if stage_channels is None:
            stage_channels = [24, 32, 48]
        self.n_mfcc = n_mfcc
        self.n_time = n_time
        self.kernel = kernel

        # Stem: 1×k conv mapping MFCC channels → base_channels
        pad = kernel // 2
        self.stem_conv = nn.Conv2d(
            n_mfcc,
            base_channels,
            kernel_size=(1, kernel),
            stride=(1, 1),
            padding=(0, pad),
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=False)

        # Residual stages — first block of each stage strides by 2 along time.
        stages = []
        in_ch = base_channels
        for out_ch in stage_channels:
            down = _TCDownsample(in_ch, out_ch, stride=2)
            stages.append(TCResBlock(in_ch, out_ch, kernel, stride=2, downsample=down))
            stages.append(TCResBlock(out_ch, out_ch, kernel, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_mfcc, 1, n_time) → (B, num_classes)."""
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.stages(x)
        # Global average along time → (B, C, 1, 1).
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def tc_resnet8(num_classes: int = 12, n_mfcc: int = 40, n_time: int = 49) -> TCResNet:
    """TC-ResNet8 — 3 stages × 2 blocks (~66 K params at standard width)."""
    return TCResNet(
        num_classes=num_classes,
        n_mfcc=n_mfcc,
        n_time=n_time,
        base_channels=16,
        stage_channels=[24, 32, 48],
        kernel=9,
    )


def tc_resnet14(num_classes: int = 12, n_mfcc: int = 40, n_time: int = 49) -> TCResNet:
    """TC-ResNet14 — 6 stages × 2 blocks (~310 K params)."""
    return TCResNet(
        num_classes=num_classes,
        n_mfcc=n_mfcc,
        n_time=n_time,
        base_channels=16,
        stage_channels=[24, 24, 32, 32, 48, 48],
        kernel=9,
    )
