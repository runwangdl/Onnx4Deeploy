# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNetV1 Model for ONNX Export.

Based on "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications" - Howard et al. (2017)

Primarily targeted at the MLPerf Tiny Visual Wake Words benchmark (alpha=0.25,
96x96 RGB input, 2 classes). Standard depthwise-separable convolution stack.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 -> BN -> ReLU -> Pointwise 1x1 -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.dw(x)))
        x = self.relu(self.bn_pw(self.pw(x)))
        return x


class MobileNetV1(nn.Module):
    """MobileNetV1 with configurable width multiplier (alpha)."""

    # (out_channels, stride) for each DSConv block assuming alpha=1.0.
    _BLOCK_SPEC: List[Tuple[int, int]] = [
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

    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 0.25,
        input_channels: int = 3,
    ):
        super().__init__()

        def c(channels: int) -> int:
            return max(8, int(channels * width_mult))

        stem_out = c(32)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=False),
        )

        blocks = []
        prev = stem_out
        for out_ch, stride in self._BLOCK_SPEC:
            blocks.append(DepthwiseSeparableConv(prev, c(out_ch), stride))
            prev = c(out_ch)
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v1(
    num_classes: int = 2,
    width_mult: float = 0.25,
    input_channels: int = 3,
) -> MobileNetV1:
    return MobileNetV1(
        num_classes=num_classes,
        width_mult=width_mult,
        input_channels=input_channels,
    )
