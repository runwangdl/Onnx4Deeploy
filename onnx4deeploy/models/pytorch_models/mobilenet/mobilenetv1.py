# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNetV1 Model for ONNX Export.

Based on "MobileNets: Efficient Convolutional Neural Networks for Mobile
Vision Applications" — Howard et al. (2017).

The original embedded-vision baseline. Pure depthwise-separable convs
(no inverted residuals, no SE, no expand). Many industry MCU SDKs still
ship MobileNetV1 as their reference for VWW / image classification.

This implementation is optimized for ONNX export with clean computation
graphs (no inplace ReLU, no AdaptiveAvgPool — ``torch.mean`` is used so
Deeploy's PULP backend can lower to ReduceMean).
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Standard MobileNetV1 building block: 3×3 DW + 1×1 PW, BN + ReLU after each."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        """Initialize the DW+PW pair."""
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DW → BN → ReLU → PW → BN → ReLU."""
        x = self.relu(self.bn_dw(self.dw(x)))
        x = self.relu(self.bn_pw(self.pw(x)))
        return x


class MobileNetV1(nn.Module):
    """
    MobileNetV1 architecture.

    Stride / channel schedule follows the original paper at width_mult=1.0:
      Stem  (3,32,s=2) → DS(32,64) → DS(64,128,s=2) → DS(128,128) →
      DS(128,256,s=2) → DS(256,256) → DS(256,512,s=2) →
      5× DS(512,512) → DS(512,1024,s=2) → DS(1024,1024) →
      GAP → FC
    """

    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, input_channels: int = 3):
        """Build the MobileNetV1 with optional width multiplier."""
        super().__init__()

        def c(ch: int) -> int:
            return int(ch * width_mult)

        # (out_channels_at_width_1.0, stride)
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

        # Stem: standard 3×3 stride-2 conv
        self.stem_conv = nn.Conv2d(input_channels, c(32), kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(c(32))
        self.relu = nn.ReLU(inplace=False)

        # Depthwise-separable stack
        in_ch = c(32)
        blocks = []
        for out_ch, stride in cfg:
            blocks.append(DepthwiseSeparableConv(in_ch, c(out_ch), stride=stride))
            in_ch = c(out_ch)
        self.ds_blocks = nn.Sequential(*blocks)

        self.last_channel = in_ch
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize Conv2d / BatchNorm2d / Linear with sensible defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MobileNetV1."""
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.ds_blocks(x)
        # Global avg via ReduceMean (Deeploy-supported), not AdaptiveAvgPool.
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def mobilenet_v1(
    num_classes: int = 1000, width_mult: float = 1.0, input_channels: int = 3
) -> MobileNetV1:
    """Factory for MobileNetV1 (Howard et al. 2017)."""
    return MobileNetV1(
        num_classes=num_classes, width_mult=width_mult, input_channels=input_channels
    )
