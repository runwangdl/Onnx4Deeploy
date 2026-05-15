# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""BC-ResNet: Broadcasted Residual Learning for Efficient Keyword Spotting.

Reference: Kim et al. (2021), "Broadcasted Residual Learning for Efficient
Keyword Spotting" — INTERSPEECH 2021. Qualcomm's KWS sota using factored
frequency + temporal convolutions.

Deploy-simplified variant:
- Drop SubSpectralNorm (uses regular BN — paper notes it gives ~1% accuracy
  bump but isn't load-bearing for the architecture).
- Drop the SE block (similarly a small accuracy add-on; needs new int8
  fold support to deploy properly).
- Keep the core factored frequency-conv + temporal-conv pattern.
- Keep the residual skip path.

Input: (B, 1, F_mel, T) — typical 40 mel bins × 49 time frames @ 25ms shift.
"""

import torch
import torch.nn as nn


def _conv_freq(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
    """Frequency-only DW conv: kernel ``(k, 1)`` on the F axis."""
    pad = kernel // 2
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=(kernel, 1),
        stride=(stride, 1),
        padding=(pad, 0),
        groups=in_ch if in_ch == out_ch else 1,
        bias=False,
    )


def _conv_time(in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
    """Temporal DW conv: kernel ``(1, k)`` on the T axis."""
    pad = (kernel // 2) * dilation
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=(1, kernel),
        stride=(1, 1),
        padding=(0, pad),
        dilation=(1, dilation),
        groups=in_ch if in_ch == out_ch else 1,
        bias=False,
    )


class TransitionBlock(nn.Module):
    """Stride-2 block with channel/frequency reduction + projection residual."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, dilation: int = 1):
        """Build the transition block."""
        super().__init__()
        # 1×1 channel-mixing conv lifts to out_ch.
        self.expand = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_ch)
        # Frequency conv with stride on F axis.
        self.f_conv = _conv_freq(out_ch, out_ch, kernel=3, stride=stride)
        self.bn_f = nn.BatchNorm2d(out_ch)
        # Temporal conv (optionally dilated).
        self.t_conv = _conv_time(out_ch, out_ch, kernel=3, dilation=dilation)
        self.bn_t = nn.BatchNorm2d(out_ch)
        # Projection: 1×1 conv with same F stride for skip path.
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1), bias=False)
        self.bn_proj = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand → freq-conv → temporal-conv + skip → ReLU."""
        skip = self.bn_proj(self.proj(x))
        out = self.relu(self.bn_expand(self.expand(x)))
        out = self.relu(self.bn_f(self.f_conv(out)))
        out = self.bn_t(self.t_conv(out))
        return self.relu(out + skip)


class NormalBlock(nn.Module):
    """Stride-1 residual block with frequency + temporal convs."""

    def __init__(self, channels: int, dilation: int = 1):
        """Build the normal residual block."""
        super().__init__()
        self.f_conv = _conv_freq(channels, channels, kernel=3, stride=1)
        self.bn_f = nn.BatchNorm2d(channels)
        self.t_conv = _conv_time(channels, channels, kernel=3, dilation=dilation)
        self.bn_t = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """freq-conv → temporal-conv + residual skip."""
        identity = x
        out = self.relu(self.bn_f(self.f_conv(x)))
        out = self.bn_t(self.t_conv(out))
        return self.relu(out + identity)


class BCResNet(nn.Module):
    """
    BC-ResNet for keyword spotting.

    Parameters
    ----------
    num_classes : int
        Number of output classes (12 for MLperf Tiny KWS).
    n_mel : int
        Number of mel bins (40 = paper default).
    n_time : int
        Time dimension (49 for 1s @ 25ms shift).
    base_channels : int
        Stem-conv channel width (8 for BC-ResNet-1, 24 for BC-ResNet-3).
    block_counts : list of int
        Number of normal blocks per stage (paper default: [2, 2, 4, 4]).
    block_dilations : list of int
        Temporal-conv dilation per stage. The standard BC-ResNet uses
        increasing dilation [1, 2, 4, 8] for receptive-field growth.
    """

    def __init__(
        self,
        num_classes: int = 12,
        n_mel: int = 40,
        n_time: int = 49,
        base_channels: int = 8,
        block_counts=None,
        block_dilations=None,
    ):
        """Build BC-ResNet."""
        super().__init__()
        if block_counts is None:
            block_counts = [2, 2, 4, 4]
        if block_dilations is None:
            block_dilations = [1, 2, 4, 8]
        assert len(block_counts) == len(block_dilations)
        self.n_mel = n_mel
        self.n_time = n_time

        # Stem: 5×5 conv with stride (2, 1) — halves F, keeps T.
        self.stem_conv = nn.Conv2d(
            1, base_channels, kernel_size=5, stride=(2, 1), padding=2, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=False)

        # Stages: TransitionBlock + (N-1) × NormalBlock.
        stages = []
        prev = base_channels
        for stage_idx, (n_blocks, dilation) in enumerate(zip(block_counts, block_dilations)):
            stage_ch = base_channels * (stage_idx + 1)
            stride = 2 if stage_idx > 0 else 1  # first stage uses stride 1
            stages.append(TransitionBlock(prev, stage_ch, stride=stride, dilation=dilation))
            for _ in range(n_blocks - 1):
                stages.append(NormalBlock(stage_ch, dilation=dilation))
            prev = stage_ch
        self.stages = nn.Sequential(*stages)
        self.last_channel = prev

        # Head: 1×1 conv classifier with global avg over (F, T).
        self.head_conv = nn.Conv2d(self.last_channel, num_classes, kernel_size=1, bias=True)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, n_mel, n_time) → (B, num_classes)."""
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.stages(x)
        x = self.head_conv(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return self.flatten(x)


def bc_resnet1(num_classes: int = 12, n_mel: int = 40, n_time: int = 49) -> BCResNet:
    """BC-ResNet-1 — base_channels=8, smallest variant (~6K params)."""
    return BCResNet(
        num_classes=num_classes,
        n_mel=n_mel,
        n_time=n_time,
        base_channels=8,
        block_counts=[2, 2, 4, 4],
        block_dilations=[1, 2, 4, 8],
    )


def bc_resnet3(num_classes: int = 12, n_mel: int = 40, n_time: int = 49) -> BCResNet:
    """BC-ResNet-3 — base_channels=24, ~50K params, KWS sota at < 100K."""
    return BCResNet(
        num_classes=num_classes,
        n_mel=n_mel,
        n_time=n_time,
        base_channels=24,
        block_counts=[2, 2, 4, 4],
        block_dilations=[1, 2, 4, 8],
    )
