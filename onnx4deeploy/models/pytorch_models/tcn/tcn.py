# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TCN — Temporal Convolutional Network for sequence modeling.

Reference: Bai et al. (2018), "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling." The modern
replacement for LSTMs on most embedded sensor / HAR / ECG tasks: dilated
1D convolutions with residual connections, no recurrence, all-feed-forward.

Deploy-simplified variant:
- Drop WeightNorm (training-only normalization; removed at inference).
- Use symmetric padding (instead of strict-causal left-pad) so the lowered
  ONNX uses standard Conv2D with kernel ``(1, k)`` and padding ``(0, p)``
  — Deeploy's PULPConv2D parser handles this cleanly with ``dilation>1``.
- Layout: ``(B, in_channels, 1, n_time)`` so all convs are 2D.
"""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """One TCN residual block: 2 dilated convs + skip 1×1."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
    ):
        """Build the temporal block."""
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            dilation=(1, dilation),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, pad),
            dilation=(1, dilation),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

        # Residual skip: 1×1 if channels or stride differ, else identity.
        self.use_skip = in_ch != out_ch or stride != 1
        if self.use_skip:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(1, stride), bias=False)
            self.skip_bn = nn.BatchNorm2d(out_ch)
        else:
            self.skip_conv = None
            self.skip_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two dilated convs + residual."""
        identity = x if not self.use_skip else self.skip_bn(self.skip_conv(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class TCN(nn.Module):
    """
    Temporal Convolutional Network.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    in_channels : int
        Number of input feature channels (sensor count, MFCC bins, etc.).
    n_time : int
        Input time-series length.
    channels : list of int
        Per-block channel widths. Default ``[16, 32, 64]`` = 3 stages.
    kernel_size : int
        Conv kernel size along the time axis (paper: 3 or 7).
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 9,
        n_time: int = 128,
        channels=None,
        kernel_size: int = 3,
    ):
        """Build the TCN."""
        super().__init__()
        if channels is None:
            channels = [16, 32, 64]
        self.n_time = n_time
        self.in_channels = in_channels

        # Dilation doubles each block (1, 2, 4, ...) — the receptive-field-
        # growth knob that lets TCN match LSTM's effective context.
        blocks = []
        prev = in_channels
        for i, ch in enumerate(channels):
            dilation = 2**i
            blocks.append(TemporalBlock(prev, ch, kernel_size=kernel_size, dilation=dilation))
            prev = ch
        self.blocks = nn.Sequential(*blocks)

        self.head_conv = nn.Conv2d(prev, num_classes, kernel_size=1, bias=True)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, 1, n_time) → (B, num_classes)."""
        x = self.blocks(x)
        x = self.head_conv(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        return self.flatten(x)


def tcn_har(num_classes: int = 6, in_channels: int = 9, n_time: int = 128) -> TCN:
    """TCN sized for Human Activity Recognition (UCI-HAR / WISDM)."""
    return TCN(
        num_classes=num_classes,
        in_channels=in_channels,
        n_time=n_time,
        channels=[16, 32, 64],
        kernel_size=3,
    )


def tcn_ecg(num_classes: int = 5, in_channels: int = 1, n_time: int = 256) -> TCN:
    """TCN sized for single-lead ECG classification."""
    return TCN(
        num_classes=num_classes,
        in_channels=in_channels,
        n_time=n_time,
        channels=[8, 16, 32, 64],
        kernel_size=7,
    )
