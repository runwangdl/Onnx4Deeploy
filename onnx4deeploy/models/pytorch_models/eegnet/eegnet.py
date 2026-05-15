# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""EEGNet: Compact CNN for EEG-based brain-computer interfaces.

Reference: Lawhern et al. (2018), "EEGNet: A Compact Convolutional Neural
Network for EEG-based Brain-Computer Interfaces."

The canonical embedded-BCI baseline. ~2 K params at default settings —
fits trivially on any PULP target.

Deployment-friendly substitutions vs. the paper:
- ELU → ReLU (Deeploy backends ship ReLU; ELU would need a custom op).
  This is the standard embedded-EEG variant (see Brain-CNN / TinyEEG).
- AvgPool2D → ``torch.mean`` over the time axis (lowers to ReduceMean).

Layout: (B, 1, Channels, Time) — height = EEG electrode count, width = time
samples. All convs are 2D so Deeploy's PULPConv2D parser handles them.
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet-8,2 default (F1=8, D=2, F2=F1*D=16).

    Parameters
    ----------
    num_classes : int
        Number of motor-imagery / event classes (e.g. 2 for BCI Competition IV-2b).
    n_channels : int
        Number of EEG electrodes (8 for typical embedded BCI, 22 for IV-2a).
    n_samples : int
        Number of time samples (128 = 0.5s @ 250Hz).
    F1 : int
        Number of temporal filters in block 1.
    D : int
        Depth multiplier (depthwise conv filter count per F1).
    F2 : int or None
        Separable conv filter count; defaults to F1*D.
    kernel_time : int
        Block-1 temporal kernel size (paper: ``n_samples // 2``; smaller for embedded).
    dropout : float
        Currently unused at export time (kept for training-mode parity).
    """

    def __init__(
        self,
        num_classes: int = 2,
        n_channels: int = 8,
        n_samples: int = 128,
        F1: int = 8,
        D: int = 2,
        F2: int = None,
        kernel_time: int = 64,
        dropout: float = 0.5,
    ):
        """Build EEGNet."""
        super().__init__()
        if F2 is None:
            F2 = F1 * D
        self.n_channels = n_channels
        self.n_samples = n_samples
        self._F1 = F1
        self._D = D
        self._F2 = F2

        pad_time = kernel_time // 2

        # ── Block 1: temporal conv + depthwise spatial conv + pool ────────
        self.temporal_conv = nn.Conv2d(
            1, F1, kernel_size=(1, kernel_time), padding=(0, pad_time), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise spatial conv: kernel covers all electrodes at once.
        self.depthwise_conv = nn.Conv2d(
            F1,
            F1 * D,
            kernel_size=(n_channels, 1),
            padding=0,
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.relu = nn.ReLU(inplace=False)
        # Block-1 pool reduces time by 4.
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # ── Block 2: separable conv (depthwise + pointwise) + pool ────────
        self.sep_depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        # Block-2 pool reduces time by 8.
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        # ── Classifier ────────────────────────────────────────────────────
        # After two avgpools: time = n_samples / 4 / 8 = n_samples / 32.
        time_after = (n_samples // 4) // 8
        self._flat_features = F2 * 1 * time_after  # height collapsed to 1 in block-1
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(self._flat_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, Channels, Time) → (B, num_classes)."""
        # Block 1
        x = self.bn1(self.temporal_conv(x))
        x = self.relu(self.bn2(self.depthwise_conv(x)))
        x = self.pool1(x)
        # Block 2
        x = self.sep_pointwise(self.sep_depthwise(x))
        x = self.relu(self.bn3(x))
        x = self.pool2(x)
        # Classifier
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def eegnet(
    num_classes: int = 2,
    n_channels: int = 8,
    n_samples: int = 128,
    F1: int = 8,
    D: int = 2,
    kernel_time: int = 64,
) -> EEGNet:
    """Factory for EEGNet-8,2 (default)."""
    return EEGNet(
        num_classes=num_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        F1=F1,
        D=D,
        kernel_time=kernel_time,
    )
