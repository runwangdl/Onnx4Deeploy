# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Brevitas-quantized EEGNet for embedded BCI deployments.

Mirrors ``eegnet.EEGNet`` with Brevitas int8 substitutions. The two
AvgPool2D layers are replaced with ``torch.mean`` (lowers to ReduceMean,
which Deeploy's PULP backend supports), wrapped in QuantIdentity so the
running int8 scale survives the pool.
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


class QuantEEGNet(nn.Module):
    """Brevitas-quantized EEGNet-8,2."""

    def __init__(
        self,
        num_classes: int = 2,
        n_channels: int = 8,
        n_samples: int = 128,
        F1: int = 8,
        D: int = 2,
        F2: int = None,
        kernel_time: int = 64,
    ):
        """Build the quantized EEGNet."""
        super().__init__()
        if F2 is None:
            F2 = F1 * D
        self.n_channels = n_channels
        self.n_samples = n_samples

        pad_time = kernel_time // 2

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        # Block 1
        self.temporal_conv = qnn.QuantConv2d(
            1, F1, kernel_size=(1, kernel_time), padding=(0, pad_time), bias=True, **_QUANT_KW
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = qnn.QuantConv2d(
            F1,
            F1 * D,
            kernel_size=(n_channels, 1),
            padding=0,
            groups=F1,
            bias=True,
            **_QUANT_KW,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.relu_b1 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        # Use AvgPool2D — exports as ONNX AveragePool which Deeploy's PULP
        # AveragePool2DParser handles. (ReduceMean only does global average.)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.pool1_dq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

        # Block 2
        self.sep_depthwise = qnn.QuantConv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=True,
            **_QUANT_KW,
        )
        self.sep_pointwise = qnn.QuantConv2d(F1 * D, F2, kernel_size=1, bias=True, **_QUANT_KW)
        self.bn3 = nn.BatchNorm2d(F2)
        self.relu_b2 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.pool2_dq = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=False
        )

        time_after = (n_samples // 4) // 8
        self._flat_features = F2 * 1 * time_after

        self.flatten = nn.Flatten(start_dim=1)
        self.fc_iq = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)
        self.classifier = qnn.QuantLinear(
            self._flat_features,
            num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, Channels, Time) → (B, num_classes)."""
        x = self.input_quant(x)
        # Block 1
        x = self.bn1(self.temporal_conv(x))
        x = self.relu_b1(self.bn2(self.depthwise_conv(x)))
        x = self.pool1_dq(self.pool1(x))
        # Block 2
        x = self.sep_pointwise(self.sep_depthwise(x))
        x = self.relu_b2(self.bn3(x))
        x = self.pool2_dq(self.pool2(x))
        # Classifier
        x = self.flatten(x)
        x = self.fc_iq(x)
        x = self.classifier(x)
        return x


def quant_eegnet(
    num_classes: int = 2,
    n_channels: int = 8,
    n_samples: int = 128,
    F1: int = 8,
    D: int = 2,
    kernel_time: int = 64,
) -> QuantEEGNet:
    """Brevitas-quantized EEGNet factory."""
    return QuantEEGNet(
        num_classes=num_classes,
        n_channels=n_channels,
        n_samples=n_samples,
        F1=F1,
        D=D,
        kernel_time=kernel_time,
    )
