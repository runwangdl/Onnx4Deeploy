# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""EEGNet (Lawhern 2018) models for ONNX export."""

from .eegnet import EEGNet, eegnet

try:
    from .eegnet_quant import QuantEEGNet, quant_eegnet

    __all__ = ["EEGNet", "eegnet", "QuantEEGNet", "quant_eegnet"]
except ImportError:
    __all__ = ["EEGNet", "eegnet"]
