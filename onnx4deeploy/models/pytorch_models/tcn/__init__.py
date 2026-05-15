# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TCN (Bai 2018) models for ONNX export."""

from .tcn import TCN, TemporalBlock, tcn_ecg, tcn_har

try:
    from .tcn_quant import QuantTCN, QuantTemporalBlock, quant_tcn_ecg, quant_tcn_har

    __all__ = [
        "TCN",
        "TemporalBlock",
        "tcn_ecg",
        "tcn_har",
        "QuantTCN",
        "QuantTemporalBlock",
        "quant_tcn_ecg",
        "quant_tcn_har",
    ]
except ImportError:
    __all__ = ["TCN", "TemporalBlock", "tcn_ecg", "tcn_har"]
