# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MatchboxNet (Majumdar & Ginsburg 2020) models for ONNX export."""

from .matchboxnet import MatchboxBlock, MatchboxNet, TimeChannelSepConv, matchboxnet

try:
    from .matchboxnet_quant import (
        QuantMatchboxBlock,
        QuantMatchboxNet,
        QuantTimeChannelSepConv,
        quant_matchboxnet,
    )

    __all__ = [
        "MatchboxBlock",
        "MatchboxNet",
        "TimeChannelSepConv",
        "matchboxnet",
        "QuantMatchboxBlock",
        "QuantMatchboxNet",
        "QuantTimeChannelSepConv",
        "quant_matchboxnet",
    ]
except ImportError:
    __all__ = ["MatchboxBlock", "MatchboxNet", "TimeChannelSepConv", "matchboxnet"]
