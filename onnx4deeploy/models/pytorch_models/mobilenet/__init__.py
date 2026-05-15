# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNet models for ONNX export."""

from .mobilenetv1 import MobileNetV1, mobilenet_v1
from .mobilenetv2 import MobileNetV2, mobilenet_v2

# Brevitas-quantized variants. Imported lazily so that environments without
# brevitas don't fail at package import time.
try:
    from .mobilenetv1_quant import QuantMobileNetV1, quant_mobilenet_v1
    from .mobilenetv2_quant import QuantMobileNetV2, quant_mobilenet_v2

    __all__ = [
        "MobileNetV1",
        "mobilenet_v1",
        "MobileNetV2",
        "mobilenet_v2",
        "QuantMobileNetV1",
        "quant_mobilenet_v1",
        "QuantMobileNetV2",
        "quant_mobilenet_v2",
    ]
except ImportError:
    __all__ = ["MobileNetV1", "mobilenet_v1", "MobileNetV2", "mobilenet_v2"]
