# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNetV1 Model Exporter — inference + quantized export support.

MobileNetV1 (Howard et al. 2017) is the original mobile-vision baseline.
Many embedded MCU SDKs still ship it as their reference for Visual Wake
Words / image classification because of its predictable per-layer FLOP
budget.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter
from .pytorch_models.mobilenet import mobilenet_v1


class MobileNetV1Exporter(BaseONNXExporter):
    """ONNX exporter for MobileNetV1."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """Initialize the exporter."""
        super().__init__(save_path, config_file)
        self.model_config = {}

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def load_config(self) -> Dict[str, Any]:
        """Default config + variant resolution."""
        config = {
            "batch_size": 1,
            "img_size": 224,
            "input_channels": 3,
            "num_classes": 1000,
            "width_mult": 1.0,  # 0.25 / 0.5 / 0.75 / 1.0 — MLperf-tiny-style: 0.25 @ 96
            "opset_version": 17,
        }

        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        self.model_config = config
        return config

    # ------------------------------------------------------------------ #
    # Model factory                                                        #
    # ------------------------------------------------------------------ #

    def create_model(self) -> torch.nn.Module:
        """Build the FP32 MobileNetV1."""
        return mobilenet_v1(
            num_classes=self.model_config["num_classes"],
            width_mult=self.model_config["width_mult"],
            input_channels=self.model_config["input_channels"],
        )

    # ------------------------------------------------------------------ #
    # Brevitas-quantized factory (for `-mode quant`)                       #
    # ------------------------------------------------------------------ #

    def create_brevitas_model(self) -> torch.nn.Module:
        """Build the Brevitas-quantized MobileNetV1 for ``-mode quant``."""
        from .pytorch_models.mobilenet import quant_mobilenet_v1

        return quant_mobilenet_v1(
            num_classes=self.model_config["num_classes"],
            width_mult=self.model_config["width_mult"],
            input_channels=self.model_config["input_channels"],
        )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the (B, C, H, W) input shape."""
        return (
            self.config["batch_size"],
            self.config["input_channels"],
            self.config["img_size"],
            self.config["img_size"],
        )

    def _get_config_string(self) -> str:
        width = self.config["width_mult"]
        return f"_mobilenetv1_{width}_{self.config['img_size']}_{self.config['num_classes']}"

    # ------------------------------------------------------------------ #
    # Inference test data                                                  #
    # ------------------------------------------------------------------ #

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """Save random inputs.npz + model(input) outputs.npz for golden testing."""
        print("💾 Saving inference test data...")
        input_shape = self.get_input_shape()
        test_input = np.random.randn(*input_shape).astype(np.float32)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            test_output = model(torch.from_numpy(test_input)).numpy()
        if was_training:
            model.train()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)
        print(f"   Input: {test_input.shape}  Output: {test_output.shape}")
