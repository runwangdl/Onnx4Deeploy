# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TCN Model Exporter — sensor/time-series classification.

Default config is sized for UCI-HAR (Human Activity Recognition):
  Input (B, 9, 1, 128) — 9 accelerometer/gyroscope channels, 128 samples.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter


class TCNExporter(BaseONNXExporter):
    """ONNX exporter for TCN."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """Initialize the exporter."""
        super().__init__(save_path, config_file)
        self.model_config = {}

    def load_config(self) -> Dict[str, Any]:
        """Default config."""
        config = {
            "batch_size": 1,
            "in_channels": 9,  # UCI-HAR default (3-axis accel + gyro + body-accel)
            "n_time": 128,
            "num_classes": 6,  # UCI-HAR activity classes
            "variant": "har",  # "har" | "ecg"
            "kernel_size": 3,
            "opset_version": 17,
        }
        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)
        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        """Build the FP32 TCN."""
        from .pytorch_models.tcn import tcn_ecg, tcn_har

        variant = self.model_config.get("variant", "har")
        if variant == "ecg":
            return tcn_ecg(
                num_classes=self.model_config["num_classes"],
                in_channels=self.model_config["in_channels"],
                n_time=self.model_config["n_time"],
            )
        return tcn_har(
            num_classes=self.model_config["num_classes"],
            in_channels=self.model_config["in_channels"],
            n_time=self.model_config["n_time"],
        )

    def create_brevitas_model(self) -> torch.nn.Module:
        """Build the Brevitas-quantized TCN."""
        from .pytorch_models.tcn import quant_tcn_ecg, quant_tcn_har

        variant = self.model_config.get("variant", "har")
        if variant == "ecg":
            return quant_tcn_ecg(
                num_classes=self.model_config["num_classes"],
                in_channels=self.model_config["in_channels"],
                n_time=self.model_config["n_time"],
            )
        return quant_tcn_har(
            num_classes=self.model_config["num_classes"],
            in_channels=self.model_config["in_channels"],
            n_time=self.model_config["n_time"],
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the (B, in_channels, 1, n_time) input shape."""
        return (
            self.config["batch_size"],
            self.config["in_channels"],
            1,
            self.config["n_time"],
        )

    def _get_config_string(self) -> str:
        v = self.config.get("variant", "har")
        return (
            f"_tcn_{v}_{self.config['in_channels']}x{self.config['n_time']}"
            f"_k{self.config['kernel_size']}_{self.config['num_classes']}"
        )

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """Save random inputs.npz + model(input) outputs.npz."""
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
