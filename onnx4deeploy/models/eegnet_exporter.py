# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""EEGNet Model Exporter — EEG / BCI baseline.

EEGNet (Lawhern et al. 2018) is the canonical embedded BCI baseline.
The deploy-friendly variant here uses ReLU (instead of paper ELU) and
``torch.mean`` (instead of AvgPool2D) so the graph lowers cleanly through
Deeploy's PULP backend.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter


class EEGNetExporter(BaseONNXExporter):
    """ONNX exporter for EEGNet."""

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """Initialize the exporter."""
        super().__init__(save_path, config_file)
        self.model_config = {}

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def load_config(self) -> Dict[str, Any]:
        """Default config."""
        config = {
            "batch_size": 1,
            "n_channels": 8,  # EEG electrode count (8 = standard embedded BCI)
            "n_samples": 128,  # Time samples (128 = 0.5s @ 250Hz)
            "num_classes": 2,  # Binary motor-imagery
            "F1": 8,
            "D": 2,
            "kernel_time": 64,
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
        """Build the FP32 EEGNet."""
        from .pytorch_models.eegnet import eegnet

        return eegnet(
            num_classes=self.model_config["num_classes"],
            n_channels=self.model_config["n_channels"],
            n_samples=self.model_config["n_samples"],
            F1=self.model_config["F1"],
            D=self.model_config["D"],
            kernel_time=self.model_config["kernel_time"],
        )

    # ------------------------------------------------------------------ #
    # Brevitas-quantized factory                                           #
    # ------------------------------------------------------------------ #

    def create_brevitas_model(self) -> torch.nn.Module:
        """Build the Brevitas-quantized EEGNet."""
        from .pytorch_models.eegnet import quant_eegnet

        return quant_eegnet(
            num_classes=self.model_config["num_classes"],
            n_channels=self.model_config["n_channels"],
            n_samples=self.model_config["n_samples"],
            F1=self.model_config["F1"],
            D=self.model_config["D"],
            kernel_time=self.model_config["kernel_time"],
        )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the (B, 1, n_channels, n_samples) input shape."""
        return (
            self.config["batch_size"],
            1,
            self.config["n_channels"],
            self.config["n_samples"],
        )

    def _get_config_string(self) -> str:
        return (
            f"_eegnet_{self.config['n_channels']}x{self.config['n_samples']}"
            f"_F{self.config['F1']}D{self.config['D']}_{self.config['num_classes']}"
        )

    # ------------------------------------------------------------------ #
    # Inference test data                                                  #
    # ------------------------------------------------------------------ #

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
