# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MatchboxNet Model Exporter — NVIDIA's low-latency edge KWS.

MatchboxNet (Majumdar & Ginsburg 2020) is a 1D time-channel-separable
conv keyword spotter. Input layout is (B, MFCC_bins, 1, Time) so all
convs lower through Deeploy's PULPConv2D parser.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter


class MatchboxNetExporter(BaseONNXExporter):
    """ONNX exporter for MatchboxNet."""

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
            "in_features": 64,  # MFCC bins (64 = paper default)
            "n_time": 128,  # Time frames (paper used ~256; smaller for embedded)
            "channels": 64,  # Body channel width
            "R": 2,  # Sub-blocks per body block
            "num_classes": 12,  # MLperf-style: 10 commands + silence + unknown
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
        """Build the FP32 MatchboxNet."""
        from .pytorch_models.matchboxnet import matchboxnet

        return matchboxnet(
            num_classes=self.model_config["num_classes"],
            in_features=self.model_config["in_features"],
            n_time=self.model_config["n_time"],
            channels=self.model_config["channels"],
            R=self.model_config["R"],
        )

    # ------------------------------------------------------------------ #
    # Brevitas-quantized factory                                           #
    # ------------------------------------------------------------------ #

    def create_brevitas_model(self) -> torch.nn.Module:
        """Build the Brevitas-quantized MatchboxNet."""
        from .pytorch_models.matchboxnet import quant_matchboxnet

        return quant_matchboxnet(
            num_classes=self.model_config["num_classes"],
            in_features=self.model_config["in_features"],
            n_time=self.model_config["n_time"],
            channels=self.model_config["channels"],
            R=self.model_config["R"],
        )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the (B, in_features, 1, n_time) input shape."""
        return (
            self.config["batch_size"],
            self.config["in_features"],
            1,
            self.config["n_time"],
        )

    def _get_config_string(self) -> str:
        return (
            f"_matchboxnet_{self.config['in_features']}x{self.config['n_time']}"
            f"_C{self.config['channels']}R{self.config['R']}_{self.config['num_classes']}"
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
