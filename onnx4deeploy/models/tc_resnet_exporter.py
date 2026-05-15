# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""TC-ResNet Model Exporter — Keyword Spotting (KWS) benchmark.

TC-ResNet (Choi et al. 2019) is the temporal-conv KWS baseline that
replaced DS-CNN in many embedded SDKs. Input is shaped (B, n_mfcc, 1,
n_time) — the spatial height is fixed to 1 so Deeploy's 2D conv parser
can handle the 1×k temporal kernels.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter


class TCResNetExporter(BaseONNXExporter):
    """ONNX exporter for TC-ResNet (KWS)."""

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
            "n_mfcc": 40,  # MFCC feature bins (40 = paper default)
            "n_time": 49,  # 1-second @ 25-ms shift
            "num_classes": 12,  # MLperf-style: 10 commands + silence + unknown
            "variant": "tc_resnet8",  # "tc_resnet8" | "tc_resnet14"
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
        """Build the FP32 TC-ResNet."""
        from .pytorch_models.tc_resnet import tc_resnet8, tc_resnet14

        variant = self.model_config.get("variant", "tc_resnet8")
        factory = {"tc_resnet8": tc_resnet8, "tc_resnet14": tc_resnet14}[variant]
        return factory(
            num_classes=self.model_config["num_classes"],
            n_mfcc=self.model_config["n_mfcc"],
            n_time=self.model_config["n_time"],
        )

    # ------------------------------------------------------------------ #
    # Brevitas-quantized factory (for `-mode quant`)                       #
    # ------------------------------------------------------------------ #

    def create_brevitas_model(self) -> torch.nn.Module:
        """Build the Brevitas-quantized TC-ResNet for ``-mode quant``."""
        from .pytorch_models.tc_resnet import quant_tc_resnet8, quant_tc_resnet14

        variant = self.model_config.get("variant", "tc_resnet8")
        factory = {"tc_resnet8": quant_tc_resnet8, "tc_resnet14": quant_tc_resnet14}[variant]
        return factory(
            num_classes=self.model_config["num_classes"],
            n_mfcc=self.model_config["n_mfcc"],
            n_time=self.model_config["n_time"],
        )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the (B, n_mfcc, 1, n_time) input shape."""
        return (
            self.config["batch_size"],
            self.config["n_mfcc"],
            1,
            self.config["n_time"],
        )

    def _get_config_string(self) -> str:
        v = self.config.get("variant", "tc_resnet8")
        return f"_{v}_{self.config['n_mfcc']}x{self.config['n_time']}_{self.config['num_classes']}"

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
