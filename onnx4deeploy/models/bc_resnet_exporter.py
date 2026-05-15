# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""BC-ResNet Model Exporter — KWS sota under 100K params.

BC-ResNet (Kim et al. 2021) uses factored frequency + temporal convolutions
plus dilated temporal kernels for receptive-field growth. Deploy-simplified
variant here drops SubSpectralNorm and SE (small accuracy adds, big infra
cost) but keeps the core factored-conv structure.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ..core.base_exporter import BaseONNXExporter


class BCResNetExporter(BaseONNXExporter):
    """ONNX exporter for BC-ResNet."""

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
            "n_mel": 40,
            "n_time": 49,
            "num_classes": 12,
            "variant": "bc_resnet1",  # "bc_resnet1" | "bc_resnet3"
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
        """Build the FP32 BC-ResNet."""
        from .pytorch_models.bc_resnet import bc_resnet1, bc_resnet3

        variant = self.model_config.get("variant", "bc_resnet1")
        factory = {"bc_resnet1": bc_resnet1, "bc_resnet3": bc_resnet3}[variant]
        return factory(
            num_classes=self.model_config["num_classes"],
            n_mel=self.model_config["n_mel"],
            n_time=self.model_config["n_time"],
        )

    # ------------------------------------------------------------------ #
    # Brevitas-quantized factory                                           #
    # ------------------------------------------------------------------ #

    def create_brevitas_model(self) -> torch.nn.Module:
        """Build the Brevitas-quantized BC-ResNet."""
        from .pytorch_models.bc_resnet import quant_bc_resnet1, quant_bc_resnet3

        variant = self.model_config.get("variant", "bc_resnet1")
        factory = {"bc_resnet1": quant_bc_resnet1, "bc_resnet3": quant_bc_resnet3}[variant]
        return factory(
            num_classes=self.model_config["num_classes"],
            n_mel=self.model_config["n_mel"],
            n_time=self.model_config["n_time"],
        )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the (B, 1, n_mel, n_time) input shape."""
        return (self.config["batch_size"], 1, self.config["n_mel"], self.config["n_time"])

    def _get_config_string(self) -> str:
        v = self.config.get("variant", "bc_resnet1")
        return f"_{v}_{self.config['n_mel']}x{self.config['n_time']}_{self.config['num_classes']}"

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
