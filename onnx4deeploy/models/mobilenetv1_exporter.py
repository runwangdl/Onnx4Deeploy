# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""MobileNetV1 Model Exporter - inference + training graph support."""

from typing import Any, Dict, Tuple

import torch

from .mobilenetv2_exporter import MobileNetV2Exporter
from .pytorch_models.mobilenet import mobilenet_v1


class MobileNetV1Exporter(MobileNetV2Exporter):
    """ONNX exporter for MobileNetV1 (MLPerf Tiny VWW benchmark).

    Reuses all training/test-data infrastructure from MobileNetV2Exporter;
    overrides only model factory, default config, and config-string tag.
    """

    def load_config(self) -> Dict[str, Any]:
        config = {
            "batch_size": 1,
            "img_size": 96,
            "input_channels": 3,
            "num_classes": 2,
            "width_mult": 0.25,
            "opset_version": 17,
            # Training
            "training_strategy": "full",
            "custom_trainable_params": [],
            "learning_rate": 0.001,
            "n_batches": 4,
            "n_accum": 1,
            "data_size": None,
        }

        if hasattr(self, "_config_overrides") and self._config_overrides:
            config.update(self._config_overrides)

        self.model_config = config
        return config

    def create_model(self) -> torch.nn.Module:
        return mobilenet_v1(
            num_classes=self.model_config["num_classes"],
            width_mult=self.model_config["width_mult"],
            input_channels=self.model_config["input_channels"],
        )

    def get_input_shape(self) -> Tuple[int, ...]:
        return (
            self.config["batch_size"],
            self.config["input_channels"],
            self.config["img_size"],
            self.config["img_size"],
        )

    def _get_config_string(self) -> str:
        width = self.config["width_mult"]
        return f"_mobilenetv1_{width}_{self.config['img_size']}_{self.config['num_classes']}"
