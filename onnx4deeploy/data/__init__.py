# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional

from .base_datasource import DataSource
from .file_datasource import FileDataSource
from .mnist_datasource import MNISTDataSource
from .random_datasource import RandomDataSource

__all__ = [
    "DataSource",
    "RandomDataSource",
    "MNISTDataSource",
    "FileDataSource",
    "resolve_data_source",
]


def resolve_data_source(config: Optional[Dict[str, Any]]) -> DataSource:
    """按 ``config['dataset']`` 选数据源,让 ``TrainCfg.dataset/data_path`` 对所有
    **没 override** ``get_data_source()`` 的模型也生效(默认仍是 random,行为不变)。

    - ``"random"``(默认) → :class:`RandomDataSource`
    - ``"mnist"``          → :class:`MNISTDataSource`(``data_path``/``data_split``/``data_size``)
    - ``"file"``/``"npz"``/``"custom"``,或给了非 random 的 ``data_path`` → :class:`FileDataSource`
      (从用户 ``.npz`` 加载真实 inputs/labels,匹配模型输入形状)
    """
    cfg = config or {}
    ds = cfg.get("dataset", "random")
    if ds == "mnist":
        return MNISTDataSource(
            data_path=cfg.get("data_path"),
            split=cfg.get("data_split", "train"),
            data_size=cfg.get("data_size"),
        )
    if ds in ("file", "npz", "custom") or (ds != "random" and cfg.get("data_path")):
        return FileDataSource(cfg.get("data_path"), cfg.get("data_size"))
    return RandomDataSource()
