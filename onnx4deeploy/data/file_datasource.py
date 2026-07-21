# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""FileDataSource — bring-your-own dataset loaded from an ``.npz`` at ``data_path``.

Lets ``TrainCfg.dataset="file"`` + ``TrainCfg.data_path=<x.npz>`` propagate a real
(train or eval) dataset to ANY model, matching the model's input shape. The npz
must hold inputs + integer labels under one of these key pairs:
``inputs``/``labels`` · ``x``/``y`` · ``X``/``Y`` · ``arr_0``/``arr_1``.
"""

from typing import List, Optional, Tuple

import numpy as np

from .base_datasource import DataSource

_INPUT_KEYS = ("inputs", "x", "X", "arr_0")
_LABEL_KEYS = ("labels", "y", "Y", "arr_1")


class FileDataSource(DataSource):
    """Load (inputs, labels) from a user-provided ``.npz`` and slice into batches."""

    def __init__(self, data_path: str, data_size: Optional[int] = None):
        if not data_path:
            raise ValueError("FileDataSource 需要 data_path(指向 .npz)")
        self.data_path = data_path
        self.data_size = data_size
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def _ensure_loaded(self) -> None:
        if self._x is not None:
            return
        d = np.load(self.data_path)

        def _pick(keys):
            for k in keys:
                if k in d.files:
                    return d[k]
            return None

        x, y = _pick(_INPUT_KEYS), _pick(_LABEL_KEYS)
        if x is None or y is None:
            raise ValueError(
                f"{self.data_path} 需含 inputs/labels(或 x/y、X/Y、arr_0/arr_1);"
                f"实际 keys={list(d.files)}")
        self._x = np.asarray(x, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.int64).reshape(-1)
        if len(self._x) != len(self._y):
            raise ValueError(f"inputs({len(self._x)}) 与 labels({len(self._y)}) 数量不一致")

    def load_batches(
        self,
        n_batches: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        self._ensure_loaded()
        rng = np.random.RandomState(seed)
        batch_size = input_shape[0]
        n = len(self._x)
        # 固定一个 pool(可复现),不足 n_batches*batch_size 时按 epoch 洗牌循环。
        pool = n if self.data_size is None else min(self.data_size, n)
        pool_idx = rng.choice(n, pool, replace=False)
        need = n_batches * batch_size
        order: List[int] = []
        while len(order) < need:
            order.extend(pool_idx[rng.permutation(pool)].tolist())
        order = np.array(order[:need])

        inputs_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        for b in range(n_batches):
            sel = order[b * batch_size:(b + 1) * batch_size]
            xb = self._x[sel]
            # reshape 到模型的 input_shape(逐样本 feature 尺寸需匹配)
            if xb.shape != tuple(input_shape):
                xb = xb.reshape(input_shape)
            inputs_list.append(xb.astype(np.float32))
            labels_list.append(self._y[sel].astype(np.int64))
        return inputs_list, labels_list
