from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowMetadata:
    start: int
    input_start: int
    input_end: int
    target_start: int
    target_end: int


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int, pred_len: int, stride: int = 1, offset: int = 0):
        if seq_len <= 0 or pred_len <= 0 or stride <= 0:
            raise ValueError("seq_len, pred_len, and stride must be positive")
        values = np.asarray(values, dtype="float32")
        if values.ndim == 1:
            values = values[:, None]
        self.values = values
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.offset = offset
        max_start = len(values) - seq_len - pred_len
        self.starts = list(range(0, max_start + 1, stride)) if max_start >= 0 else []

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        x = self.values[start : start + self.seq_len]
        y = self.values[start + self.seq_len : start + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def metadata(self) -> list[WindowMetadata]:
        return [
            WindowMetadata(
                start=s + self.offset,
                input_start=s + self.offset,
                input_end=s + self.offset + self.seq_len,
                target_start=s + self.offset + self.seq_len,
                target_end=s + self.offset + self.seq_len + self.pred_len,
            )
            for s in self.starts
        ]


def make_windows(values: np.ndarray, seq_len: int, pred_len: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    ds = WindowDataset(values, seq_len=seq_len, pred_len=pred_len, stride=stride)
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
    if not xs:
        features = 1 if np.asarray(values).ndim == 1 else np.asarray(values).shape[-1]
        return np.empty((0, seq_len, features), dtype="float32"), np.empty((0, pred_len, features), dtype="float32")
    return np.stack(xs), np.stack(ys)
