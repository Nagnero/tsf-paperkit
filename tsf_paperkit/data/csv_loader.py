from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from tsf_paperkit.data.scaler import TrainOnlyScaler
from tsf_paperkit.data.windowing import WindowDataset


@dataclass
class SplitData:
    values: np.ndarray
    start_index: int
    end_index: int
    dataset: WindowDataset | None = None


@dataclass
class PreparedData:
    frame: pd.DataFrame
    train: SplitData
    val: SplitData
    test: SplitData
    scaler: TrainOnlyScaler
    target_cols: list[str]
    timestamp_col: str


def _split_indices(n: int, split: dict[str, float]) -> tuple[int, int]:
    train_ratio = float(split.get("train", 0.7))
    val_ratio = float(split.get("val", 0.1))
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must satisfy train > 0, val >= 0, train + val < 1")
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return train_end, val_end


def load_csv_dataset(config: dict[str, Any]) -> PreparedData:
    path = Path(config["path"])
    timestamp_col = config.get("timestamp_col", "date")
    target_cols = list(config.get("target_cols") or [])
    if not target_cols:
        raise ValueError("data.target_cols must include at least one column")
    frame = pd.read_csv(path)
    missing = [c for c in [timestamp_col, *target_cols] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    frame = frame.sort_values(timestamp_col).reset_index(drop=True)
    values = frame[target_cols].to_numpy(dtype="float32")
    train_end, val_end = _split_indices(len(values), config.get("split", {}))
    train_raw = values[:train_end]
    val_raw = values[train_end:val_end]
    test_raw = values[val_end:]
    scaler = TrainOnlyScaler(enabled=bool(config.get("scale", True)))
    train_values = scaler.fit_transform(train_raw, split_name="train")
    val_values = scaler.transform(val_raw)
    test_values = scaler.transform(test_raw)
    return PreparedData(
        frame=frame,
        train=SplitData(train_values, 0, train_end),
        val=SplitData(val_values, train_end, val_end),
        test=SplitData(test_values, val_end, len(values)),
        scaler=scaler,
        target_cols=target_cols,
        timestamp_col=timestamp_col,
    )


def prepare_dataloaders(config: dict[str, Any], batch_size: int, drop_last: bool = False) -> tuple[PreparedData, dict[str, DataLoader]]:
    prepared = load_csv_dataset(config)
    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])
    stride = int(config.get("stride", 1))
    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        split = getattr(prepared, name)
        split.dataset = WindowDataset(split.values, seq_len, pred_len, stride, offset=split.start_index)
        loaders[name] = DataLoader(
            split.dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=drop_last if name == "test" else False,
        )
    return prepared, loaders
