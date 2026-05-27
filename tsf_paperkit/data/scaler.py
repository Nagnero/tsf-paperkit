from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainOnlyScaler:
    enabled: bool = True
    scaler: StandardScaler | None = None
    fit_split: str | None = None

    def fit(self, values: np.ndarray, split_name: str = "train") -> "TrainOnlyScaler":
        if not self.enabled:
            self.fit_split = split_name
            return self
        if split_name != "train":
            raise ValueError("Scaler must be fit on the train split only.")
        self.scaler = StandardScaler()
        self.scaler.fit(values.reshape(-1, values.shape[-1]))
        self.fit_split = split_name
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return values.astype("float32")
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fit.")
        shape = values.shape
        scaled = self.scaler.transform(values.reshape(-1, shape[-1])).reshape(shape)
        return scaled.astype("float32")

    def fit_transform(self, values: np.ndarray, split_name: str = "train") -> np.ndarray:
        self.fit(values, split_name=split_name)
        return self.transform(values)
