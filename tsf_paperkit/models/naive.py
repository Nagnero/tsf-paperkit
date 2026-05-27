from __future__ import annotations

from pathlib import Path

import numpy as np

from tsf_paperkit.models.base import BaseForecastModel


class NaiveLastValue(BaseForecastModel):
    name = "naive"

    def fit(self, train_loader, val_loader=None, config=None):
        return self

    def predict(self, test_loader):
        preds, trues = [], []
        for x, y in test_loader:
            last = x[:, -1:, :].repeat(1, y.shape[1], 1)
            preds.append(last.numpy())
            trues.append(y.numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text("naive-last-value\n")

    def load(self, path: str | Path):
        Path(path).read_text()
        return self
