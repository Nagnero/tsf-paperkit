from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from tsf_paperkit.models.base import BaseForecastModel


class LinearNet(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, seq, channels -> batch, pred, channels
        return self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class LinearForecastModel(BaseForecastModel):
    name = "linear"

    def __init__(self, seq_len: int, pred_len: int, channels: int, device: str = "cpu"):
        self.net = LinearNet(seq_len, pred_len, channels).to(device)
        self.device = torch.device(device)

    def fit(self, train_loader, val_loader=None, config=None):
        config = config or {}
        epochs = int(config.get("epochs", 1))
        lr = float(config.get("learning_rate", 1e-3))
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optim.zero_grad()
                loss = loss_fn(self.net(x), y)
                loss.backward()
                optim.step()
        return self

    def predict(self, test_loader):
        self.net.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                out = self.net(x.to(self.device)).cpu().numpy()
                preds.append(out)
                trues.append(y.numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def save(self, path: str | Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
