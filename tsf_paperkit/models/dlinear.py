from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tsf_paperkit.models.base import BaseForecastModel


class DLinearNet(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, moving_avg_kernel: int = 25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.kernel = max(1, int(moving_avg_kernel))
        self.trend_proj = nn.Linear(seq_len, pred_len)
        self.seasonal_proj = nn.Linear(seq_len, pred_len)

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, C. AvgPool over time per channel.
        if self.kernel <= 1:
            return x
        pad_left = (self.kernel - 1) // 2
        pad_right = self.kernel - 1 - pad_left
        z = x.permute(0, 2, 1)
        z = F.pad(z, (pad_left, pad_right), mode="replicate")
        z = F.avg_pool1d(z, kernel_size=self.kernel, stride=1)
        return z.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        seasonal = x - trend
        trend_out = self.trend_proj(trend.permute(0, 2, 1))
        seasonal_out = self.seasonal_proj(seasonal.permute(0, 2, 1))
        return (trend_out + seasonal_out).permute(0, 2, 1)


class DLinearForecastModel(BaseForecastModel):
    name = "dlinear"

    def __init__(self, seq_len: int, pred_len: int, channels: int, device: str = "cpu", moving_avg_kernel: int = 25):
        self.net = DLinearNet(seq_len, pred_len, channels, moving_avg_kernel).to(device)
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
                preds.append(self.net(x.to(self.device)).cpu().numpy())
                trues.append(y.numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def save(self, path: str | Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
