from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from tsf_paperkit.models.base import BaseForecastModel


class PatchTSTNet(nn.Module):
    """Small PatchTST-style forecaster.

    This is an MVP implementation inspired by the PatchTST paper's two core
    ideas: segment each univariate channel into patches, then share the same
    Transformer backbone independently across channels. It intentionally omits
    paper-grade options such as RevIN, self-supervised pretraining, and benchmark
    protocol tuning.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(channels)
        self.patch_len = max(1, int(patch_len))
        self.stride = max(1, int(stride))
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.pad_len = max(0, self.patch_len - self.seq_len)
        effective_len = self.seq_len + self.pad_len
        self.n_patches = ((effective_len - self.patch_len) // self.stride) + 1
        if self.n_patches < 1:
            raise ValueError("PatchTST requires at least one patch")

        self.patch_proj = nn.Linear(self.patch_len, self.d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.n_patches * self.d_model, self.pred_len)

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, C -> B*C, N, patch_len
        if self.pad_len:
            pad = x[:, -1:, :].repeat(1, self.pad_len, 1)
            x = torch.cat([x, pad], dim=1)
        x = x.permute(0, 2, 1)  # B, C, T
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches.reshape(-1, self.n_patches, self.patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        patches = self._patch(x)
        z = self.patch_proj(patches) + self.position
        z = self.encoder(self.dropout(z))
        out = self.head(z.flatten(start_dim=1))
        return out.reshape(batch, channels, self.pred_len).permute(0, 2, 1)


class PatchTSTForecastModel(BaseForecastModel):
    name = "patchtst"

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        device: str = "cpu",
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        self.net = PatchTSTNet(
            seq_len,
            pred_len,
            channels,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)
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
