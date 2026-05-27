from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseForecastModel(ABC):
    name = "base"

    @abstractmethod
    def fit(self, train_loader, val_loader=None, config: dict[str, Any] | None = None): ...

    @abstractmethod
    def predict(self, test_loader): ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path): ...

    def num_params(self) -> int:
        return 0
