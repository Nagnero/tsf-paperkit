from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def mse(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred), "smape": smape(y_true, y_pred)}
