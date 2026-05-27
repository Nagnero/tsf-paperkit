from __future__ import annotations

from tsf_paperkit.metrics.forecasting import all_metrics


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return all_metrics(y_true, y_pred)
