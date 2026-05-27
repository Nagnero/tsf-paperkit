import numpy as np

from tsf_paperkit.metrics.forecasting import mae, mape, mse, rmse, smape


def test_metric_correctness():
    y = np.array([1.0, 2.0, 4.0])
    p = np.array([1.0, 1.0, 2.0])
    assert mse(y, p) == (0 + 1 + 4) / 3
    assert mae(y, p) == 1.0
    assert round(rmse(y, p), 6) == round(((5 / 3) ** 0.5), 6)
    assert mape([0.0], [1.0]) > 0
    assert smape(y, p) >= 0
