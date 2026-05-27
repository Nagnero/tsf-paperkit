import numpy as np

from tsf_paperkit.data.csv_loader import load_csv_dataset
from tsf_paperkit.runner.experiment import load_config
from tsf_paperkit.data.windowing import make_windows


def test_windowing_shape():
    x = np.arange(20, dtype="float32")[:, None]
    xs, ys = make_windows(x, seq_len=5, pred_len=2, stride=3)
    assert xs.shape == (5, 5, 1)
    assert ys.shape == (5, 2, 1)


def test_zero_validation_split_keeps_scaler_safe():
    cfg = load_config("configs/example.yaml")["data"]
    cfg = {**cfg, "split": {"train": 0.8, "val": 0.0, "test": 0.2}}
    prepared = load_csv_dataset(cfg)
    assert prepared.val.values.shape[0] == 0
    assert prepared.scaler.fit_split == "train"
