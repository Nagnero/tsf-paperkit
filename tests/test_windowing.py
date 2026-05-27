import numpy as np

from tsf_paperkit.data.windowing import make_windows


def test_windowing_shape():
    x = np.arange(20, dtype="float32")[:, None]
    xs, ys = make_windows(x, seq_len=5, pred_len=2, stride=3)
    assert xs.shape == (5, 5, 1)
    assert ys.shape == (5, 2, 1)
