"""İnternetsiz çalışan sentetik veri testleri (model eğitimi gerektirmez)."""
import numpy as np
import pandas as pd
from src.preprocessing import split_and_scale, make_windows
from src.evaluate import metrics


def test_split_scaler_does_not_see_future():
    df = pd.DataFrame({'x': np.r_[np.zeros(70), np.full(30, 100.0)],
                       'target': np.r_[np.zeros(70), np.full(30, 100.0)]})
    x, y, te, ve, xs, ys = split_and_scale(df, ['x'], 'target', 0.7, 0.2)
    assert te == 70 and ve == 90
    assert xs.mean_[0] == 0  # Testteki 100'ler scaler'ı etkilememeli.
    assert y[80] == 100.0


def test_windows_shape_and_no_future_input():
    x = np.arange(40, dtype='float32').reshape(-1, 1)
    y = np.arange(40, dtype='float32')
    X, Y, indices = make_windows(x, y, 10, 14, lookback=4, horizon=1)
    assert X.shape == (4, 4, 1)
    assert X[0, :, 0].tolist() == [6, 7, 8, 9]
    assert Y.tolist() == [10, 11, 12, 13]
    assert all(X[i, -1, 0] < Y[i] for i in range(len(Y)))


def test_metrics_perfect_prediction():
    m = metrics([0, 1, 2], [0, 1, 2])
    assert m['mae_c'] == 0 and m['rmse_c'] == 0 and m['r2'] == 1
