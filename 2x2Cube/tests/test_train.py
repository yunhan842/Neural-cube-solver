import numpy as np
import pytest

from dataset import table_to_arrays, save_dataset
from search import bfs
from train import make_xy, split_indices, mean_baseline, regression_metrics, per_distance_report
from model import DistanceNet

@pytest.fixture(scope="module")
def arrays():
    cp, co, d = table_to_arrays(bfs(4)[0])
    return cp, co, d

def test_make_xy_shapes_and_types(arrays):
    cp, co, d = arrays
    X, y = make_xy(cp, co, d)
    assert X.shape == (len(d), 88) and X.dtype == np.float32
    assert y.shape == (len(d), 1) and y.dtype == np.float32
    assert np.array_equal(y[:, 0], d)

def test_splits_cover_every_row_exactly_once(arrays):
    _, _, d = arrays
    train, val, test = split_indices(d)
    assert np.array_equal(np.sort(np.concatenate([train, val, test])), np.arange(len(d)))

def test_splits_are_reproducible(arrays):
    _, _, d = arrays
    for a, b in zip(split_indices(d, seed=0), split_indices(d, seed=0)):
        assert np.array_equal(a, b)
    assert not np.array_equal(split_indices(d, seed=0)[0], split_indices(d, seed=1)[0])

def test_splits_keep_distance_proportions(arrays):
    _, _, d = arrays
    train, val, test = split_indices(d)
    for dist in range(1, 5):  # distance 0 is a single state, so it can only be in one split
        total = (d == dist).sum()
        for idx, frac in [(train, 0.8), (val, 0.1), (test, 0.1)]:
            count = (d[idx] == dist).sum()
            assert count >= 1
            assert abs(count - frac * total) <= 1

def test_mean_baseline():
    pred = mean_baseline(np.array([1, 2, 3, 6], dtype=np.int8), 5)
    assert pred.shape == (5,) and np.allclose(pred, 3.0)

def test_regression_metrics():
    m = regression_metrics(np.array([[1.0], [2.0], [5.0]]), np.array([1, 3, 3], dtype=np.int8))
    # errors 0, -1, 2
    assert m["mse"] == pytest.approx(5 / 3)
    assert m["mae"] == pytest.approx(1.0)

def test_per_distance_report():
    pred = np.array([0.2, 1.4, 0.6, 2.0, 3.0])
    dist = np.array([0, 1, 1, 2, 2], dtype=np.int8)
    rows = {r["dist"]: r for r in per_distance_report(pred, dist)}
    assert rows[1]["count"] == 2
    assert rows[1]["mean_pred"] == pytest.approx(1.0)
    assert rows[1]["mae"] == pytest.approx(0.4)
    assert rows[1]["rounded_acc"] == pytest.approx(1.0)   # 1.4 and 0.6 both round to 1
    assert rows[1]["overestimate"] == pytest.approx(0.5)  # only 1.4 > 1
    assert rows[2]["overestimate"] == pytest.approx(0.5)  # 2.0 isn't > 2, 3.0 is

def test_make_xy_uses_requested_encoding(arrays):
    cp, co, d = arrays
    X, y = make_xy(cp, co, d, "stickers_144")
    assert X.shape == (len(d), 144) and y.shape == (len(d), 1)