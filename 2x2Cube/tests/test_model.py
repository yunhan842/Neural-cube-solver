import numpy as np
import pytest
import torch
import json
import subprocess
import sys

from pathlib import Path
from dataset import table_to_arrays, save_dataset
from model import DistanceNet, save_model, load_model
from search import bfs
from train import make_xy, split_indices, regression_metrics, predict, train_model

@pytest.fixture(scope="module")
def data():
    cp, co, d = table_to_arrays(bfs(4)[0])
    X, y = make_xy(cp, co, d)
    train_idx, val_idx, _ = split_indices(d)
    return X, y, d, train_idx, val_idx

def quiet(*args, **kwargs):
    pass

def test_output_shape():
    model = DistanceNet(88, (32,16))
    assert model(torch.zeros(5,88)).shape == (5,1)

def test_hidden_layers_match_sizes():
    model = DistanceNet(88, (64,32,16))
    linear = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(88,64), (64,32), (32,16), (16,1)]

def test_predict_is_flat_and_builds_no_gradients(data):
    X, _, _, _, _ = data
    model = DistanceNet(88, (32,))
    model.train()
    pred = predict(model, X[:10], batch_size=3)
    assert pred.shape == (10,) and pred.dtype == np.float32
    assert not model.training
    with torch.no_grad():
        expected = model(torch.from_numpy(X[:10])).numpy().ravel()
    assert np.allclose(pred, expected)

def test_training_can_fit_a_small_set(data):
    # Checks the training loop works: a network should be able to memorize 200 states.
    # (Beating the mean on unseen states is the real experiment, not a unit test.)
    X, y, d, train_idx, _ = data
    idx = train_idx[:200]
    model, history = train_model(X[idx], y[idx], X[idx], d[idx],
                                 hidden_sizes=(128,), epochs=150, batch_size=32, lr=3e-3, log=quiet)
    assert history[-1][1] < history[0][1]
    assert (np.rint(predict(model, X[idx])) == d[idx]).mean() > 0.95

def test_returned_model_has_best_validation_weights(data):
    X, y, d, train_idx, val_idx = data
    model, history = train_model(X[train_idx], y[train_idx], X[val_idx], d[val_idx],
                                 hidden_sizes=(32,), epochs=5, log=quiet)
    best_val = min(h[2] for h in history)
    assert regression_metrics(predict(model, X[val_idx]), d[val_idx])["mse"] == pytest.approx(best_val)


def test_training_is_reproducible(data):
    X, y, d, train_idx, val_idx = data
    runs = [train_model(X[train_idx], y[train_idx], X[val_idx], d[val_idx],
                        hidden_sizes=(32,), epochs=2, seed=3, log=quiet)[0] for _ in range(2)]
    assert np.array_equal(predict(runs[0], X), predict(runs[1], X))

def test_save_and_load_give_identical_predictions(data, tmp_path):
    X, _, _, _, _ = data
    model = DistanceNet(88, (32, 16))
    save_model(tmp_path / "sub" / "model.pt", model, encoding="cubie_onehot_88", max_depth=4)
    loaded, info = load_model(tmp_path / "sub" / "model.pt")
    assert loaded.hidden_sizes == (32, 16)
    assert not loaded.training
    assert info == {"encoding": "cubie_onehot_88", "max_depth": 4}
    assert np.array_equal(predict(model, X), predict(loaded, X))


def test_train_script_saves_a_reloadable_run(tmp_path):
    # End-to-end: run train.py on a tiny dataset and check what it writes
    cp, co, d = table_to_arrays(bfs(3)[0])
    save_dataset(tmp_path / "tiny.npz", cp, co, d, 3)
    run_dir = tmp_path / "run"
    subprocess.run([sys.executable, "train.py", str(tmp_path / "tiny.npz"), "--epochs", "2",
                    "--hidden", "16", "--split-seed", "1", "--seed", "2", "--run-dir", str(run_dir)],
                   cwd=Path(__file__).parent.parent, check=True, capture_output=True)

    record = json.loads((run_dir / "run.json").read_text())
    assert record["settings"]["split_seed"] == 1 and record["settings"]["seed"] == 2
    assert [h["epoch"] for h in record["history"]] == [1, 2]
    assert sum(record["split_sizes"]) == len(d)

    # The saved model reproduces the recorded validation numbers
    model, info = load_model(run_dir / "model.pt")
    assert info["max_depth"] == 3
    X, _ = make_xy(cp, co, d)
    _, val_idx, _ = split_indices(d, seed=1)
    mse = regression_metrics(predict(model, X[val_idx]), d[val_idx])["mse"]
    assert mse == pytest.approx(record["val"]["network"]["mse"])