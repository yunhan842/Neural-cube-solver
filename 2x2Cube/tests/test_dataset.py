import numpy as np
import pytest

from pathlib import Path
from cube import Kociemba2x2, encode_onehot
from dataset import table_to_arrays, save_dataset, load_dataset, require_npz
from search import bfs

MAX_DEPTH = 5

@pytest.fixture(scope="module")
def arrays():
    dist, _, layer_counts = bfs(MAX_DEPTH)
    cp, co, d = table_to_arrays(dist)
    return dist, layer_counts, cp, co, d

def test_shapes_and_types(arrays):
    dist, _, cp, co, d = arrays
    n = len(dist)
    assert cp.shape == co.shape == (n, 8)
    assert d.shape == (n,)
    assert cp.dtype == co.dtype == d.dtype == np.int8

def test_every_row_matches_table(arrays):
    dist, _, cp, co, d = arrays
    for i in range(len(d)):
        cube = Kociemba2x2(cp[i], co[i])
        assert cube.is_valid()
        assert dist[cube.key()] == d[i]

def test_label_counts_match_layer_counts(arrays):
    _, layer_counts, _, _, d = arrays
    assert np.bincount(d).tolist() == layer_counts

def test_save_load_round_trip(arrays, tmp_path):
    _, _, cp, co, d = arrays
    path = tmp_path / "bfs.npz"
    save_dataset(path, cp, co, d, MAX_DEPTH)
    cp2, co2, d2, max_depth = load_dataset(path)
    assert np.array_equal(cp, cp2) and np.array_equal(co, co2) and np.array_equal(d, d2)
    assert max_depth == MAX_DEPTH

def test_dataset_encodes_for_network(arrays):
    dist, _, cp, co, _ = arrays
    x = encode_onehot(cp, co)
    assert x.shape == (len(dist), 88)
    assert (x.sum(axis=1) == 16).all()

@pytest.mark.parametrize("name", ["train", "train.npy", "train.NPZ", "train.npz.bak"])
def test_save_and_load_reject_non_npz_names(arrays, tmp_path, name):
    _, _, cp, co, d = arrays
    with pytest.raises(ValueError, match=r"\.npz"):
        save_dataset(tmp_path / name, cp, co, d, MAX_DEPTH)
    with pytest.raises(ValueError, match=r"\.npz"):
        load_dataset(tmp_path / name)

    # Nothing should be written
    assert list(tmp_path.iterdir()) == []

def test_require_npz_accepts_strings_and_paths(tmp_path):
    assert require_npz("data/train.npz") == Path("data/train.npz")
    assert require_npz(tmp_path / "train.npz") == tmp_path / "train.npz"