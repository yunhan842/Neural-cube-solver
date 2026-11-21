import numpy as np
import pytest
import torch
import json
import subprocess
import sys

from pathlib import Path
from cube import Kociemba2x2, MOVE_NAMES
from dataset import table_to_arrays, save_dataset
from evaluate_search import dist_table, sample_by_distance, compare_heuristics, summarize
from model import DistanceNet, network_heuristic
from search import bfs, astar, zero_heuristic, table_heuristic

MAX_DEPTH = 4

@pytest.fixture(scope="module")
def table():
    dist, _, _ = bfs(MAX_DEPTH)
    cp, co, d = table_to_arrays(dist)
    return dist, cp, co, d

def test_dist_table_matches_bfs(table):
    dist, cp, co, d = table
    assert dist_table(cp, co, d) == dist

def test_sample_by_distance(table):
    _, _, _, d = table
    idx = np.arange(len(d))
    picked = sample_by_distance(idx, d, per_distance=3)
    counts = np.bincount(d[picked])
    assert counts.tolist() == [1,3,3,3,3]
    assert len(set(picked)) == len(picked)

def test_network_heuristic_returns_non_negative_float():
    torch.manual_seed(0)
    model = DistanceNet(88, (16,))
    with torch.no_grad():
        model.layers[-1].bias.fill_(-110.0) # Forcing negative raw outputs
    h = network_heuristic(model)
    value = h(Kociemba2x2().apply_sequence("R U F"))
    assert isinstance(value, float) and value == 0.0

def test_network_heuristic_plugs_into_astar():
    # Any heuristic gives a valid solution; may not be optimal
    model = DistanceNet(88, (16,))
    cube = Kociemba2x2().apply_sequence("R U F' U2")
    path, _ = astar(cube, network_heuristic(model), max_expansions=10_000)
    assert path is not None
    assert cube.copy().apply_sequence([MOVE_NAMES[m] for m in path]).is_solved()

def test_compare_and_summarize(table):
    dist, cp, co, d = table
    states = [Kociemba2x2(cp[i], co[i]) for i in sample_by_distance(np.arange(len(d)), d, 2)]
    heuristics = {"zero": zero_heuristic, "table": table_heuristic(dist, MAX_DEPTH)}
    results = compare_heuristics(states, heuristics, dist)
    assert len(results) == 2 * len(states)
    rows = {(r["heuristic"], r["dist"]): r for r in summarize(results)}
    for k in range(1, MAX_DEPTH + 1):
        assert rows[("zero", k)]["optimal"] == 1.0 and rows[("table", k)]["optimal"] == 1.0
        assert rows[("table", k)]["expanded"] == k
        assert rows[("zero", k)]["extra_moves"] == 0 and rows[("table", k)]["extra_moves"] == 0

def test_max_expansion_marks_unresolved(table):
    dist, cp, co, d = table
    i = int(np.flatnonzero(d == MAX_DEPTH)[0])
    rows = summarize(compare_heuristics([Kociemba2x2(cp[i], co[i])], {"zero": zero_heuristic}, dist, max_expansions=2))
    assert rows[0]["solved"] == 0.0 and rows[0]["extra_moves"] is None
    assert rows[0]["optimal"] == 0.0

def test_network_heuristic_is_zero_at_solved():
    model = DistanceNet(88, (16,))
    with torch.no_grad():
        model.layers[-1].bias.fill_(100.0) # raw outputs above 0
    h = network_heuristic(model)
    assert h(Kociemba2x2()) == 0.0
    assert h(Kociemba2x2().apply_move("R")) > 0

def test_measure_memory_records_peak(table):
    dist, cp, co, d = table
    i = int(np.flatnonzero(d == 3)[0])
    states, heuristics = [Kociemba2x2(cp[i], co[i])], {"zero": zero_heuristic}
    traced = compare_heuristics(states, heuristics, dist, measure_memory=True)
    untraced = compare_heuristics(states, heuristics, dist)
    assert traced[0]["peak_mb"] > 0 and untraced[0]["peak_mb"] is None
    assert summarize(untraced)[0]["peak_mb"] is None
    assert traced[0]["stored"] == untraced[0]["stored"] > traced[0]["expanded"]

def test_evaluate_script_uses_the_requested_split(tmp_path):
    # End to end: train small run and then evaluate it on the test split
    cp, co, d = table_to_arrays(bfs(3)[0])
    save_dataset(tmp_path / "tiny.npz", cp, co, d, 3)
    run_dir = tmp_path / "run"
    root = Path(__file__).parent.parent
    subprocess.run([sys.executable, "train.py", str(tmp_path / "tiny.npz"), "--epochs", "1",
                    "--hidden", "16", "--run-dir", str(run_dir)], cwd=root, check=True, capture_output=True)
    subprocess.run([sys.executable, "evaluate_search.py", str(run_dir), "--split", "test",
                    "--per-distance", "1"], cwd=root, check=True, capture_output=True)
    out = json.loads((run_dir / "eval_test.json").read_text())
    test_size = json.loads((run_dir / "run.json").read_text())["split_sizes"][2]
    assert out["settings"]["split"] == "test"
    assert sum(r["count"] for r in out["regression"]["per_distance"]) == test_size
    assert {r["heuristic"] for r in out["search"]} == {"zero", "network", "table"}
    assert all(r["solved"] == 1.0 for r in out["search"])

def test_network_heuristic_uses_the_models_encoding():
    model = DistanceNet(144, (16,))
    h = network_heuristic(model, "stickers_144")
    assert h(Kociemba2x2().apply_sequence("R U F")) >= 0.0