import numpy as np
import pytest
import torch

from cube import Kociemba2x2, MOVE_NAMES
from dataset import table_to_arrays
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


