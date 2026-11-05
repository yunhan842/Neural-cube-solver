import random
import numpy as np
import pytest

from cube import Kociemba2x2, MOVE_NAMES, random_scrambles
from search import bfs, bfs_solution, INVERSE_MOVE, astar, zero_heuristic, table_heuristic

MAX_DEPTH = 5

@pytest.fixture(scope="module")
def table():
    # Run BFS exactly once and share amongst test cases
    return bfs(MAX_DEPTH)

def test_from_key_round_trip():
    cube = Kociemba2x2().apply_sequence("R U F' U2 R")
    assert Kociemba2x2.from_key(cube.key()) == cube

def test_bfs_layer_counts():
    dist, parent_move, layer_counts = bfs(5)
    assert layer_counts == [1, 9, 54, 321, 1847, 9992]
    assert len(dist) == len(parent_move) == sum(layer_counts)

def test_bfs_distances_are_exact():
    # A state at distance d must have a neighbor at d-1 and none closer than that
    max_depth = 5
    dist, _, _ = bfs(max_depth)
    keys = random.Random(0).sample(sorted(dist), 500)
    for k in keys:
        d = dist[k]
        cube = Kociemba2x2.from_key(k)
        neighbor_dists = [dist.get(cube.copy().apply_move(m).key()) for m in range(len(MOVE_NAMES))]
        known = [nd for nd in neighbor_dists if nd is not None]
        assert min(known) >= d-1
        if d > 0:
            assert d-1 in known

def test_inverse_moves_undo_each_move():
    for m in range(len(MOVE_NAMES)):
        assert Kociemba2x2().apply_move(m).apply_move(INVERSE_MOVE[m]).is_solved()

def test_bfs_solutions_are_valid_and_optimal(table):
    dist, parent_move, _ = table
    for k, d in dist.items():
        cube = Kociemba2x2.from_key(k)
        path = bfs_solution(cube, parent_move)
        assert len(path) == d
        for m in path:
            cube.apply_move(m)
        assert cube.is_solved()

def test_bfs_solution_solves_scrambles(table):
    _, parent_move, _ = table
    cps, cos = random_scrambles(200, MAX_DEPTH, rng=np.random.default_rng(0))
    for cp, co in zip(cps, cos):
        cube = Kociemba2x2(cp, co)
        path = bfs_solution(cube, parent_move)
        assert path is not None and len(path) <= MAX_DEPTH
        assert cube.apply_sequence([MOVE_NAMES[m] for m in path]).is_solved()

def test_bfs_solution_outside_table_is_none(table):
    _, parent_move, _ = table
    # Exactly 6 moves from solved (checked with bfs(6)) -> should NOT be in depth 5 table
    deep = Kociemba2x2().apply_sequence("R U2 F R' U F2")
    assert deep.key() not in parent_move
    assert bfs_solution(deep, parent_move) is None

def solves(cube, path):
    return cube.copy().apply_sequence([MOVE_NAMES[m] for m in path]).is_solved()

def test_astar_zero_heuristic_matches_bfs(table):
    # h = 0 -> expands everything closer than the solution
    dist, _, _ = table
    keys = [k for k in sorted(dist) if dist[k] <= 4]
    for k in random.Random(1).sample(keys, 30):
        cube = Kociemba2x2.from_key(k)
        path, _ = astar(cube, zero_heuristic)
        assert solves(cube, path)
        assert len(path) == dist[k]

def test_astar_table_heuristic_goes_straight_to_solved(table):
    # Perfect heuristic should expand exactly one state per move of the solution
    dist, _, _ = table
    h = table_heuristic(dist, MAX_DEPTH)
    for k in random.Random(2).sample(sorted(dist), 300):
        cube = Kociemba2x2.from_key(k)
        path, stats = astar(cube, h)
        assert solves(cube, path)
        assert len(path) == dist[k]
        assert stats["expanded"] == dist[k]

def test_astar_table_heuristic_is_optimal_beyond_table(table):
    dist, _, _ = table
    deep = Kociemba2x2().apply_sequence("R U2 F R' U F2")
    path, _ = astar(deep, table_heuristic(dist, MAX_DEPTH))
    assert solves(deep, path)
    assert len(path) == 6

def test_astar_gives_up_after_max_expansions():
    deep = Kociemba2x2().apply_sequence("R U2 F R' U F2")
    path, stats = astar(deep, zero_heuristic, max_expansions=10)
    assert path is None and stats["expanded"] == 10