import sys
import time
import tracemalloc
import heapq
import itertools

from cube import Kociemba2x2, MOVE_NAMES

# Searching over 2x2 state space. Each move costs 1, so BFS depth is equivalent to the exact distance
def bfs(max_depth):
    """
    BFS from solved state up to max_depth moves. Returns: dist: {key: # moves from solved}, parent_move:
    {key: move id that first reached this state}, layer_counts: # new states found at each depth
    """
    if max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    start = Kociemba2x2()
    dist = {start.key(): 0}
    parent_move = {start.key(): None}
    layer_counts = [1]
    frontier = [start]

    for depth in range(1, max_depth + 1):
        next_frontier = []
        for cube in frontier:
            for m in range(len(MOVE_NAMES)):
                child = cube.copy().apply_move(m)
                k = child.key()
                # First discovery is the shortest path, so never overwrite
                if k not in dist:
                    dist[k] = depth
                    parent_move[k] = m
                    next_frontier.append(child)
        if not next_frontier:
            break
        layer_counts.append(len(next_frontier))
        frontier = next_frontier

    return dist, parent_move, layer_counts

# Inverse mapping for each move id: U <-> U', U2 <-> U2, R' <-> R, ...
INVERSE_MOVE = [3 * (m // 3) + (2 - m % 3) for m in range(len(MOVE_NAMES))]

def bfs_solution(cube, parent_move):
    # Returns optimal solution (list of move ids) for a state found by BFS or none if the table is not available
    if cube.key() not in parent_move:
        return None
    cube = cube.copy()
    path = []
    # Find inverse move that discovered each state until reaching solved state
    while (m := parent_move[cube.key()]) is not None:
        undo = INVERSE_MOVE[m]
        cube.apply_move(undo)
        path.append(undo)
    return path

def zero_heuristic(cube):
    # h = 0 turns A* into Dijkstra (same as BFS for our case), which is slow but always optimal
    return 0

def table_heuristic(dist, max_depth):
    """
    Heuristic from bfs() table: exact distance if know, otherwise max_depth + 1
    Missing state from table are > max_depth moves away, so max_depth + 1 never overestimates and the
    heuristic stays admissble
    """
    def h(cube):
        return dist.get(cube.key(), max_depth + 1)
    return h

def astar(start, heuristic, max_expansions=None):
    """
    A* search from start to solved. h(cube) estimates # moves remaining
    returns (path, stats): path is list of move ids or None if search gives up
    after max_expansions. Stats counts "expanded" and "generated" states.
    """
    start_key = start.key()
    best_g = {start_key: 0}
    parent = {start_key: None} # key -> (parent key, move id)
    stats = {"expanded": 0, "generated": 0}
    tiebreak = itertools.count() # Entries are (f, -g, tiebreak, g, key, cube). When equal f, prefer 
    # deeper state. On equal f and g, prefer earlier pushed. Counter also stops comparisons before the cube
    frontier = [(heuristic(start), 0, next(tiebreak), 0, start_key, start)]

    while frontier:
        _, _, _, g, k, cube = heapq.heappop(frontier)
        # Stale entry: shorter path was pushed after this one
        if g > best_g[k]:
            continue
        # Goal test on pop instead of push, so that the first found solution is indeed optimal
        # We also require a admissible heuristic (zero and table heuristics are admissible, but a NN one might not be) 
        if cube.is_solved():
            return reconstruct_path(parent, k), stats
        if max_expansions is not None and stats["expanded"] >= max_expansions:
            return None, stats
        stats["expanded"] += 1

        for m in range(len(MOVE_NAMES)):
            child = cube.copy().apply_move(m)
            child_key = child.key()
            child_g = g + 1
            if child_g < best_g.get(child_key, float("inf")):
                best_g[child_key] = child_g
                parent[child_key] = (k, m)
                f = child_g + heuristic(child)
                heapq.heappush(frontier, (f, -child_g, next(tiebreak), child_g, child_key, child))
                stats["generated"] += 1
    return None, stats

def reconstruct_path(parent, key):
    # Walk backwards from solved state to start then reverse
    path = []
    while parent[key] is not None:
        key, m = parent[key]
        path.append(m)
    return path[::-1]

if __name__ == "__main__":
    # Use: python search.py 6 -> run BFS to depth 6 and report time/memory
    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tracemalloc.start()
    t0 = time.perf_counter()
    dist, parent_move, layer_counts = bfs(max_depth)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"layer counts: {layer_counts}")
    print(f"states: {len(dist):,}   time: {elapsed:.1f}s   peak memory: {peak / 1e6:.1f} MB")