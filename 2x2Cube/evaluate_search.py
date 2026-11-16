import argparse
import json
import time
from pathlib import Path

import numpy as np

from cube import Kociemba2x2
from dataset import load_dataset
from model import load_model, network_heuristic
from search import astar, zero_heuristic, table_heuristic
from train import split_indices

# Comparing A* heuristics (h = 0, trained network, table) on held out states

def dist_table(cp, co, d):
    # {key, dist} from dataset arrays, with same keys as bfs() and Kociemba2x2.key()
    return {cp[i].tobytes() + co[i].tobytes(): int(d[i]) for i in range(len(d))}

def sample_by_distance(idx, d, per_distance, seed=0):
    # Up to per_distance rows from idx at each distance, so that short distances aren't drowned out
    rng = np.random.default_rng(seed)
    picked = []
    for dist in np.unique(d[idx]):
        rows = idx[d[idx] == dist]
        picked.extend(rng.choice(rows, min(per_distance, len(rows)), replace=False).tolist())
    return picked

def compare_heuristics(states, heuristics, dist, max_expansions=None):
    # A* with each heuristic on each state with one result dict per (heuristic, state)
    results = []
    for name, h in heuristics.items():
        for cube in states:
            start = time.perf_counter()
            path, stats = astar(cube, h, max_expansions)
            elapsed = time.perf_counter() - start
            if path is not None:
                assert cube.copy().apply_sequence(path).is_solved(), \
                    "Search returned an invalid solution"
            results.append({
                "heuristic": name,
                "dist": dist[cube.key()],
                "solved": path is not None,
                "length": None if path is None else len(path),
                "expanded": stats["expanded"],
                "seconds": elapsed,
            })
    return results

def summarize(results):
    # One row per (heuristic, distance): averages over the states that were tried
    rows = []
    for name in dict.fromkeys(r["heuristic"] for r in results):
        for dist in sorted({r["dist"] for r in results if r["heuristic"] == name}):
            group = [r for r in results if r["heuristic"] == name and r["dist"] == dist]
            solved = [r for r in group if r["solved"]]
            rows.append({
                "heuristic": name,
                "dist": dist,
                "count": len(group),
                "solved": len(solved) / len(group),
                "optimal": sum(r["length"] == dist for r in solved) / len(group),
                "extra_moves": float(np.mean([r["length"] - dist for r in solved])) if solved else None,
                "expanded": float(np.mean([r["expanded"] for r in group])),
                "seconds": float(np.mean([r["seconds"] for r in group]))
            })
    return rows

def print_summary(rows):
    print(f"{'heuristic':<9} {'dist':>4} {'n':>3} {'solved':>7} {'optimal':>8} {'extra':>6} "
          f"{'expanded':>9} {'time':>7}")
    for r in rows:
        extra = "-" if r["extra_moves"] is None else f"{r['extra_moves']:.2f}"
        print(f"{r['heuristic']:<9} {r['dist']:>4} {r['count']:>3} {r['solved']:>7.0%} {r['optimal']:>8.0%} "
              f"{extra:>6} {r['expanded']:>9,.0f} {r['seconds']:>6.2f}s")

if __name__ == "__main__":
    # Use: python evaluate_search.py runs/<run> [--per-distance 5 --max-expansions 100000]
    # Uses the run's validation states; the test set stays untouched
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--per-distance", type=int, default=5)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    record = json.loads((run_dir / "run.json").read_text())
    cp, co, d, max_depth = load_dataset(record["dataset"])
    _, val_idx, _ = split_indices(d, seed=record["settings"]["split_seed"])
    model, info = load_model(run_dir / "model.pt")
    dist = dist_table(cp, co, d)

    states = [Kociemba2x2(cp[i], co[i]) for i in sample_by_distance(val_idx, d, args.per_distance, args.seed)]
    heuristics = {
        "zero": zero_heuristic,
        "network": network_heuristic(model),
        "table": table_heuristic(dist, max_depth),
    }
    rows = summarize(compare_heuristics(states, heuristics, dist, args.max_expansions))
    print_summary(rows)
    out = run_dir / "search_val.json"
    out.write_text(json.dumps({"settings": vars(args), "summary": rows}, indent=2))
    print(f"\nSaved to {out}")