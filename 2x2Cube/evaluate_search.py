import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from cube import Kociemba2x2, ENCODINGS
from dataset import load_dataset
from model import load_model, network_heuristic
from search import astar, zero_heuristic, table_heuristic
from train import (split_indices, mean_baseline, regression_metrics, balanced_mae, 
                   per_distance_report, print_report, predict)

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

def compare_heuristics(states, heuristics, dist, max_expansions=None, measure_memory=False):
    # A* with each heuristic on each state with one result dict per (heuristic, state)
    # measure_memory records each search's peak Python memory, but tracing slows everything down,
    # so don't compare times from a run that measured memory
    results = []
    for name, h in heuristics.items():
        for cube in states:
            if measure_memory:
                tracemalloc.start()
            start = time.perf_counter()
            path, stats = astar(cube, h, max_expansions)
            elapsed = time.perf_counter() - start
            peak_mb = None
            if measure_memory:
                peak_mb = tracemalloc.get_traced_memory()[1] / 1e6 # Retrieve peak memory
                tracemalloc.stop()
            if path is not None:
                assert cube.copy().apply_sequence(path).is_solved(), \
                    "Search returned an invalid solution"
            results.append({
                "heuristic": name,
                "dist": dist[cube.key()],
                "solved": path is not None,
                "length": None if path is None else len(path),
                "expanded": stats["expanded"],
                "stored": stats["stored"],
                "seconds": elapsed,
                "peak_mb": peak_mb,
            })
    return results

def summarize(results):
    # One row per (heuristic, distance): averages over the states that were tried
    rows = []
    for name in dict.fromkeys(r["heuristic"] for r in results):
        for dist in sorted({r["dist"] for r in results if r["heuristic"] == name}):
            group = [r for r in results if r["heuristic"] == name and r["dist"] == dist]
            solved = [r for r in group if r["solved"]]
            peaks = [r["peak_mb"] for r in group if r["peak_mb"] is not None]
            rows.append({
                "heuristic": name,
                "dist": dist,
                "count": len(group),
                "solved": len(solved) / len(group),
                "optimal": sum(r["length"] == dist for r in solved) / len(group),
                "extra_moves": float(np.mean([r["length"] - dist for r in solved])) if solved else None,
                "expanded": float(np.mean([r["expanded"] for r in group])),
                "stored": float(np.mean([r["stored"] for r in group])),
                "seconds": float(np.mean([r["seconds"] for r in group])),
                "peak_mb": float(np.mean(peaks)) if peaks else None,
            })
    return rows

def print_summary(rows):
    print(f"{'heuristic':<9} {'dist':>4} {'n':>3} {'solved':>7} {'optimal':>8} {'extra':>6} "
          f"{'expanded':>9} {'stored':>9} {'time':>7} {'memory':>9}")
    for r in rows:
        extra = "-" if r["extra_moves"] is None else f"{r['extra_moves']:.2f}"
        memory = "-" if r["peak_mb"] is None else f"{r['peak_mb']:.1f} MB"
        print(f"{r['heuristic']:<9} {r['dist']:>4} {r['count']:>3} {r['solved']:>7.0%} {r['optimal']:>8.0%} "
              f"{extra:>6} {r['expanded']:>9,.0f} {r['stored']:>9,.0f} {r['seconds']:>6.2f}s {memory:>9}")

if __name__ == "__main__":
    # Use: python evaluate_search.py runs/<run> [--split val --per-distance 5 --max-expansions 100000]
    # Keep -split val while choosing settings; run --split test once, on the final frozen model
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--per-distance", type=int, default=5)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory", action="store_true", help="record peak memory (slows down timing)")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    record = json.loads((run_dir / "run.json").read_text())
    cp, co, d, max_depth = load_dataset(record["dataset"])
    train_idx, val_idx, test_idx = split_indices(d, seed=record["settings"]["split_seed"])
    idx = val_idx if args.split == "val" else test_idx
    model, info = load_model(run_dir / "model.pt")
    dist = dist_table(cp, co, d)

    # Regression on every state of the split, against the training mean baseline
    pred = predict(model, ENCODINGS[info["encoding"]](cp[idx], co[idx]))
    baseline = mean_baseline(d[train_idx], len(idx))
    regression = {
        "network": {**regression_metrics(pred, d[idx]), "balanced_mae": balanced_mae(pred, d[idx])},
        "mean_baseline": {**regression_metrics(baseline, d[idx]), "balanced_mae": balanced_mae(baseline, d[idx])},
        "per_distance": per_distance_report(pred, d[idx]),
    }
    print(f"Regression on {args.split} ({len(idx):,} states)")
    for name, m in [("network", regression["network"]), ("mean", regression["mean_baseline"])]:
        print(f"  {name:<8} MSE {m['mse']:.4f}   MAE {m['mae']:.4f}   balanced MAE {m['balanced_mae']:.4f}")
    print_report(regression["per_distance"])   

    # Search on a per-distance sample of the split
    states = [Kociemba2x2(cp[i], co[i]) for i in sample_by_distance(val_idx, d, args.per_distance, args.seed)]
    heuristics = {
        "zero": zero_heuristic,
        "network": network_heuristic(model, info["encoding"]),
        "table": table_heuristic(dist, max_depth),
    }
    rows = summarize(compare_heuristics(states, heuristics, dist, args.max_expansions, args.memory))
    print(f"\nSearch on {args.split} ({len(states)} states)")
    print_summary(rows)
    out = run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps({"settings": vars(args), "regression": regression, "search": rows}, indent=2))
    print(f"\nSaved to {out}")