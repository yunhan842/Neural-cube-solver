import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from cube import ENCODINGS
from dataset import load_dataset
from model import DistanceNet, save_model

# Splits, baselines, and eval for distance regressor
SPLIT_FRACTIONS = (0.8, 0.1, 0.1) # train, validation, test

def make_xy(cp, co, dist, encoding="cubie_onehot_88"):
    # Network inputs (N,88) and float targets (N,1); matching shapes will avoid accidental broadcasting
    X = ENCODINGS[encoding](cp, co)
    y = dist.astype(np.float32).reshape(-1, 1)
    return X, y

def split_indices(dist, fractions=SPLIT_FRACTIONS, seed=0):
    """
    Shuffled train/validation/test row indices
    Each distance is split separately, so every split gets each distance in about the same
    proportion (otherwise the few states at distances 0-2 could miss a split).
    """
    dist = np.asarray(dist)
    if dist.ndim != 1 or dist.size == 0:
        raise ValueError("dist must be a nonempty one-dimensional array")
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.shape != (3,) or not np.isfinite(fractions).all()
        or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-8)):
        raise ValueError("fractions must contain three positive values summing to 1.0")
    rng = np.random.default_rng(seed)
    splits = ([], [], [])
    for d in np.unique(dist):
        rows = rng.permutation(np.flatnonzero(dist == d))
        n_train = round(fractions[0] * len(rows))
        n_val = round(fractions[1] * len(rows))
        splits[0].append(rows[:n_train])
        splits[1].append(rows[n_train:n_train + n_val])
        splits[2].append(rows[n_train + n_val:])
    # Shuffle again so each split isn't ordered by distance
    result = tuple(rng.permutation(np.concatenate(s)) for s in splits)
    empty = [name for name, rows in zip(("train", "validation", "test"), result) if len(rows) == 0]
    if empty:
        raise ValueError(f"Empty overall split(s): {', '.join(empty)}. "
                         "Use more states or adjust the split fractions.")
    return result

def mean_baseline(train_dist, n):
    # Always predict training set mean distance (constant with lowest MSE)
    return np.full(n, train_dist.mean(), dtype=np.float32)

def regression_metrics(pred, dist):
    err = np.ravel(pred) - np.ravel(dist).astype(np.float32)
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err)))}

def per_distance_report(pred, dist):
    # One row per true distance: count, mean pred, MAE, rounded accuracy, overestimate rate
    pred, dist = np.ravel(pred), np.ravel(dist)
    rows = []
    for d in np.unique(dist):
        p = pred[dist == d]
        rows.append({
            "dist": int(d),
            "count": len(p),
            "mean_pred": float(p.mean()),
            "mae": float(np.abs(p - d).mean()),
            "rounded_acc": float((np.rint(p) == d).mean()),
            "overestimate": float((p > d).mean()),
        })
    return rows

def print_report(rows):
    print(f"{'dist':>4} {'count':>7} {'mean pred':>9} {'MAE':>6} {'rounded acc':>11} {'overest.':>8}")
    for r in rows:
        print(f"{r['dist']:>4} {r['count']:>7,} {r['mean_pred']:>9.2f} {r['mae']:>6.2f} "
              f"{r['rounded_acc']:>11.1%} {r['overestimate']:>8.1%}")

def balanced_mae(pred, dist):
    # Average of per-distance MAEs so every distance counts equally however many states it has
    return float(np.mean([r["mae"] for r in per_distance_report(pred, dist)]))

def predict(model, X, batch_size=4096):
    # Model predictions for X (N,88) as a NumPy (N,) array, without building gradient graphs
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    model.eval()
    if len(X) == 0:
        return np.empty(0, dtype=np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            preds.append(model(torch.from_numpy(X[start:start + batch_size])).numpy())
    return np.concatenate(preds).ravel()

def train_model(X_train, y_train, X_val, d_val, hidden_sizes=(256,256), epochs=20, batch_size=256, lr=1e-3, seed=0, log=print):
    """
    Train DistanceNet with MSE and Adam optimizer. Returns model with the weights from the epoch
    with lowest validation MSE and history of (epoch, train MSE, val MSE)
    """
    if not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must be two-dimensional arrays")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation inputs must both be nonempty")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Training and validation inputs must have the same feature count")
    if y_train.shape != (len(X_train), 1):
        raise ValueError("y_train must have shape (len(X_train), 1)")
    if np.asarray(d_val).shape != (len(X_val),):
        raise ValueError("d_val must have shape (len(X_val),)")
    torch.manual_seed(seed)
    model = DistanceNet(X_train.shape[1], hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # Seeded generator so shuffle order is the same every run
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_val, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        # Avg loss over epoch's batches (weights changed during the epoch)
        train_mse = total_loss / len(X_train)
        val_mse = regression_metrics(predict(model, X_val), d_val)["mse"]
        if not np.isfinite(train_mse) or not np.isfinite(val_mse):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: train={train_mse}, validation={val_mse}")
        history.append((epoch, train_mse, val_mse))
        # Keep a copy of the best weights; later epochs keep changing model's own weights
        if val_mse < best_val:
            best_val, best_state = val_mse, copy.deepcopy(model.state_dict())
        log(f"epoch {epoch:>3}   train MSE {train_mse:.4f}   val MSE {val_mse:.4f}"
            + ("   *" if val_mse == best_val else ""))

    model.load_state_dict(best_state)
    return model, history

if __name__ == "__main__":
    # Use: python train.py data/bfs_depth6.npz [--epochs 20 --hidden 256 256 --lr 1e-3 ...]
    # Saves model.pt and run.json to --run-dir (default: runs/<date>_<time>)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--split-seed", type=int, default=0, help="which states go to train/val/test")
    parser.add_argument("--seed", type=int, default=0, help="initial weights and shuffle order")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--encoding", choices=list(ENCODINGS), default="cubie_onehot_88")
    args = parser.parse_args()
    run_dir = Path(args.run_dir or f"runs/{time.strftime('%Y-%m-%d_%H%M%S')}")

    cp, co, d, max_depth = load_dataset(args.dataset)
    X, y = make_xy(cp, co, d, args.encoding)
    train_idx, val_idx, test_idx = split_indices(d, seed=args.split_seed)
    print(f"{len(d):,} states up to depth {max_depth}: "
          f"train {len(train_idx):,}, val {len(val_idx):,}, test {len(test_idx):,}")

    train_d, val_d = d[train_idx], d[val_idx]
    baseline = mean_baseline(train_d, len(val_idx))
    baseline_metrics = {**regression_metrics(baseline, val_d), "balanced_mae": balanced_mae(baseline, val_d)}
    print(f"\nMean baseline on val: MSE {baseline_metrics['mse']:.3f}   MAE {baseline_metrics['mae']:.3f}   "
          f"balanced MAE {baseline_metrics['balanced_mae']:.3f}\n")

    start = time.perf_counter()
    model, history = train_model(X[train_idx], y[train_idx], X[val_idx], val_d,
                                 hidden_sizes=args.hidden, epochs=args.epochs,
                                 batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    train_seconds = time.perf_counter() - start

    pred = predict(model, X[val_idx])
    val_metrics = {**regression_metrics(pred, val_d), "balanced_mae": balanced_mae(pred, val_d)}
    val_report = per_distance_report(pred, val_d)
    best_epoch = min(history, key=lambda h: h[2])[0]
    print(f"\nNetwork on val (best epoch {best_epoch}): MSE {val_metrics['mse']:.4f}   "
          f"MAE {val_metrics['mae']:.4f}   balanced MAE {val_metrics['balanced_mae']:.4f}")
    print_report(val_report)

    save_model(run_dir / "model.pt", model, encoding=args.encoding, max_depth=max_depth)
    record = {
        "dataset": args.dataset,
        "max_depth": max_depth,
        "split_sizes": [len(train_idx), len(val_idx), len(test_idx)],
        "settings": vars(args),
        "torch_version": torch.__version__,
        "train_seconds": round(train_seconds, 1),
        "history": [{"epoch": e, "train_mse": t, "val_mse": v} for e, t, v in history],
        "best_epoch": best_epoch,
        "val": {"baseline": baseline_metrics, "network": val_metrics, "by_distance": val_report},
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    print(f"\nSaved model and run record to {run_dir}/")

