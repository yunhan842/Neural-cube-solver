import sys
from pathlib import Path

import numpy as np

from search import bfs

# Turn bfs() table into training arrays and save/load as .npz files

def table_to_arrays(dist):
    # bfs() table -> cp (N,8), co (N,8), dist (N,), one row per state
    # Every key is 16 bytes (8 from cp and 8 from co), so all keys joined form (N,16) array
    raw = np.frombuffer(b"".join(dist.keys()), dtype=np.int8).reshape(-1,16)
    cp = raw[:, :8].copy()
    co = raw[:, 8:].copy()
    d = np.fromiter(dist.values(), dtype=np.int8, count=len(dist))
    return cp, co, d

def require_npz(path):
    # np.savez adds ".npz" to names without it but np.load does not so only accept .npz
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"dataset path must end in .npz, got {str(path)!r}")
    return path

def save_dataset(path, cp, co, dist, max_depth):
    path = require_npz(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cp=cp, co=co, dist=dist, max_depth=max_depth)

def load_dataset(path):
    # Return cp, co, dist, max_depth
    with np.load(require_npz(path)) as data:
        return data["cp"], data["co"], data["dist"], int(data["max_depth"])

if __name__ == "__main__":
    # Use: python dataset.py 7 data/train.npz
    max_depth = int(sys.argv[1])
    # Check path name before doing bfs()
    path = require_npz(sys.argv[2])
    dist, _, layer_counts = bfs(max_depth)
    cp, co, d = table_to_arrays(dist)
    save_dataset(path, cp, co, d, max_depth)
    print(f"Saved {len(d):,} states to {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")
    print(f"States per distance: {np.bincount(d).tolist()}")
