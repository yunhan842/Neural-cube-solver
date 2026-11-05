import numpy as np

# Kociemba Corner Indices mapping:
# 0: URF, 1: UFL, 2: ULB, 3: UBR, 4: DRB, 5: DFR, 6: DLF, 7: DBL
# Keep DBL fixed as we only simulate U, R, F to reach any state
URF, UFL, ULB, UBR, DRB, DFR, DLF, DBL = range(8)
SOLVED_PERMUTATION = np.arange(8, dtype=np.int8) # [0, 1, 2, 3, 4, 5, 6, 7]
SOLVED_ORIENTATION = np.zeros(8, dtype=np.int8) # Solved orientation is 0 for each piece 

# Quarter turns from cubie.py in Kociemba. Position i holds the corner
# that was at cp[i] and corner's orientation increases by co[i] modulo 3
QUARTER_TURNS = {
    "U": ([UBR, URF, UFL, ULB, DRB, DFR, DLF, DBL], np.zeros(8, dtype=np.int8)),
    "R": ([DFR, UFL, ULB, URF, UBR, DRB, DLF, DBL], [2, 0, 0, 1, 2, 1, 0, 0]),
    "F": ([UFL, DLF, ULB, UBR, DRB, URF, DFR, DBL], [1, 2, 0, 0, 0, 2, 1, 0])
}

# Apply move (move_cp, move_co) to an existing state (cp, co) and should work on (8,) or (N,8) arrays
def compose(cp, co, move_cp, move_co):
    indices = np.broadcast_to(np.asarray(move_cp), cp.shape)
    new_cp = np.take_along_axis(cp, indices, axis=-1)
    new_co = np.take_along_axis(co, indices, axis=-1)
    return new_cp, (new_co + move_co) % 3

# Build each of the possible moves (U,U2,U',R,R2,R',F,F2,F') along with their resulting permutation/orientation
def build_move_tables():
    names, cps, cos = [], [], []
    for face, (qcp, qco) in QUARTER_TURNS.items():
        qcp, qco = np.array(qcp, dtype=np.int8), np.array(qco, dtype=np.int8)
        cp, co = SOLVED_PERMUTATION, SOLVED_ORIENTATION
        for suffix in ("", "2", "'"):
            cp, co = compose(cp, co, qcp, qco)
            names.append(face + suffix)
            cps.append(cp)
            cos.append(co)
    return names, np.stack(cps), np.stack(cos)

# Kociemba move enum -> U1, U2, U3, R1, R2, R3, F1, F2, F3
MOVE_NAMES, MOVE_CP, MOVE_CO = build_move_tables()
MOVE_IDX = {name: i for i, name in enumerate(MOVE_NAMES)}
MOVE_FACE = np.arange(len(MOVE_NAMES)) // 3

class Kociemba2x2:
    def __init__(self, cp=None, co=None):
        self.cp = SOLVED_PERMUTATION.copy() if cp is None else np.array(cp, dtype=np.int8)
        self.co = SOLVED_ORIENTATION.copy() if co is None else np.array(co, dtype=np.int8)

    def reset(self):
        # Copy so cubes never share/mutate module level arrays
        self.cp = SOLVED_PERMUTATION.copy()
        self.co = SOLVED_ORIENTATION.copy()

    def copy(self):
        return Kociemba2x2(self.cp, self.co)

    def apply_move(self, move):
        # Move should be a name ("R", "U2", "F'") or an index into MOVE_NAMES
        m = MOVE_IDX[move] if isinstance(move, str) else move
        self.cp, self.co = compose(self.cp, self.co, MOVE_CP[m], MOVE_CO[m])
        return self

    def apply_sequence(self, sequence):
        # A sequence expects something like ["R", "U2", "F'"] or "R U2 F'"
        if isinstance(sequence, str):
            sequence = sequence.split()
        for move in sequence:
            self.apply_move(move)
        return self

    def is_solved(self):
        return np.array_equal(self.cp, SOLVED_PERMUTATION) and not self.co.any()

    def is_valid(self):
        return (
            self.cp.shape == (8,) and self.co.shape == (8,)
            and sorted(self.cp.tolist()) == list(range(8))
            and ((self.co >= 0) & (self.co <= 2)).all()
            and self.co.sum() % 3 == 0
            and self.cp[DBL] == DBL
            and self.co[DBL] == 0
        )

    def key(self):
        # Hash state for dicts/sets e.g. doing BFS over all states
        return self.cp.tobytes() + self.co.tobytes()

    @classmethod
    def from_key(cls, key):
        # Inverse of key(): the first 8 bytes are cp and last 8 are co
        return cls(np.frombuffer(key[:8], dtype=np.int8), np.frombuffer(key[8:], dtype=np.int8))

    def __eq__(self, other):
        return np.array_equal(self.cp, other.cp) and np.array_equal(self.co, other.co)

    def __repr__(self):
        return f"Kociemba2x2(cp={self.cp.tolist()}, co={self.co.tolist()})"

def apply_batch_moves(cp, co, moves):
    # Apply moves[i] (idx into MOVE_NAMES) to cube i
    return compose(cp, co, MOVE_CP[moves], MOVE_CO[moves])

def random_scrambles(n, length, rng=None):
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    # n random scrambles of given length (never turns same face twice in a row)
    rng = np.random.default_rng() if rng is None else rng
    # N cubes in solved state
    cp = np.tile(SOLVED_PERMUTATION, (n, 1))
    co = np.tile(SOLVED_ORIENTATION, (n, 1))
    if length == 0:
        return cp, co
    moves = rng.integers(0, 9, size=n)
    cp, co = apply_batch_moves(cp, co, moves)
    for _ in range(length - 1):
        next = rng.integers(0, 6, size=n)
        moves = np.where(next // 3 >= MOVE_FACE[moves], next + 3, next)
        cp, co = apply_batch_moves(cp, co, moves)
    return cp, co

def encode_onehot(cp, co):
    # (N,8) cp/co -> (N,88) float32 network input: per position, one-hot corner (8) + twist (3)
    n = cp.shape[0]
    perm = np.eye(8, dtype=np.float32)[cp]
    twist = np.eye(3, dtype=np.float32)[co]
    return np.concatenate([perm, twist], axis=2).reshape(n, 88)