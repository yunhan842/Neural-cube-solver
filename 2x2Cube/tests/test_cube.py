import numpy as np
import pytest

from cube import (
    Kociemba2x2, MOVE_NAMES, MOVE_CP, MOVE_CO, SOLVED_PERMUTATION, SOLVED_ORIENTATION,
    compose, apply_batch_moves, random_scrambles, encode_onehot
)

def test_new_cube_is_solved_and_valid():
    cube = Kociemba2x2()
    assert cube.is_solved() and cube.is_valid()

@pytest.mark.parametrize("move", MOVE_NAMES)
def test_every_move_gives_valid_unsolved_state(move):
    cube = Kociemba2x2().apply_move(move)
    assert cube.is_valid() and not cube.is_solved()

@pytest.mark.parametrize("face", ["U", "R", "F"])
def test_four_quarter_turns_return_to_solved(face):
    assert Kociemba2x2().apply_sequence([face] * 4).is_solved()

@pytest.mark.parametrize("face", ["U", "R", "F"])
def test_move_then_inverse_returns_to_solved(face):
    assert Kociemba2x2().apply_sequence(f"{face} {face}'").is_solved()

@pytest.mark.parametrize("face", ["U", "R", "F"])
def test_two_quarter_turns_equal_half_turn(face):
    assert Kociemba2x2().apply_sequence(f"{face} {face}") == Kociemba2x2().apply_move(face + "2")

def test_r_u_r_prime_u_prime_has_order_6():
    cube = Kociemba2x2()
    for i in range(1, 7):
        cube.apply_sequence("R U R' U'")
        assert cube.is_solved() == (i == 6)

def test_scramble_then_inverse_returns_to_solved():
    scramble = "R U F' U2 R' F2 U' R2 F U2 F' R U' R'"
    inverse = "R U R' F U2 F' R2 U F2 R U2 F U' R'"
    assert Kociemba2x2().apply_sequence(scramble).apply_sequence(inverse).is_solved()

def test_string_and_list_sequences_agree():
    assert Kociemba2x2().apply_sequence("R U2 F'") == Kociemba2x2().apply_sequence(["R", "U2", "F'"])

def test_constructor_defaults_each_array():
    only_cp = Kociemba2x2(cp=MOVE_CP[3])
    assert np.array_equal(only_cp.cp, MOVE_CP[3]) and not only_cp.co.any()
    only_co = Kociemba2x2(co=[1, 2, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(only_co.cp, SOLVED_PERMUTATION)
    assert only_co.co.tolist() == [1, 2, 0, 0, 0, 0, 0, 0]

def test_copy_is_independent():
    a = Kociemba2x2().apply_move("R")
    b = a.copy()
    b.apply_move("U")
    b.co[0] = 1
    assert a == Kociemba2x2().apply_move("R")

def test_in_place_edits_do_not_change_solved_constants():
    cube = Kociemba2x2()
    cube.cp[0], cube.co[0] = 5, 1
    cube.reset()
    cube.co[1] = 2
    assert np.array_equal(SOLVED_PERMUTATION, np.arange(8))
    assert not SOLVED_ORIENTATION.any()

@pytest.mark.parametrize("cp, co", [
    (None, [1, 0, 0, 0, 0, 0, 0, 0]), # one twisted corner
    (None, [3, 0, 0, 0, 0, 0, 0, 0]), # twist out of range (sum is still 0 mod 3)
    ([0, 0, 2, 3, 4, 5, 6, 7], None), # piece 0 twice, piece 1 missing
    ([7, 1, 2, 3, 4, 5, 6, 0], None), # DBL moved
    (None, [2, 0, 0, 0, 0, 0, 0, 1]), # DBL twisted
])
def test_is_valid_rejects_impossible_states(cp, co):
    assert not Kociemba2x2(cp, co).is_valid()

def test_batch_moves_match_single_moves():
    rng = np.random.default_rng(0)
    cp, co = random_scrambles(100, 10, rng)
    moves = rng.integers(0, 9, size=100)
    bcp, bco = apply_batch_moves(cp, co, moves)
    for i in range(100):
        single = Kociemba2x2(cp[i], co[i]).apply_move(int(moves[i]))
        assert single == Kociemba2x2(bcp[i], bco[i])

def test_same_move_applied_to_whole_batch():
    cp, co = random_scrambles(20, 10, np.random.default_rng(1))
    bcp, bco = compose(cp, co, MOVE_CP[3], MOVE_CO[3])
    for i in range(20):
        assert Kociemba2x2(cp[i], co[i]).apply_move("R") == Kociemba2x2(bcp[i], bco[i])

def test_scrambles_are_valid():
    cp, co = random_scrambles(500, 20, np.random.default_rng(2))
    assert all(Kociemba2x2(cp[i], co[i]).is_valid() for i in range(500))

def test_zero_length_scramble_is_solved():
    cp, co = random_scrambles(5, 0)
    assert all(Kociemba2x2(cp[i], co[i]).is_solved() for i in range(5))

def test_negative_scramble_length_raises():
    with pytest.raises(ValueError):
        random_scrambles(5, -1)

def test_onehot_encoding():
    cp, co = random_scrambles(50, 10, np.random.default_rng(3))
    x = encode_onehot(cp, co)
    assert x.shape == (50, 88) and x.dtype == np.float32
    assert (x.sum(axis=1) == 16).all()
    solved = encode_onehot(SOLVED_PERMUTATION[None], SOLVED_ORIENTATION[None])
    # Position 0 holds piece 0 with twist 0
    assert solved[0, :11].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]