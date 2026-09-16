import itertools

import numpy as np
import pytest

from disaster_mm.dtw import cosine_cost_matrix, dtw


def _brute_force_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Exhaustively enumerate all valid monotonic warping paths from (0,0)
    to (Ta-1,Tb-1) with unit/diagonal steps, and return the minimum cost.
    Only tractable for small inputs -- used purely as a test oracle."""
    ta, tb = a.shape[0], b.shape[0]
    cost = cosine_cost_matrix(a, b)

    best = [np.inf]

    def rec(i: int, j: int, acc: float) -> None:
        acc = acc + cost[i, j]
        if i == ta - 1 and j == tb - 1:
            best[0] = min(best[0], acc)
            return
        if acc >= best[0]:
            return
        if i < ta - 1:
            rec(i + 1, j, acc)
        if j < tb - 1:
            rec(i, j + 1, acc)
        if i < ta - 1 and j < tb - 1:
            rec(i + 1, j + 1, acc)

    rec(0, 0, 0.0)
    return best[0]


@pytest.mark.parametrize("ta,tb", [(1, 1), (2, 3), (3, 2), (4, 4), (3, 5)])
def test_dtw_matches_brute_force(ta, tb):
    rng = np.random.default_rng(ta * 100 + tb)
    a = rng.normal(size=(ta, 4))
    b = rng.normal(size=(tb, 4))
    _, cost = dtw(a, b)
    expected = _brute_force_dtw(a, b)
    assert cost == pytest.approx(expected, abs=1e-6)


def test_dtw_identical_sequences_has_zero_cost():
    a = np.eye(4)[:4]
    path, cost = dtw(a, a.copy())
    assert cost == pytest.approx(0.0, abs=1e-6)
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)


def test_dtw_empty_sequence():
    a = np.zeros((0, 4))
    b = np.random.default_rng(0).normal(size=(3, 4))
    path, cost = dtw(a, b)
    assert path == []
    assert cost == 0.0


def test_dtw_path_is_monotonic():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(5, 3))
    b = rng.normal(size=(6, 3))
    path, _ = dtw(a, b)
    for (i0, j0), (i1, j1) in itertools.pairwise(path):
        assert i1 - i0 in (0, 1)
        assert j1 - j0 in (0, 1)
