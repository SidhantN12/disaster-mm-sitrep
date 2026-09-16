"""Dynamic Time Warping with a cosine-distance local cost.

Used to explicitly align a generated SITREP's sentence embeddings against a
tile's report embeddings (or any two token/sentence embedding sequences),
as a diagnostic/explicit-alignment complement to the learned attention.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def cosine_cost_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (Ta, D), b: (Tb, D) -> cost[i, j] = 1 - cosine(a_i, b_j)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + _EPS)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + _EPS)
    sim = a_norm @ b_norm.T
    return 1.0 - sim


def dtw(a: np.ndarray, b: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Standard DP dynamic time warping.

    Returns (path, cost) where path is a list of (i, j) index pairs from
    (0, 0) to (Ta-1, Tb-1) and cost is the cumulative alignment cost.
    """
    ta, tb = a.shape[0], b.shape[0]
    if ta == 0 or tb == 0:
        return [], 0.0

    cost = cosine_cost_matrix(a, b)
    acc = np.full((ta, tb), np.inf, dtype=np.float64)
    acc[0, 0] = cost[0, 0]
    for i in range(1, ta):
        acc[i, 0] = acc[i - 1, 0] + cost[i, 0]
    for j in range(1, tb):
        acc[0, j] = acc[0, j - 1] + cost[0, j]
    for i in range(1, ta):
        for j in range(1, tb):
            acc[i, j] = cost[i, j] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    i, j = ta - 1, tb - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choices = [(acc[i - 1, j], (i - 1, j)), (acc[i, j - 1], (i, j - 1)), (acc[i - 1, j - 1], (i - 1, j - 1))]
            _, (i, j) = min(choices, key=lambda c: c[0])
        path.append((i, j))
    path.reverse()
    return path, float(acc[ta - 1, tb - 1])
