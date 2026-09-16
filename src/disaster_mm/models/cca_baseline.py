"""CCA baseline / diagnostic for cross-modal alignment.

A classical (non-deep) baseline for comparison against the learned shared
encoder: fit ``sklearn.cross_decomposition.CCA`` on pooled image/text
features and report the canonical correlations. The same routine doubles as
a diagnostic you can run on the trained model's ``z_s^I`` / ``z_s^T``
outputs to check how linearly aligned the learned shared space already is
(a well-trained shared encoder should make even a linear CCA on top of it
show high canonical correlation).
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA


def canonical_correlations(x: np.ndarray, y: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Fit CCA(x, y) and return the per-component Pearson correlation."""
    n_components = min(n_components, x.shape[1], y.shape[1], x.shape[0] - 1)
    n_components = max(n_components, 1)
    cca = CCA(n_components=n_components)
    x_c, y_c = cca.fit_transform(x, y)
    corrs = np.array(
        [np.corrcoef(x_c[:, i], y_c[:, i])[0, 1] for i in range(n_components)]
    )
    return np.nan_to_num(corrs)


def cca_baseline_report(h_i: np.ndarray, h_t: np.ndarray, n_components: int = 2) -> dict:
    corrs = canonical_correlations(h_i, h_t, n_components)
    return {
        "canonical_correlations": corrs.tolist(),
        "mean_canonical_correlation": float(np.mean(corrs)) if len(corrs) else 0.0,
    }
