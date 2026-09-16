import numpy as np

from disaster_mm.models.cca_baseline import cca_baseline_report


def test_cca_baseline_high_correlation_for_linearly_related_features():
    rng = np.random.default_rng(0)
    n, d = 40, 6
    z = rng.normal(size=(n, d))
    h_i = z @ rng.normal(size=(d, 5))
    h_t = z @ rng.normal(size=(d, 5))  # both derived from the same latent z
    report = cca_baseline_report(h_i, h_t, n_components=2)
    assert report["mean_canonical_correlation"] > 0.5


def test_cca_baseline_low_correlation_for_independent_features():
    rng = np.random.default_rng(1)
    n = 30
    h_i = rng.normal(size=(n, 5))
    h_t = rng.normal(size=(n, 5))
    report = cca_baseline_report(h_i, h_t, n_components=2)
    assert report["mean_canonical_correlation"] < 0.9
