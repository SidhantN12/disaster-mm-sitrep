import numpy as np

from disaster_mm.data.synthetic import SyntheticConfig, generate_dataset, generate_tile


def test_tile_shapes():
    rng = np.random.default_rng(0)
    cfg = SyntheticConfig(grid_rows=3, grid_cols=3)
    tile = generate_tile(rng, cfg)
    assert tile.image.shape == (6, 256, 256)
    assert tile.image.dtype == np.float32
    assert tile.image.min() >= 0.0 and tile.image.max() <= 1.0
    n_struct = 3 * 3 + 1
    assert tile.building_masks.shape == (n_struct, 256, 256)
    assert tile.building_damage.shape == (n_struct,)
    assert tile.damage in (0, 1, 2, 3)
    assert tile.bag_label in (0, 1)
    assert "Overall damage level" in tile.sitrep


def test_k0_tiles_exist_and_have_no_nans():
    cfg = SyntheticConfig(prob_zero_reports=1.0)  # force k=0 every time
    dataset = generate_dataset(5, seed=0, cfg=cfg)
    assert all(len(t.reports) == 0 for t in dataset)
    for t in dataset:
        assert t.bag_label == 0
        assert not np.isnan(t.image).any()


def test_reports_within_token_budget():
    cfg = SyntheticConfig(prob_zero_reports=0.0, max_reports=8)
    dataset = generate_dataset(10, seed=2, cfg=cfg)
    for t in dataset:
        assert len(t.reports) <= 8
        for r in t.reports:
            assert isinstance(r.text, str) and len(r.text) > 0


def test_distractor_reports_can_make_bag_label_zero_even_with_k_gt_0():
    cfg = SyntheticConfig(prob_zero_reports=0.0, prob_distractor=1.0, max_reports=3, min_reports=1)
    dataset = generate_dataset(10, seed=3, cfg=cfg)
    assert any(len(t.reports) > 0 and t.bag_label == 0 for t in dataset)


def test_deterministic_given_seed():
    cfg = SyntheticConfig()
    a = generate_dataset(3, seed=42, cfg=cfg)
    b = generate_dataset(3, seed=42, cfg=cfg)
    for ta, tb in zip(a, b):
        assert np.array_equal(ta.image, tb.image)
        assert ta.sitrep == tb.sitrep
