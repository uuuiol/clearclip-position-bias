"""CPU-only sanity tests over synthetic data — no GPU, no CLIP checkpoint,
no dataset download needed. Run these BEFORE spending GPU time on --stage
extract: `pip install -e ".[test]" && pytest tests/ -q`.
"""

import numpy as np
import pytest

from clearclip import diagnostics as diag
from clearclip.datasets import IGNORE_LABEL


def test_radial_grid_center_is_zero_corner_is_one():
    r = diag.radial_grid(3, 3)
    assert r[1, 1] == pytest.approx(0.0)
    assert r[0, 0] == pytest.approx(1.0)
    assert r[0, 2] == pytest.approx(1.0)
    assert r.max() == pytest.approx(1.0)
    assert r.min() == pytest.approx(0.0)


def test_radial_grid_symmetric():
    r = diag.radial_grid(5, 7)
    assert r == pytest.approx(np.fliplr(r))
    assert r == pytest.approx(np.flipud(r))


def test_quartile_bin_edges():
    edges = diag.quartile_bin_edges(4)
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert len(edges) == 5
    assert np.allclose(np.diff(edges), 0.25)


def test_nearest_upsample_preserves_block_values():
    grid = np.array([[0.0, 1.0], [2.0, 3.0]])
    up = diag.nearest_upsample(grid, 4, 4)
    assert up.shape == (4, 4)
    # each source cell should map to a contiguous 2x2 block of the same value
    assert np.all(up[:2, :2] == 0.0)
    assert np.all(up[:2, 2:] == 1.0)
    assert np.all(up[2:, :2] == 2.0)
    assert np.all(up[2:, 2:] == 3.0)


def test_bin_confusion_perfect_prediction_gives_miou_one():
    n_classes = 3
    bc = diag.BinConfusion(n_bins=2, n_classes=n_classes)
    gt = np.array([[0, 1], [2, 0]])
    pred = gt.copy()
    bin_idx = np.array([[0, 0], [1, 1]])
    bc.update(pred, gt, bin_idx)
    miou = bc.per_bin_miou()
    assert miou[0] == pytest.approx(1.0)
    assert miou[1] == pytest.approx(1.0)


def test_bin_confusion_ignores_255():
    n_classes = 2
    bc = diag.BinConfusion(n_bins=1, n_classes=n_classes)
    gt = np.array([[0, IGNORE_LABEL], [1, 1]])
    pred = np.array([[0, 1], [0, 1]])
    bin_idx = np.zeros_like(gt)
    bc.update(pred, gt, bin_idx)
    miou = bc.per_bin_miou()
    # valid (non-255) positions only: (0,0) gt=0/pred=0 correct,
    # (1,0) gt=1/pred=0 wrong, (1,1) gt=1/pred=1 correct.
    # class 0: intersection=1, union=2 -> IoU 0.5; class 1: intersection=1,
    # union=2 -> IoU 0.5 (the (0,1) cell's pred=1 must NOT count towards
    # class 1's union, since its gt is the ignore label 255).
    assert miou[0] == pytest.approx(0.5)


def test_bootstrap_bias_gap_zero_when_bins_identical():
    per_image_correct = [np.array([8, 8]), np.array([9, 9]), np.array([7, 7])]
    per_image_total = [np.array([10, 10])] * 3
    gap, lo, hi = diag.bootstrap_bias_gap(per_image_correct, per_image_total, n_resamples=200, seed=0)
    assert gap == pytest.approx(0.0)
    assert lo <= 0.0 <= hi


def test_bootstrap_bias_gap_detects_real_gap():
    # bin 0 (center) always perfect, bin -1 (boundary) always wrong
    per_image_correct = [np.array([10, 0])] * 20
    per_image_total = [np.array([10, 10])] * 20
    gap, lo, hi = diag.bootstrap_bias_gap(per_image_correct, per_image_total, n_resamples=500, seed=0)
    assert gap == pytest.approx(1.0)
    assert lo > 0.0  # CI should exclude zero -> "GO" in run.py's go/no-go check


def test_bootstrap_paired_delta_zero_when_conditions_identical():
    correct = [np.array([8, 3])] * 10
    total = [np.array([10, 10])] * 10
    point, lo, hi = diag.bootstrap_paired_delta(correct, total, correct, total, n_resamples=200, seed=0)
    assert point == pytest.approx(0.0)
    assert lo <= 0.0 <= hi


def test_bootstrap_paired_delta_detects_consistent_improvement():
    # baseline: boundary bin (index -1) always wrong; calibrated: always fixed.
    # Both conditions share identical center-bin (index 0) behavior and
    # per-image variability, which a paired test should cancel out even
    # though a naive per-condition bootstrap would call both "noisy".
    rng = np.random.default_rng(0)
    n_images = 30
    center_correct = rng.integers(4, 9, size=n_images)  # noisy but shared
    base_correct = [np.array([c, 0]) for c in center_correct]
    cal_correct = [np.array([c, 10]) for c in center_correct]
    total = [np.array([10, 10])] * n_images

    point, lo, hi = diag.bootstrap_paired_delta(base_correct, total, cal_correct, total, n_resamples=500, seed=0)
    # gap = acc[0]-acc[-1]; baseline gap ~ center_acc - 0; calibrated gap ~ center_acc - 1
    # delta = calibrated_gap - baseline_gap = -1.0 always, regardless of the
    # shared noisy center accuracy -> should be a tight, clearly-negative CI.
    assert point == pytest.approx(-1.0)
    assert hi < 0.0


def test_attention_centrality_uniform_attention_is_constant():
    n = 5
    r = np.linspace(0, 1, n)
    uniform_attn = np.full((n, n), 1.0 / n)
    c = diag.attention_centrality(uniform_attn, r)
    expected = np.mean(1.0 - r)
    assert np.allclose(c, expected)


def test_attention_centrality_center_biased_attention_favors_center_queries():
    # every query attends only to the single most-central key (r=0) -> c_i
    # should be identical (and maximal) for every query i, since it's always
    # (1 - 0) = 1 regardless of the query's own position. This is the
    # degenerate "total hub" case the diagnostic is designed to catch.
    n = 4
    r = np.array([0.0, 0.3, 0.6, 1.0])
    hub_attn = np.zeros((n, n))
    hub_attn[:, 0] = 1.0  # every row attends 100% to patch 0 (the center)
    c = diag.attention_centrality(hub_attn, r)
    assert np.allclose(c, 1.0)


def test_object_level_correlation_pools_components_and_computes_rho():
    # Filler value 99 is outside range(n_classes) so it's never matched by
    # `gt == c` — keeps components to exactly the three blobs placed below,
    # avoiding an ambiguous giant "background" component. Three components is
    # also the minimum object_level_correlation requires before it computes a
    # (non-nan) rho at all.
    n_classes = 2
    gt = np.full((30, 30), 99, dtype=np.int64)
    gt[1:6, 1:6] = 0         # top-left corner, class 0, high r
    gt[24:29, 24:29] = 0     # bottom-right corner, class 0, equally high r
    gt[12:17, 12:17] = 1     # near the center, class 1, low r
    r_pixel = diag.radial_grid(30, 30)

    pred = gt.copy()
    pred[1:6, 1:6] = 1       # only the top-left object is mispredicted

    centroid_r, obj_acc, rho, p = diag.object_level_correlation(gt, pred, r_pixel, n_classes)
    assert len(centroid_r) == 3
    assert obj_acc.min() == pytest.approx(0.0)   # the mispredicted corner object
    assert (obj_acc == 1.0).sum() == 2           # the other two are perfect
    assert rho < 0  # higher centroid distance <-> lower accuracy, on average
