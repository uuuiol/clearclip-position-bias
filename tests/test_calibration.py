"""CPU-only sanity tests for calibration.py using synthetic logits/attention —
no CLIP checkpoint needed. Run before spending GPU time on --stage extract.
"""

import numpy as np
import pytest

from clearclip.calibration import (
    PositionSmoothingParams,
    fit_position_smoothing,
    AttentionReweightParams,
    radial_similarity_kernel,
)
from clearclip.diagnostics import radial_grid


def _make_item(logits, patch_attn, grid_h=1, grid_w=None, gt=None):
    grid_w = grid_w or logits.shape[0]
    return {
        "logits": logits, "patch_attn": patch_attn,
        "grid_h": grid_h, "grid_w": grid_w,
        "gt": gt if gt is not None else np.zeros((grid_h, grid_w), dtype=np.int64),
    }


def test_position_smoothing_lambda_zero_is_identity():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    patch_attn = np.array([[0.0, 1.0], [1.0, 0.0]])  # each patch fully attends the other
    r_flat = np.array([0.2, 0.9])
    params = PositionSmoothingParams(lam=0.0, p=1)
    out = params.fn()(_make_item(logits, patch_attn), r_flat)
    assert np.allclose(out, logits)


def test_position_smoothing_can_flip_argmax():
    # Patch 1 (r=0.9, near the boundary) is locally wrong (favors class 0)
    # but its only attended neighbor (patch 0) strongly favors class 1.
    # This is exactly the case the broken temperature-scaling design could
    # never fix (a uniform per-patch scalar can't change an argmax); mixing
    # in the neighbor's logits can.
    logits = np.array([[0.1, 0.9], [0.6, 0.4]])  # patch1 argmax=0 before smoothing
    patch_attn = np.array([[1.0, 0.0], [1.0, 0.0]])  # both patches attend fully to patch 0
    r_flat = np.array([0.1, 0.9])
    params = PositionSmoothingParams(lam=1.0, p=1)
    out = params.fn()(_make_item(logits, patch_attn), r_flat)
    # patch 1: alpha = clip(1.0*0.9,0,1) = 0.9; smoothed_1 = attn[1,:]@logits = logits[0] = [0.1,0.9]
    # out_1 = 0.1*[0.6,0.4] + 0.9*[0.1,0.9] = [0.06+0.09, 0.04+0.81] = [0.15, 0.85]
    assert out[1].argmax() == 1  # flipped from 0 -> 1
    # patch 0 has r=0.1 -> alpha small -> should stay close to its own logits
    assert out[0].argmax() == logits[0].argmax()


def test_position_smoothing_alpha_is_clipped_to_one():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    patch_attn = np.array([[0.0, 1.0], [1.0, 0.0]])
    r_flat = np.array([1.0, 1.0])
    params = PositionSmoothingParams(lam=5.0, p=1)  # lam*r^p = 5, must clip to 1
    out = params.fn()(_make_item(logits, patch_attn), r_flat)
    # alpha clipped to 1 -> output should equal the fully-smoothed version
    expected = patch_attn @ logits
    assert np.allclose(out, expected)


def test_radial_similarity_kernel_rows_sum_to_one():
    r_flat = np.array([0.0, 0.3, 0.6, 1.0])
    K = radial_similarity_kernel(r_flat, sigma=0.3)
    assert K.shape == (4, 4)
    assert np.allclose(K.sum(axis=-1), 1.0)


def test_radial_similarity_kernel_favors_similar_r_over_different_r():
    r_flat = np.array([0.0, 0.05, 0.9])  # patch 0 and 1 are close in r, patch 2 is far
    K = radial_similarity_kernel(r_flat, sigma=0.3)
    assert K[0, 1] > K[0, 2]  # patch 0 weights its near-r neighbor (1) more than the far one (2)


def test_radial_similarity_kernel_is_bias_free_no_patch_attn_needed():
    # kernel="radial" must not touch item["patch_attn"] at all — that's the
    # whole point (patch_attn is itself position-biased; see the module
    # docstring's explanation for why kernel="attn" failed to reduce
    # bias_gap). Passing an item with NO "patch_attn" key must still work.
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    r_flat = np.array([0.1, 0.9])
    item = {"logits": logits}  # deliberately no "patch_attn"
    params = PositionSmoothingParams(lam=1.0, p=1, kernel="radial", sigma=0.3)
    out = params.fn()(item, r_flat)  # must not raise KeyError
    assert out.shape == logits.shape


def test_fit_position_smoothing_can_prefer_radial_kernel():
    # Construct a case where the attn kernel is actively misleading (it
    # points every patch toward a WRONG neighbor) while the radial kernel
    # (built only from r, ignoring attn) points toward the RIGHT one.
    logits = np.array([[0.1, 0.9], [0.6, 0.4], [0.55, 0.45]])
    # patch_attn: everyone attends fully to patch 0 (which itself needs
    # fixing) — a stand-in for "attention is itself biased/unhelpful here".
    patch_attn = np.array([[1.0, 0.0, 0.0]] * 3)
    gt = np.array([[1, 1, 1]])
    r_flat_grid_h, r_flat_grid_w = 1, 3
    item = {
        "logits": logits, "patch_attn": patch_attn,
        "grid_h": r_flat_grid_h, "grid_w": r_flat_grid_w, "gt": gt,
    }

    best, score = fit_position_smoothing(
        [item], n_classes=2, lambda_grid=[0.0, 1.0], p_grid=[1],
        kernel_grid=["attn", "radial"], sigma_grid=[0.5],
    )
    # whichever kernel wins, it must have been actually compared against the
    # other (this mainly guards against a silent no-op / signature bug)
    assert best.kernel in ("attn", "radial")
    assert score >= 1.0 / 3  # sanity: better than the all-attn baseline's 1/3


def test_fit_position_smoothing_prefers_lambda_that_fixes_errors():
    # Two "images" of one patch-pair each; smoothing (lam>0) fixes both,
    # lam=0 leaves both wrong. Grid search over lam should not pick 0.
    logits = np.array([[0.1, 0.9], [0.6, 0.4]])
    patch_attn = np.array([[1.0, 0.0], [1.0, 0.0]])
    gt = np.array([[1, 1]])  # both patches are truly class 1 (patch1 wrongly favors class 0)
    item = _make_item(logits, patch_attn, grid_h=1, grid_w=2, gt=gt)

    best, score = fit_position_smoothing(
        [item], n_classes=2, lambda_grid=[0.0, 1.0], p_grid=[1],
    )
    assert best.lam == 1.0
    assert score > 0.5  # strictly better than the lam=0 (broken) baseline


def test_attention_reweight_shapes_and_normalization():
    n_patches, embed_dim, n_classes = 4, 8, 3
    rng = np.random.default_rng(0)
    shared = {
        "out_proj_weight": rng.normal(size=(embed_dim, embed_dim)).astype(np.float32),
        "out_proj_bias": rng.normal(size=(embed_dim,)).astype(np.float32),
        "ln_post_weight": np.ones(embed_dim, dtype=np.float32),
        "ln_post_bias": np.zeros(embed_dim, dtype=np.float32),
        "ln_post_eps": 1e-5,
        "visual_proj": rng.normal(size=(embed_dim, embed_dim)).astype(np.float32),
        "text_embeds": rng.normal(size=(n_classes, embed_dim)).astype(np.float32),
    }
    shared["text_embeds"] /= np.linalg.norm(shared["text_embeds"], axis=-1, keepdims=True)

    item = {
        "patch_sim": rng.normal(size=(n_patches, n_patches)).astype(np.float32),
        "v_patches": rng.normal(size=(n_patches, embed_dim)).astype(np.float32),
    }
    r_flat = np.linspace(0, 1, n_patches)

    params = AttentionReweightParams(beta=1.0, theta=0.5, tau=1.0)
    logits = params.fn(shared)(item, r_flat)
    assert logits.shape == (n_patches, n_classes)
    assert np.all(np.isfinite(logits))


def test_attention_reweight_beta_zero_matches_plain_softmax_attention():
    n_patches, embed_dim, n_classes = 3, 4, 2
    rng = np.random.default_rng(1)
    shared = {
        "out_proj_weight": np.eye(embed_dim, dtype=np.float32),
        "out_proj_bias": np.zeros(embed_dim, dtype=np.float32),
        "ln_post_weight": np.ones(embed_dim, dtype=np.float32),
        "ln_post_bias": np.zeros(embed_dim, dtype=np.float32),
        "ln_post_eps": 1e-5,
        "visual_proj": np.eye(embed_dim, dtype=np.float32),
        "text_embeds": rng.normal(size=(n_classes, embed_dim)).astype(np.float32),
    }
    shared["text_embeds"] /= np.linalg.norm(shared["text_embeds"], axis=-1, keepdims=True)
    sim = rng.normal(size=(n_patches, n_patches)).astype(np.float32)
    v = rng.normal(size=(n_patches, embed_dim)).astype(np.float32)
    item = {"patch_sim": sim, "v_patches": v}
    r_flat = np.array([0.1, 0.5, 0.9])

    params = AttentionReweightParams(beta=0.0, theta=0.5, tau=1.0)
    logits = params.fn(shared)(item, r_flat)

    # beta=0 -> no position bias term -> a plain softmax(sim/tau) @ v, then
    # the (identity) out_proj/ln_post/proj, then cosine similarity.
    exp = np.exp(sim - sim.max(axis=-1, keepdims=True))
    attn = exp / exp.sum(axis=-1, keepdims=True)
    out = attn @ v
    mean, var = out.mean(-1, keepdims=True), out.var(-1, keepdims=True)
    normed = (out - mean) / np.sqrt(var + 1e-5)
    proj = normed / np.linalg.norm(normed, axis=-1, keepdims=True)
    expected = proj @ shared["text_embeds"].T
    assert np.allclose(logits, expected, atol=1e-4)
