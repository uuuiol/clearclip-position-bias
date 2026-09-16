"""Phase 2 calibration methods — both training-free (no gradient descent
anywhere in this file; every parameter is fit by grid search on a small
held-out CALIBRATION split, disjoint from the split used for final reported
eval — see datasets.split_calibration).

Method A (primary): position-conditioned attention smoothing.

    CORRECTION (caught before any GPU run, so worth stating explicitly): the
    original design here was `s'_{i,c} = s_{i,c} / T(r_i)` — a per-patch
    scalar dividing every class's score equally. That is a positive affine
    rescaling applied uniformly across c for fixed i, so argmax_c s'_{i,c} ==
    argmax_c s_{i,c} ALWAYS — it cannot change a single predicted label, and
    therefore cannot move mIoU at all (it would only have been useful for
    confidence calibration / ECE, not for an argmax-based metric). Fixed by
    making the correction mix in *other patches'* class scores, which can
    change which class wins:

        s'_i = (1 - alpha_i) * s_i + alpha_i * (patch_attn[i, :] @ s)
        alpha_i = clip(lambda * r_i^p, 0, 1)

    patch_attn is the already-cached self-self attention (rows sum to 1), so
    this reuses it as a content-adaptive smoothing kernel: boundary patches
    (high r_i) lean more on an attention-weighted average of the whole image's
    class scores, borrowing signal the diagnostic hypothesizes they lack
    locally. No new cache fields needed for Method A.

Method B (ablation): self-self attention reweighting, re-deriving the last
block's softmax from the cached pre-softmax similarity so boundary query
patches (r_i > theta) attend more to other nearby (similar-r) patches instead
of being pulled toward the interior:
    A'_{ij} = softmax_j( sim_{ij}/tau + beta * 1[r_i>theta] * (1-|r_i-r_j|) )
This one operates on value vectors before projection, not on final class
scores, so it does not have the Method-A argmax-invariance problem above.

Method A, round 2 (kernel fix): a 5-seed run showed Method A's mIoU gain
(+0.3 to +0.7pp, significant in all 5 seeds) does NOT come from shrinking the
center-vs-boundary bias_gap (paired-bootstrap delta was non-significant in
all 5 seeds, and point estimates were mostly *positive*, i.e. the gap if
anything widened slightly). The likely reason: patch_attn — used as the
smoothing kernel — is itself position-biased (attention_centrality_vs_position
rho=-0.72 from the Phase-1 diagnostic). So a boundary patch "borrowing from
its attended neighbors" mostly borrows from center content, not from other
boundary patches, inheriting the very bias it's meant to correct; center
patches get a generic denoising benefit from smoothing towards each other,
which explains the overall mIoU gain without a bias_gap gain.

`PositionSmoothingParams.kernel` fixes this by swapping the smoothing source
for one built purely from patch geometry, with NO dependence on the model's
own (biased) attention:
    kernel="radial":  K_ij ~ exp(-(r_i - r_j)^2 / 2*sigma^2), row-normalized
                       — patches at a similar distance from the image center
                       smooth toward each other (mirrors Method B's
                       (1 - |r_i - r_j|) bias term, but as the entire kernel
                       here rather than an attention-logit addend).
    kernel="attn"  :   the original patch_attn-based kernel, kept for direct
                       comparison in the same grid search.
Needs no new cache fields or re-extraction — r_flat/logits are already cached.

RESULT: kernel="radial" made bias_gap significantly WORSE in all 5 seeds
(paired-bootstrap CI entirely positive every time), and grid search always
picked the narrowest sigma available. Reading together with Method A(attn)'s
non-significant-but-mostly-positive result and Method B's beta->0 preference:
all three independently-designed corrections share one property — they all
correct a patch's score by MIXING IT WITH OTHER PATCHES' scores (attention-
weighted, radial-weighted, or value-vector-weighted). Mixing/averaging is a
variance-reduction operation: it cancels INDEPENDENT noise, which is why
overall mIoU rises (most of an image is spatially coherent, so smoothing
toward any nearby patch reinforces an already-correct majority). But the
diagnosed problem is a BIAS — a systematic, reproducible, same-direction
error at high r — and averaging several instances of the same systematic
error does not cancel it (radial's own failure mode: boundary patches mixed
with OTHER boundary patches, which share the same bias, if anything
reinforces it, matching the observed sign). Bias needs bias-correction, not
variance-reduction. Method C below is designed for that distinction.

Method C: Radial Logit Adjustment (RLA). Transplants the standard
class-imbalance "logit adjustment" technique (Menon et al., ICLR 2021 —
originally: correct a classifier's known train/test class-prior mismatch by
subtracting log(train_prior/test_prior) from each class's logit) onto the
r-axis instead of the class-frequency axis. Unlike Methods A/B, this never
mixes one patch's score with another patch's score at all — it estimates,
from the calibration split, how much each CLASS is systematically over- or
under-predicted at each r-bin (a population-level statistic, exactly the
kind of aggregate, reproducible pattern a "bias" is), and subtracts that
estimated distortion directly:

    delta_{k,c} = log(true_freq_{k,c} + eps) - log(pred_freq_{k,c} + eps)
    s'_{i,c}    = s_{i,c} + gamma * delta_{bin(r_i), c}

This is class-differentiated (unlike the original broken temperature-scaling
idea), so it can change argmax; it needs no cache beyond what's already
there; and it targets the systematic component the diagnostic actually
measured, rather than trying to smooth it away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import IGNORE_LABEL
from .diagnostics import radial_grid, nearest_upsample, quartile_bin_edges
from .eval import evaluate_split


def radial_similarity_kernel(r_flat: np.ndarray, sigma: float) -> np.ndarray:
    """Row-normalized Gaussian kernel over |r_i - r_j|. Built purely from
    patch position — no dependence on patch_attn, so it can't inherit
    attention's own position bias the way the original Method A kernel does.
    """
    diff = r_flat[:, None] - r_flat[None, :]
    k = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    return k / k.sum(axis=-1, keepdims=True)


@dataclass
class PositionSmoothingParams:
    lam: float
    p: float
    kernel: str = "attn"    # "attn" (original) or "radial" (bias-free, see module docstring)
    sigma: float = 0.3      # only used when kernel == "radial"

    def fn(self):
        lam, p, kernel, sigma = self.lam, self.p, self.kernel, self.sigma
        def _apply(item: dict, r_flat: np.ndarray) -> np.ndarray:
            alpha = np.clip(lam * np.power(r_flat, p), 0.0, 1.0)
            if kernel == "attn":
                K = item["patch_attn"]
            elif kernel == "radial":
                K = radial_similarity_kernel(r_flat, sigma)
            else:
                raise ValueError(f"unknown kernel: {kernel!r}")
            smoothed = K @ item["logits"]  # (n_patches, n_classes)
            return (1.0 - alpha[:, None]) * item["logits"] + alpha[:, None] * smoothed
        return _apply


def fit_position_smoothing(
    calib_items: list[dict],
    n_classes: int,
    lambda_grid: list[float],
    p_grid: list[float],
    kernel_grid: list[str] = ("attn",),
    sigma_grid: list[float] = (0.3,),
) -> tuple[PositionSmoothingParams, float]:
    """Grid search (lambda, p, kernel[, sigma]) maximizing overall mIoU on the
    calibration split — deliberately NOT bias_gap, so that any bias_gap
    improvement measured afterward on the eval split is an honest downstream
    check rather than the thing directly optimized for. Returns (best_params,
    best_calib_miou). sigma is only varied for kernel == "radial" (looping it
    for "attn" would just re-evaluate the same identical model n times).
    """
    best = None
    best_score = -1.0
    for kernel in kernel_grid:
        sigmas = sigma_grid if kernel == "radial" else (sigma_grid[0],)
        for lam in lambda_grid:
            for p in p_grid:
                for sigma in sigmas:
                    params = PositionSmoothingParams(lam=lam, p=p, kernel=kernel, sigma=sigma)
                    result = evaluate_split(calib_items, n_classes, logits_fn=params.fn())
                    score = result["overall_miou"]
                    if score > best_score:
                        best_score, best = score, params
    return best, best_score


@dataclass
class AttentionReweightParams:
    beta: float
    theta: float
    tau: float

    def fn(self, shared: dict):
        beta, theta, tau = self.beta, self.theta, self.tau

        def _apply(item: dict, r_flat: np.ndarray) -> np.ndarray:
            sim = item["patch_sim"]          # (n_patches, n_patches), pre-softmax
            v = item["v_patches"]            # (n_patches, embed_dim), pre-out_proj

            # NOTE (documented approximation): `sim`/`v` here are already
            # head-averaged/head-merged at cache time (see model.py), so this
            # redoes attention as one fused matrix rather than exactly
            # per-head. That's an acceptable ablation-level approximation —
            # the baseline/Method A numbers (which use the real per-head
            # forward pass at extraction time) are unaffected by it.
            boundary_query = (r_flat[:, None] > theta).astype(np.float32)
            closeness = 1.0 - np.abs(r_flat[:, None] - r_flat[None, :])
            bias = beta * boundary_query * closeness

            logits_attn = sim / tau + bias
            logits_attn = logits_attn - logits_attn.max(axis=-1, keepdims=True)
            exp = np.exp(logits_attn)
            attn = exp / exp.sum(axis=-1, keepdims=True)

            out = attn @ v                                    # (n_patches, embed_dim)
            out = out @ shared["out_proj_weight"].T + shared["out_proj_bias"]

            mean = out.mean(axis=-1, keepdims=True)
            var = out.var(axis=-1, keepdims=True)
            normed = (out - mean) / np.sqrt(var + shared["ln_post_eps"])
            normed = normed * shared["ln_post_weight"] + shared["ln_post_bias"]

            proj = normed @ shared["visual_proj"]
            proj = proj / np.linalg.norm(proj, axis=-1, keepdims=True)
            return proj @ shared["text_embeds"].T

        return _apply


def fit_attention_reweight(
    calib_items: list[dict],
    shared: dict,
    n_classes: int,
    beta_grid: list[float],
    theta_grid: list[float],
    tau_grid: list[float],
) -> tuple[AttentionReweightParams, float]:
    """Grid search (beta, theta, tau) maximizing overall mIoU on the
    calibration split. `calib_items` must include "patch_sim"/"v_patches"
    (see run.py's `_load_cached_items(..., load_attn_extras=True)`).
    """
    best = None
    best_score = -1.0
    for beta in beta_grid:
        for theta in theta_grid:
            for tau in tau_grid:
                params = AttentionReweightParams(beta=beta, theta=theta, tau=tau)
                result = evaluate_split(calib_items, n_classes, logits_fn=params.fn(shared))
                score = result["overall_miou"]
                if score > best_score:
                    best_score, best = score, params
    return best, best_score


def _bin_index(r_flat: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip(np.digitize(r_flat, edges[1:-1]), 0, n_bins - 1)


def estimate_radial_class_frequencies(
    calib_items: list[dict], n_classes: int, n_bins: int, epsilon: float = 1e-3,
):
    """For each r-bin k and class c, estimate:
        pred_freq[k, c] = P(argmax(logits) == c | patch in bin k)     (patch-level)
        true_freq[k, c] = P(GT == c | pixel in bin k, GT != IGNORE)   (pixel-level)
    over the whole calib_items set combined (pooled, not per-image — a single
    image rarely has enough pixels per (bin, class) to estimate this alone).
    Returns (delta, edges) where delta[k, c] = log(true+eps) - log(pred+eps).
    """
    edges = quartile_bin_edges(n_bins)
    pred_count = np.zeros((n_bins, n_classes))
    pred_total = np.zeros(n_bins)
    true_count = np.zeros((n_bins, n_classes))
    true_total = np.zeros(n_bins)

    for item in calib_items:
        grid_h, grid_w, gt = item["grid_h"], item["grid_w"], item["gt"]
        r_grid = radial_grid(grid_h, grid_w)
        r_flat = r_grid.reshape(-1)
        patch_bin = _bin_index(r_flat, edges, n_bins)

        pred = item["logits"].argmax(axis=-1)
        for b in range(n_bins):
            mask = patch_bin == b
            pred_total[b] += mask.sum()
            for c in range(n_classes):
                pred_count[b, c] += (pred[mask] == c).sum()

        r_pixel = nearest_upsample(r_grid, *gt.shape)
        pixel_bin = _bin_index(r_pixel, edges, n_bins)
        valid = gt != IGNORE_LABEL
        for b in range(n_bins):
            mask = valid & (pixel_bin == b)
            true_total[b] += mask.sum()
            for c in range(n_classes):
                true_count[b, c] += (gt[mask] == c).sum()

    pred_freq = pred_count / np.maximum(pred_total[:, None], 1)
    true_freq = true_count / np.maximum(true_total[:, None], 1)
    delta = np.log(true_freq + epsilon) - np.log(pred_freq + epsilon)
    return delta, edges


@dataclass
class RadialLogitAdjustmentParams:
    delta: np.ndarray   # (n_bins, n_classes), from estimate_radial_class_frequencies
    edges: np.ndarray   # bin edges used to fit `delta` — must reuse the same ones at eval time
    gamma: float

    def fn(self):
        delta, edges, gamma, n_bins = self.delta, self.edges, self.gamma, self.delta.shape[0]
        def _apply(item: dict, r_flat: np.ndarray) -> np.ndarray:
            bin_idx = _bin_index(r_flat, edges, n_bins)
            return item["logits"] + gamma * delta[bin_idx]
        return _apply


def fit_radial_logit_adjustment(
    calib_items: list[dict],
    n_classes: int,
    n_bins: int,
    gamma_grid: list[float],
) -> tuple[RadialLogitAdjustmentParams, float]:
    """Estimate delta once from calib_items (no gradient descent — a closed-
    form frequency count), then grid-search only gamma (correction strength)
    for calib-split mIoU. Returns (best_params, best_calib_miou).
    """
    delta, edges = estimate_radial_class_frequencies(calib_items, n_classes, n_bins)
    best = None
    best_score = -1.0
    for gamma in gamma_grid:
        params = RadialLogitAdjustmentParams(delta=delta, edges=edges, gamma=gamma)
        result = evaluate_split(calib_items, n_classes, logits_fn=params.fn())
        score = result["overall_miou"]
        if score > best_score:
            best_score, best = score, params
    return best, best_score
