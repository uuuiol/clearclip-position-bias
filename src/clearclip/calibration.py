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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
