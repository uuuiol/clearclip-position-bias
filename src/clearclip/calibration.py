"""Phase 2 — Method A: position-conditioned temperature scaling.

    s'_{i,c} = s_{i,c} / T(r_i),      T(r_i) = 1 + lambda * r_i^p

lambda, p are fit by grid search on a small held-out CALIBRATION split
(disjoint from the split used for final reported eval — see
datasets.split_calibration). No gradient descent anywhere in this file.

Method B (self-self attention reweighting) is Week 4-5 ablation work and is
left as a TODO here; Method A is the one wired into run.py's "calibrate"
and "evaluate" stages first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eval import evaluate_split


@dataclass
class TemperatureParams:
    lam: float
    p: float

    def fn(self):
        lam, p = self.lam, self.p
        def _apply(logits: np.ndarray, r_flat: np.ndarray) -> np.ndarray:
            T = 1.0 + lam * np.power(r_flat, p)
            return logits / T[:, None]
        return _apply


def fit_temperature(
    calib_items: list[dict],
    n_classes: int,
    lambda_grid: list[float],
    p_grid: list[float],
) -> tuple[TemperatureParams, float]:
    """Grid search (lambda, p) maximizing overall mIoU on the calibration
    split. Returns (best_params, best_calib_miou).
    """
    best = None
    best_score = -1.0
    for lam in lambda_grid:
        for p in p_grid:
            params = TemperatureParams(lam=lam, p=p)
            result = evaluate_split(calib_items, n_classes, temperature_fn=params.fn())
            score = result["overall_miou"]
            if score > best_score:
                best_score, best = score, params
    return best, best_score


# TODO (Week 4-5, Method B ablation):
# def fit_attention_reweight(calib_items, beta_grid, theta_grid, tau_grid): ...
#   A'_ij = softmax_j(A_ij / tau + beta * 1[r_i > theta] * (1 - |r_i - r_j|))
#   Needs cached patch_attn (already saved by run.py's "extract" stage) rather
#   than logits — re-derive dense_logits from the reweighted attention output
#   instead of scaling the final cosine-similarity logits directly.
