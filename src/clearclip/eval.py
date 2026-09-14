"""mIoU evaluation matching the ClearCLIP protocol (dataset-level aggregate
intersection/union, not averaged per image), plus the position-stratified
breakdown used by the diagnostic/calibration stages.
"""

from __future__ import annotations

import numpy as np

from .datasets import IGNORE_LABEL
from .diagnostics import radial_grid, nearest_upsample, quartile_bin_edges, BinConfusion


def predict_labels(logits: np.ndarray, grid_h: int, grid_w: int, out_h: int, out_w: int) -> np.ndarray:
    """argmax over classes at patch resolution, then nearest-upsample to the
    original image resolution for pixel-level comparison against GT.
    """
    patch_labels = logits.argmax(axis=-1).reshape(grid_h, grid_w)
    return nearest_upsample(patch_labels, out_h, out_w).astype(np.int64)


class MIoUAccumulator:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.intersection = np.zeros(n_classes, dtype=np.int64)
        self.union = np.zeros(n_classes, dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        valid = gt != IGNORE_LABEL
        p, g = pred[valid], gt[valid]
        for c in range(self.n_classes):
            pc, gc = p == c, g == c
            self.intersection[c] += np.logical_and(pc, gc).sum()
            self.union[c] += np.logical_or(pc, gc).sum()

    def miou(self) -> float:
        present = self.union > 0
        if not present.any():
            return float("nan")
        iou = self.intersection[present] / self.union[present]
        return float(iou.mean())


def evaluate_split(
    items: list[dict],
    n_classes: int,
    n_bins: int = 4,
    temperature_fn=None,
):
    """items: list of {"logits", "grid_h", "grid_w", "gt"} for cached images.
    temperature_fn(logits, r_flat) -> calibrated logits; pass None for the
    uncalibrated ClearCLIP baseline.

    Returns dict with overall mIoU, per-bin mIoU, and the per-image bin
    pixel-accuracy arrays needed by diagnostics.bootstrap_bias_gap.
    """
    overall = MIoUAccumulator(n_classes)
    bin_conf = BinConfusion(n_bins=n_bins, n_classes=n_classes)
    edges = quartile_bin_edges(n_bins)

    per_image_correct, per_image_total = [], []

    for item in items:
        logits, grid_h, grid_w, gt = item["logits"], item["grid_h"], item["grid_w"], item["gt"]
        r_grid = radial_grid(grid_h, grid_w)
        r_flat = r_grid.reshape(-1)

        cal_logits = temperature_fn(logits, r_flat) if temperature_fn else logits

        out_h, out_w = gt.shape
        pred = predict_labels(cal_logits, grid_h, grid_w, out_h, out_w)
        r_pixel = nearest_upsample(r_grid, out_h, out_w)
        bin_idx = np.clip(np.digitize(r_pixel, edges[1:-1]), 0, n_bins - 1)

        overall.update(pred, gt)
        bin_conf.update(pred, gt, bin_idx)

        from .diagnostics import per_image_bin_accuracy
        c, t = per_image_bin_accuracy(pred, gt, bin_idx, n_bins)
        per_image_correct.append(c)
        per_image_total.append(t)

    return {
        "overall_miou": overall.miou(),
        "per_bin_miou": bin_conf.per_bin_miou().tolist(),
        "per_image_correct": per_image_correct,
        "per_image_total": per_image_total,
    }
