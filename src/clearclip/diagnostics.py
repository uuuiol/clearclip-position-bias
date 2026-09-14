"""Phase 1 — position-bias diagnostic.

Implements exactly the formulas from the research doc:

  r_i     = normalized distance of patch i from the image center, in [0,1]
  bins    = quartile bins of r (Q1 = center .. Q4 = boundary)
  Delta   = mIoU(Q1) - mIoU(Q4), with an image-level bootstrap CI
  c_i     = sum_j A_ij * (1 - r_j)   (attention-centrality of query patch i)
  rho     = Spearman correlation of c_i vs r_i  (and, separately, of GT-object
            centroid distance vs per-object pixel accuracy)

Everything here runs on cached logits/attention/labels — no GPU needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

from .datasets import IGNORE_LABEL


def radial_grid(grid_h: int, grid_w: int) -> np.ndarray:
    """Returns (grid_h, grid_w) array of r_i in [0, 1], distance from the
    patch-grid center normalized by the distance to the farthest grid cell.
    """
    ys, xs = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
    cy, cx = (grid_h - 1) / 2.0, (grid_w - 1) / 2.0
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    max_dist = dist.max()
    return dist / max_dist if max_dist > 0 else dist


def nearest_upsample(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor upsample of a (h, w) array of ints/floats to (out_h, out_w)."""
    img = Image.fromarray(grid.astype(np.float32), mode="F")
    img = img.resize((out_w, out_h), resample=Image.NEAREST)
    return np.array(img)


def quartile_bin_edges(n_bins: int = 4) -> np.ndarray:
    """Fixed edges over the *known* r range [0,1] (r's distribution is
    determined by grid geometry, not data, so quantile-of-data and
    quantile-of-range coincide up to the usual corner-vs-edge lattice effect;
    fixed edges keep bin membership identical across every image/dataset).
    """
    return np.linspace(0.0, 1.0, n_bins + 1)


@dataclass
class BinConfusion:
    """Per-bin, per-class intersection/union accumulators for a dataset-level
    (aggregate, not averaged-per-image) mIoU — matches how ClearCLIP itself
    reports mIoU.
    """

    n_bins: int
    n_classes: int
    intersection: np.ndarray = field(init=False)
    union: np.ndarray = field(init=False)

    def __post_init__(self):
        self.intersection = np.zeros((self.n_bins, self.n_classes), dtype=np.int64)
        self.union = np.zeros((self.n_bins, self.n_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray, bin_idx: np.ndarray):
        valid = gt != IGNORE_LABEL
        for b in range(self.n_bins):
            mask = valid & (bin_idx == b)
            if not mask.any():
                continue
            p, g = pred[mask], gt[mask]
            for c in range(self.n_classes):
                pc, gc = p == c, g == c
                self.intersection[b, c] += np.logical_and(pc, gc).sum()
                self.union[b, c] += np.logical_or(pc, gc).sum()

    def per_bin_miou(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = self.intersection / np.maximum(self.union, 1)
        present = self.union > 0
        out = np.full(self.n_bins, np.nan)
        for b in range(self.n_bins):
            if present[b].any():
                out[b] = iou[b, present[b]].mean()
        return out


def per_image_bin_accuracy(pred: np.ndarray, gt: np.ndarray, bin_idx: np.ndarray, n_bins: int):
    """Per-image (correct, valid) pixel counts per bin — the unit the
    bootstrap resamples over, since per-image mIoU is ill-defined when a
    class is absent from an image.
    """
    valid = gt != IGNORE_LABEL
    correct = np.zeros(n_bins, dtype=np.int64)
    total = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = valid & (bin_idx == b)
        total[b] = mask.sum()
        if total[b] > 0:
            correct[b] = (pred[mask] == gt[mask]).sum()
    return correct, total


def bootstrap_bias_gap(
    per_image_correct: list[np.ndarray],
    per_image_total: list[np.ndarray],
    n_resamples: int = 1000,
    seed: int = 0,
):
    """Delta = pixel-acc(bin 0, center) - pixel-acc(bin -1, boundary),
    bootstrapped over images. Returns (point_estimate, ci_low, ci_high).
    """
    correct = np.stack(per_image_correct)  # (n_images, n_bins)
    total = np.stack(per_image_total)
    n_images = correct.shape[0]

    def gap_from(idx):
        c = correct[idx].sum(axis=0)
        t = np.maximum(total[idx].sum(axis=0), 1)
        acc = c / t
        return acc[0] - acc[-1]

    point = gap_from(np.arange(n_images))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_images, size=n_images)
        boot[i] = gap_from(idx)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def attention_centrality(patch_attn: np.ndarray, r_flat: np.ndarray) -> np.ndarray:
    """c_i = sum_j A_ij * (1 - r_j) for every query patch i. `patch_attn` is
    (n_patches, n_patches), already head-averaged.
    """
    return patch_attn @ (1.0 - r_flat)


def centrality_vs_position_correlation(c_all: np.ndarray, r_all: np.ndarray):
    """Spearman rho + p-value between attention-centrality c_i and r_i,
    pooled across all patches of all diagnostic images.
    """
    rho, p = spearmanr(c_all, r_all)
    return float(rho), float(p)


def object_level_correlation(
    gt: np.ndarray, pred: np.ndarray, r_pixel: np.ndarray, n_classes: int
):
    """Connected-component proxy for 'objects': for each GT class, label
    connected regions, and for each region record (centroid r, pixel
    accuracy of that region). Returns arrays (centroid_r[], obj_acc[]) plus
    the Spearman correlation between them.
    """
    centroid_r, obj_acc = [], []
    for c in range(n_classes):
        mask = gt == c
        if not mask.any():
            continue
        labeled, n_comp = ndimage.label(mask)
        for comp_id in range(1, n_comp + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() < 20:  # drop tiny specks, not meaningful "objects"
                continue
            centroid_r.append(float(r_pixel[comp_mask].mean()))
            obj_acc.append(float((pred[comp_mask] == c).mean()))

    centroid_r = np.array(centroid_r)
    obj_acc = np.array(obj_acc)
    if len(centroid_r) < 3:
        return centroid_r, obj_acc, float("nan"), float("nan")
    rho, p = spearmanr(centroid_r, obj_acc)
    return centroid_r, obj_acc, float(rho), float(p)
