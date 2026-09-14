"""CPU-only sanity tests for eval.py, plus one end-to-end smoke test of the
evaluate_split path run.py's diagnose/calibrate/evaluate stages all share —
entirely synthetic, no CLIP checkpoint or torch import required.
"""

import numpy as np
import pytest

from clearclip.eval import MIoUAccumulator, predict_labels, evaluate_split
from clearclip.calibration import PositionSmoothingParams


def test_miou_accumulator_perfect_prediction():
    acc = MIoUAccumulator(n_classes=2)
    gt = np.array([0, 1, 0, 1])
    pred = gt.copy()
    acc.update(pred, gt)
    assert acc.miou() == pytest.approx(1.0)


def test_miou_accumulator_all_wrong_two_classes():
    acc = MIoUAccumulator(n_classes=2)
    gt = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0])
    acc.update(pred, gt)
    assert acc.miou() == pytest.approx(0.0)


def test_predict_labels_upsamples_to_target_resolution():
    # 2 patches in a 1x2 grid, 2 classes; patch0 favors class1, patch1 favors class0
    logits = np.array([[0.1, 0.9], [0.8, 0.2]])
    pred = predict_labels(logits, grid_h=1, grid_w=2, out_h=4, out_w=4)
    assert pred.shape == (4, 4)
    assert np.all(pred[:, :2] == 1)  # left half came from patch0 (argmax class 1)
    assert np.all(pred[:, 2:] == 0)  # right half came from patch1 (argmax class 0)


def test_evaluate_split_end_to_end_with_synthetic_items():
    # One synthetic "image": 1x2 patch grid, patch1 is wrong at baseline but
    # fixable by Method A smoothing (same setup as test_calibration.py's
    # fit_position_smoothing test, exercised here through evaluate_split
    # directly with logits_fn=None vs a fitted PositionSmoothingParams).
    logits = np.array([[0.1, 0.9], [0.6, 0.4]])
    patch_attn = np.array([[1.0, 0.0], [1.0, 0.0]])
    gt = np.array([[1, 1]])
    item = {"logits": logits, "patch_attn": patch_attn, "grid_h": 1, "grid_w": 2, "gt": gt}

    baseline = evaluate_split([item], n_classes=2, logits_fn=None)
    calibrated = evaluate_split([item], n_classes=2, logits_fn=PositionSmoothingParams(lam=1.0, p=1).fn())

    assert calibrated["overall_miou"] > baseline["overall_miou"]
    assert len(baseline["per_bin_miou"]) == 4  # default n_bins
    assert len(baseline["per_image_correct"]) == 1
