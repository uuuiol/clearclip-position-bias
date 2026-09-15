"""Regression test for the VOC label remap bug: raw VOC devkit masks are
0=background, 1..20=foreground, 255=ignore, but predictions are 0-indexed
over just the foreground class list. Comparing them unremapped was off-by-one
for every foreground class and never excluded background pixels — this was
the dominant cause of the 0.308-vs-~0.81 VOC20 mIoU gap.
"""

import numpy as np

from clearclip.datasets import remap_voc_label, IGNORE_LABEL


def test_remap_no_background_shifts_foreground_down_by_one():
    raw = np.array([0, 1, 2, 20, 255])
    out = remap_voc_label(raw, include_background=False)
    # background(0) -> ignore; aeroplane(1)->0; bicycle(2)->1; tvmonitor(20)->19; void stays ignore
    assert out.tolist() == [IGNORE_LABEL, 0, 1, 19, IGNORE_LABEL]


def test_remap_with_background_maps_it_to_last_column():
    raw = np.array([0, 1, 20, 255])
    out = remap_voc_label(raw, include_background=True, n_fg_classes=20)
    # background(0) -> column 20 (last, appended after the 20 fg classes)
    assert out.tolist() == [20, 0, 19, IGNORE_LABEL]


def test_remap_preserves_ignore_even_where_raw_minus_one_would_collide():
    # raw=255 must never fall through to the raw-1 branch (which would give
    # 254, a plausible-looking but wrong "valid" class index).
    raw = np.array([255, 255])
    out = remap_voc_label(raw, include_background=False)
    assert np.all(out == IGNORE_LABEL)


def test_remap_is_a_bijection_on_the_valid_label_set():
    raw = np.arange(0, 21)  # 0=background .. 20=tvmonitor, no void in this set
    out_no_bg = remap_voc_label(raw, include_background=False)
    valid = out_no_bg[out_no_bg != IGNORE_LABEL]
    assert sorted(valid.tolist()) == list(range(20))  # exactly 0..19, each once

    out_with_bg = remap_voc_label(raw, include_background=True, n_fg_classes=20)
    assert sorted(out_with_bg.tolist()) == list(range(21))  # exactly 0..20, each once
