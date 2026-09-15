"""Dataset loaders for training-free open-vocabulary segmentation benchmarks.

Only VOC20 is wired up end-to-end for now (Phase 0 / Week 1-4 scope from the
research plan). The other four benchmarks used by ClearCLIP's Table 2/3
(PASCAL Context59, COCO-Stuff, Cityscapes, ADE20K) are Phase 4 (Week 5-6)
expansion work — each needs its own class list (prompts.py) and label-id
remapping, so they're left as explicit stubs rather than silently wrong code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_LABEL = 255
VOC_N_FG_CLASSES = 20


def remap_voc_label(raw: np.ndarray, include_background: bool, n_fg_classes: int = VOC_N_FG_CLASSES) -> np.ndarray:
    """The raw VOC devkit palette is 0=background, 1..20=foreground classes,
    255=ignore/void — but our prediction logits are 0-indexed over just the
    *foreground* class list (prompts.VOC20_CLASSES), optionally with
    "background" appended LAST when include_background=True (see
    prompts.get_classes). Comparing raw labels against those predictions
    directly is off-by-one for every foreground class, and never excludes
    true background pixels from the (no-background) VOC20 protocol — this
    silently wrecked mIoU (observed 0.308 vs ClearCLIP's reported 80.9 on
    VOC20 alone) until caught here.

        include_background=False (VOC20): background(0) -> IGNORE,
                                            foreground(1..20) -> 0..19
        include_background=True  (VOC21): background(0) -> n_fg_classes (last
                                            column), foreground(1..20) -> 0..19
        either case:                       void(255) -> IGNORE (unconditionally)
    """
    background_target = n_fg_classes if include_background else IGNORE_LABEL
    out = np.where(raw == 0, background_target, raw - 1)
    out = np.where(raw == IGNORE_LABEL, IGNORE_LABEL, out)
    return out.astype(np.int64)


@dataclass
class Sample:
    image_id: str
    image_path: Path
    label_path: Path


class VOC20Dataset:
    """PASCAL VOC2012 segmentation val split, standard `SegmentationClass`
    palette-indexed masks (0 = background, 1..20 = classes, 255 = ignore) —
    remapped to our 0-indexed prediction columns via remap_voc_label.

    Directory layout expected under `root`:
        JPEGImages/<id>.jpg
        SegmentationClass/<id>.png
        ImageSets/Segmentation/val.txt
    """

    name = "voc20"

    def __init__(self, root: str, split: str = "val", include_background: bool = False):
        self.root = Path(root)
        self.include_background = include_background
        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"{split_file} not found. Download VOC2012 first "
                "(see colab notebook, cell 'Download VOC2012')."
            )
        ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.samples = [
            Sample(
                image_id=i,
                image_path=self.root / "JPEGImages" / f"{i}.jpg",
                label_path=self.root / "SegmentationClass" / f"{i}.png",
            )
            for i in ids
        ]
        self._by_id = {s.image_id: s for s in self.samples}

    def __len__(self):
        return len(self.samples)

    def load_label(self, image_id: str) -> np.ndarray:
        """Read + remap a label PNG by image id — the single place both
        __getitem__ and run.py's cache loader get GT from, so the remap is
        never applied in only one of the two paths.
        """
        raw = np.array(Image.open(self._by_id[image_id].label_path), dtype=np.int64)
        return remap_voc_label(raw, self.include_background)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return s.image_id, image, self.load_label(s.image_id)

    def split_calibration(self, calib_size: int, seed: int = 0):
        """Deterministic split: `calib_size` images held out for fitting the
        calibration curve, the rest reserved for the final reported eval.
        Returns (calib_ids, eval_ids).
        """
        ids = [s.image_id for s in self.samples]
        rng = random.Random(seed)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        if calib_size >= len(shuffled):
            raise ValueError(
                f"calib_split_size ({calib_size}) >= dataset size ({len(shuffled)})"
            )
        return set(shuffled[:calib_size]), set(shuffled[calib_size:])


_REGISTRY = {
    "voc20": VOC20Dataset,
    # TODO (Phase 4, Week 5-6): "context59", "coco_stuff", "cityscapes", "ade20k"
}


def load_dataset(name: str, root: str, split: str = "val", include_background: bool = False):
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"dataset '{name}' is not wired up yet — implemented: {list(_REGISTRY)}. "
            "See datasets.py TODO for what's needed to add it."
        )
    return _REGISTRY[name](root=root, split=split, include_background=include_background)
