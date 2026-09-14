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


@dataclass
class Sample:
    image_id: str
    image_path: Path
    label_path: Path


class VOC20Dataset:
    """PASCAL VOC2012 segmentation val split, standard `SegmentationClass`
    palette-indexed masks (0 = background, 1..20 = classes, 255 = ignore).

    Directory layout expected under `root`:
        JPEGImages/<id>.jpg
        SegmentationClass/<id>.png
        ImageSets/Segmentation/val.txt
    """

    name = "voc20"

    def __init__(self, root: str, split: str = "val"):
        self.root = Path(root)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        label = Image.open(s.label_path)  # palette-indexed, values are class ids directly
        return s.image_id, image, np.array(label, dtype=np.int64)

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


def load_dataset(name: str, root: str, split: str = "val"):
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"dataset '{name}' is not wired up yet — implemented: {list(_REGISTRY)}. "
            "See datasets.py TODO for what's needed to add it."
        )
    return _REGISTRY[name](root=root, split=split)
