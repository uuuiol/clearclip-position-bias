"""Class names and prompt templates for the supported datasets.

VOC20 is the standard 20-foreground-class split used by MaskCLIP / SCLIP /
ClearCLIP for the "without background" table. Set dataset.include_background
in config.yaml to add the 21st "background" class (the "with background" table).
"""

VOC20_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

BACKGROUND_CLASS = "background"

# Same 7-template ensemble ClearCLIP/SCLIP report using; averaging the
# resulting text embeddings before L2-normalizing gives a small but
# consistent boost over a single template.
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
    "a photo of the {}.",
    "a cropped photo of a {}.",
    "a bad photo of a {}.",
    "a close-up photo of a {}.",
]

DATASET_CLASSES = {
    "voc20": VOC20_CLASSES,
    # TODO (Week 5-6, Phase 4 dataset expansion):
    # "voc21", "context59", "coco_stuff", "cityscapes", "ade20k"
    # Each needs its own class-name list + label-id remapping in datasets.py.
}


def get_classes(dataset_name: str, include_background: bool) -> list[str]:
    classes = list(DATASET_CLASSES[dataset_name])
    if include_background:
        classes = classes + [BACKGROUND_CLASS]
    return classes
