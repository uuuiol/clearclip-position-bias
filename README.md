# Position bias in training-free CLIP dense segmentation

Baseline: ClearCLIP (ECCV 2024). Hypothesis: even after ClearCLIP removes the
residual-connection noise, patch-level self-self attention still carries a
center bias, so boundary objects segment worse than central ones. See the
Notion doc for full background/references.

## Running without a local GPU

Everything downstream of the one-time CLIP feature extraction is CPU-only.
Use `colab_run.ipynb` on Google Colab's free GPU tier for the `extract` stage,
then run `diagnose` / `calibrate` / `evaluate` there too (or locally once a
real Python env with the deps in `pyproject.toml` is available) against the
cached `.npz` files.

```
python run.py --config config.yaml --stage extract     # GPU, one-time
python run.py --config config.yaml --stage diagnose     # CPU, go/no-go checkpoint
python run.py --config config.yaml --stage calibrate    # CPU
python run.py --config config.yaml --stage evaluate     # CPU
```

## Status

- [x] VOC20 dataset loader, ClearCLIP model surgery (drop residual, q-q
      self-attention, drop FFN), Phase 1 diagnostic formulas, Phase 2 Method A
      (position-conditioned temperature scaling).
- [ ] Phase 2 Method B (attention reweighting) — stubbed in `calibration.py`.
- [ ] Phase 4 dataset expansion — PASCAL Context59 / COCO-Stuff / Cityscapes /
      ADE20K — stubbed in `datasets.py` / `prompts.py`.
- [ ] Sliding-window / multi-scale inference (current `encode_image_dense` is
      single-crop; ClearCLIP's reported numbers use multi-scale test-time
      augmentation, so expect the reproduced baseline mIoU to run a bit below
      the paper's until this is added).
