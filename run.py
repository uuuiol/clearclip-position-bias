#!/usr/bin/env python
"""Fixed run-command entrypoint: `python run.py --config config.yaml [--stage STAGE]`

Stages (each reads/writes disk artifacts so they can be run independently,
e.g. "extract" on Colab's GPU runtime and everything else later on CPU):

  extract    GPU. Runs ClearCLIP over the dataset split, caches per-image
             {logits, grid_h, grid_w, patch_attn} to config.cache.dir.
  diagnose   CPU. Phase 1: position-bias gap (bootstrap CI), attention
             centrality correlation, object-level correlation. This is the
             go/no-go checkpoint from the research plan.
  calibrate  CPU. Phase 2: fits Method A's (lambda, p) on the calibration
             split via grid search.
  evaluate   CPU. Applies the fitted calibration to the held-out eval split,
             reports baseline vs. calibrated mIoU (overall + per position bin).
  all        Runs diagnose -> calibrate -> evaluate in sequence (skips
             extract; run that separately once, typically on Colab).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from clearclip.model import ClearCLIPVisualEncoder, ClearCLIPConfig
from clearclip.datasets import load_dataset
from clearclip.prompts import get_classes, PROMPT_TEMPLATES
from clearclip import diagnostics as diag
from clearclip.calibration import fit_temperature
from clearclip.eval import evaluate_split


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def image_transform(image_size: int):
    import torchvision.transforms as T
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def stage_extract(cfg: dict):
    cache_dir = Path(cfg["cache"]["dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg["dataset"]
    dataset = load_dataset(ds_cfg["name"], ds_cfg["root"], ds_cfg["split"])
    classes = get_classes(ds_cfg["name"], ds_cfg["include_background"])

    bb_cfg = cfg["backbone"]
    cc_cfg = cfg["clearclip"]
    encoder = ClearCLIPVisualEncoder(ClearCLIPConfig(
        backbone=bb_cfg["name"], pretrained=bb_cfg["pretrained"],
        self_attn_mode=cc_cfg["self_attn_mode"], drop_residual=cc_cfg["drop_residual"],
        drop_ffn=cc_cfg["drop_ffn"], device=bb_cfg["device"],
    ))
    print(f"[extract] device={encoder.device}, {len(dataset)} images, {len(classes)} classes")

    text_embeds = encoder.encode_text(classes, PROMPT_TEMPLATES)
    transform = image_transform(ds_cfg["image_size"])

    for idx in range(len(dataset)):
        image_id, image, gt = dataset[idx]
        out_path = cache_dir / f"{image_id}.npz"
        if out_path.exists():
            continue
        pixel_values = transform(image).unsqueeze(0)
        logits, grid_h, grid_w, patch_attn = encoder.dense_logits(pixel_values, text_embeds)
        np.savez_compressed(
            out_path,
            logits=logits.cpu().numpy().astype(np.float16),
            patch_attn=patch_attn.cpu().numpy().astype(np.float16),
            grid_h=grid_h, grid_w=grid_w,
            gt_h=gt.shape[0], gt_w=gt.shape[1],
        )
        if idx % 50 == 0:
            print(f"[extract] {idx + 1}/{len(dataset)}")
    print(f"[extract] done -> {cache_dir}")


def _load_cached_items(cfg: dict, image_ids: set[str] | None = None, load_gt: bool = True):
    cache_dir = Path(cfg["cache"]["dir"])
    ds_cfg = cfg["dataset"]
    dataset = load_dataset(ds_cfg["name"], ds_cfg["root"], ds_cfg["split"]) if load_gt else None

    items = []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        image_id = npz_path.stem
        if image_ids is not None and image_id not in image_ids:
            continue
        data = np.load(npz_path)
        item = {
            "image_id": image_id,
            "logits": data["logits"].astype(np.float32),
            "grid_h": int(data["grid_h"]), "grid_w": int(data["grid_w"]),
            "patch_attn": data["patch_attn"].astype(np.float32),
        }
        if load_gt:
            from PIL import Image
            label_path = dataset.root / "SegmentationClass" / f"{image_id}.png"
            item["gt"] = np.array(Image.open(label_path), dtype=np.int64)
        items.append(item)
    return items


def stage_diagnose(cfg: dict):
    diag_cfg = cfg["diagnostics"]
    n_bins = diag_cfg["n_bins"]
    items = _load_cached_items(cfg)
    n_classes = len(get_classes(cfg["dataset"]["name"], cfg["dataset"]["include_background"]))

    result = evaluate_split(items, n_classes, n_bins=n_bins, temperature_fn=None)
    gap, lo, hi = diag.bootstrap_bias_gap(
        result["per_image_correct"], result["per_image_total"],
        n_resamples=diag_cfg["bootstrap_resamples"], seed=diag_cfg["seed"],
    )

    c_all, r_all = [], []
    obj_r_all, obj_acc_all = [], []
    for item in items:
        r_grid = diag.radial_grid(item["grid_h"], item["grid_w"])
        r_flat = r_grid.reshape(-1)
        c = diag.attention_centrality(item["patch_attn"], r_flat)
        c_all.append(c)
        r_all.append(r_flat)

        pred = item["logits"].argmax(axis=-1).reshape(item["grid_h"], item["grid_w"])
        gt = item["gt"]
        pred_px = diag.nearest_upsample(pred, *gt.shape)
        r_px = diag.nearest_upsample(r_grid, *gt.shape)
        # per-image (centroid_r, obj_acc) pairs, pooled across images below
        # for one dataset-wide Spearman correlation (per-image rho is noisy
        # with few objects per image).
        cr, ca, _, _ = diag.object_level_correlation(gt, pred_px, r_px, n_classes)
        obj_r_all.append(cr)
        obj_acc_all.append(ca)

    c_all = np.concatenate(c_all)
    r_all = np.concatenate(r_all)
    centrality_rho, centrality_p = diag.centrality_vs_position_correlation(c_all, r_all)

    obj_r_all = np.concatenate(obj_r_all) if obj_r_all else np.array([])
    obj_acc_all = np.concatenate(obj_acc_all) if obj_acc_all else np.array([])
    if len(obj_r_all) >= 3:
        from scipy.stats import spearmanr
        obj_rho, obj_p = spearmanr(obj_r_all, obj_acc_all)
    else:
        obj_rho, obj_p = float("nan"), float("nan")

    report = {
        "n_images": len(items),
        "overall_miou": result["overall_miou"],
        "per_bin_miou": result["per_bin_miou"],
        "bias_gap": {"point": gap, "ci_low": lo, "ci_high": hi},
        "attention_centrality_vs_position": {"spearman_rho": centrality_rho, "p_value": centrality_p},
        "object_level_vs_position": {"spearman_rho": float(obj_rho), "p_value": float(obj_p)},
    }

    go = lo > 0 or hi < 0  # CI excludes zero -> significant bias -> go
    report["go_no_go"] = "GO (bias confirmed, proceed to Phase 2)" if go else "NO-GO (bias not significant, revisit hypothesis)"

    out_path = Path("results") / f"{cfg['dataset']['name']}_diagnostic.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[diagnose] report written to {out_path}")


def stage_calibrate(cfg: dict):
    ds_cfg, cal_cfg = cfg["dataset"], cfg["calibration"]
    dataset = load_dataset(ds_cfg["name"], ds_cfg["root"], ds_cfg["split"])
    calib_ids, eval_ids = dataset.split_calibration(cal_cfg["calib_split_size"])

    calib_items = _load_cached_items(cfg, image_ids=calib_ids)
    n_classes = len(get_classes(ds_cfg["name"], ds_cfg["include_background"]))

    params, calib_miou = fit_temperature(
        calib_items, n_classes, cal_cfg["lambda_grid"], cal_cfg["p_grid"],
    )
    out = {"lambda": params.lam, "p": params.p, "calib_split_miou": calib_miou,
           "calib_split_size": len(calib_items)}
    out_path = Path("results") / f"{ds_cfg['name']}_calibration_params.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[calibrate] params written to {out_path}")


def stage_evaluate(cfg: dict):
    ds_cfg, cal_cfg = cfg["dataset"], cfg["calibration"]
    dataset = load_dataset(ds_cfg["name"], ds_cfg["root"], ds_cfg["split"])
    _, eval_ids = dataset.split_calibration(cal_cfg["calib_split_size"])
    n_classes = len(get_classes(ds_cfg["name"], ds_cfg["include_background"]))

    params_path = Path("results") / f"{ds_cfg['name']}_calibration_params.json"
    if not params_path.exists():
        print("[evaluate] no calibration params found — run --stage calibrate first.")
        return
    params_dict = json.loads(params_path.read_text())
    from clearclip.calibration import TemperatureParams
    params = TemperatureParams(lam=params_dict["lambda"], p=params_dict["p"])

    eval_items = _load_cached_items(cfg, image_ids=eval_ids)
    baseline = evaluate_split(eval_items, n_classes, temperature_fn=None)
    calibrated = evaluate_split(eval_items, n_classes, temperature_fn=params.fn())

    base_gap, base_lo, base_hi = diag.bootstrap_bias_gap(
        baseline["per_image_correct"], baseline["per_image_total"])
    cal_gap, cal_lo, cal_hi = diag.bootstrap_bias_gap(
        calibrated["per_image_correct"], calibrated["per_image_total"])

    report = {
        "n_eval_images": len(eval_items),
        "params": params_dict,
        "baseline": {"overall_miou": baseline["overall_miou"], "per_bin_miou": baseline["per_bin_miou"],
                     "bias_gap": base_gap, "bias_gap_ci": [base_lo, base_hi]},
        "calibrated": {"overall_miou": calibrated["overall_miou"], "per_bin_miou": calibrated["per_bin_miou"],
                       "bias_gap": cal_gap, "bias_gap_ci": [cal_lo, cal_hi]},
        "delta_overall_miou_pp": 100 * (calibrated["overall_miou"] - baseline["overall_miou"]),
        "delta_boundary_bin_miou_pp": 100 * (calibrated["per_bin_miou"][-1] - baseline["per_bin_miou"][-1]),
    }
    out_path = Path("results") / f"{ds_cfg['name']}_final_results.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[evaluate] results written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", default=None, help="overrides config.yaml's `stage` field")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        cfg["backbone"]["device"] = "cpu"
    stage = args.stage or cfg.get("stage", "diagnose")

    dispatch = {
        "extract": stage_extract,
        "diagnose": stage_diagnose,
        "calibrate": stage_calibrate,
        "evaluate": stage_evaluate,
    }
    if stage == "all":
        for s in ["diagnose", "calibrate", "evaluate"]:
            print(f"\n===== stage: {s} =====")
            dispatch[s](cfg)
    else:
        dispatch[stage](cfg)


if __name__ == "__main__":
    main()
