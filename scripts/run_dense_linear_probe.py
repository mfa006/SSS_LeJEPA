#!/usr/bin/env python3
"""
Dense Linear Probing on Side-Scan Sonar (SSS) Semantic Segmentation.
All configuration (data paths, hyperparameters, output paths) is read from
a single YAML file. Data is assumed to be already split into train/val/test
directories.

Saves:
  1. Trained model checkpoint (.pt) in output.checkpoint_dir
  2. Numerical benchmark metrics (.json and .csv) in output.results_dir
  3. Loss and mIoU training curves in output.results_dir
  4. Multi-tile test predictions on unseen test tiles in output.results_dir

Usage:
    python scripts/run_dense_linear_probe.py --config configs/linear_probe.yaml
"""

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import argparse
import yaml
from typing import Dict, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sss_probe.config import SSSDataConfig, LinearProbeConfig, SUPPORTED_BACKBONES
from sss_probe.dataset import load_presplit_dataset, SSSDataset
from sss_probe.backbones import BackboneExtractor
from sss_probe.probe import DenseLinearProbe
from sss_probe.trainer import DenseProbeTrainer
from sss_probe.metrics import compute_pca_feature_map


# ── YAML loader ──────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_data_config(cfg: dict) -> SSSDataConfig:
    d = cfg["data"]
    return SSSDataConfig(
        images_subdir           = d.get("images_subdir",      "images"),
        masks_subdir            = d.get("masks_subdir",        "masks"),
        labels_subdir           = d.get("labels_subdir",       "class_labels"),
        image_extension         = d.get("image_extension",     ".tiff"),
        target_size             = tuple(d.get("target_size",   [224, 224])),
        enable_contrast_stretch = d.get("enable_contrast_stretch", True),
        p_low                   = d.get("p_low",  0.01),
        p_high                  = d.get("p_high", 0.99),
    )


def build_probe_config(cfg: dict, device: str) -> LinearProbeConfig:
    t = cfg["training"]
    p = cfg["probe"]
    o = cfg["output"]
    return LinearProbeConfig(
        epochs                   = t.get("epochs",                   30),
        batch_size               = t.get("batch_size",               32),
        learning_rate            = t.get("learning_rate",            1e-3),
        weight_decay             = t.get("weight_decay",             1e-4),
        lr_scheduler             = t.get("lr_scheduler",             "cosine"),
        early_stopping_patience  = t.get("early_stopping_patience",  10),
        norm_type                = p.get("norm_type",                "batchnorm"),
        use_class_weights        = p.get("use_class_weights",        True),
        class_weights            = p.get("class_weights",            [1.0, 13.8, 1.5, 1.8]),
        device                   = device,
        cache_dir                = o.get("cache_dir",                "eval/cache"),
        checkpoint_dir           = o.get("checkpoint_dir",           "eval/checkpoints"),
        results_dir              = o.get("results_dir",              "eval/results"),
    )


# ── Feature caching ──────────────────────────────────────────────────────────

def get_or_extract_features(
    extractor: BackboneExtractor,
    dataset: SSSDataset,
    split_name: str,
    model_name: str,
    cache_dir: str = "eval/cache",
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (features, masks) from disk cache if available,
    otherwise extracts from the frozen backbone and saves to cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{model_name}_{split_name}_feats.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached {split_name} features from: {cache_file}")
        cached = torch.load(cache_file, map_location="cpu")
        return cached["features"].float(), cached["masks"].long()

    print(f"Extracting and caching {split_name} features ({len(dataset)} tiles)...")
    all_feats, all_masks = [], []

    for i in tqdm(range(len(dataset)), desc=f"Caching {split_name}"):
        sample = dataset[i]
        img  = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].to(torch.uint8)

        with torch.no_grad():
            spatial_feats, _, _ = extractor(img)

        all_feats.append(spatial_feats.cpu())
        all_masks.append(mask)

    cached_feats = torch.cat(all_feats, dim=0)   # [N, D, H_g, W_g]
    cached_masks = torch.stack(all_masks, dim=0)  # [N, 224, 224]

    torch.save({"features": cached_feats, "masks": cached_masks}, cache_file)
    print(f"Saved {split_name} cache to: {cache_file}")
    return cached_feats.float(), cached_masks.long()


# ── Benchmark summary ─────────────────────────────────────────────────────────

def update_benchmark_summary(
    results_dir: str,
    model_name: str,
    test_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    best_epoch: int,
    in_channels: int,
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    summary_json = os.path.join(results_dir, "benchmark_summary.json")
    summary_csv  = os.path.join(results_dir, "benchmark_summary.csv")

    record: Dict[str, Any] = {
        "model":                model_name,
        "feature_dim":          in_channels,
        "best_epoch":           best_epoch,
        "test_mIoU_pct":        round(test_metrics["mIoU"] * 100, 2),
        "test_mDice_pct":       round(test_metrics["mDice"] * 100, 2),
        "test_pixel_acc_pct":   round(test_metrics["pixel_accuracy"] * 100, 2),
        "test_iou_ripples_pct": round(test_metrics.get("iou_class_0", 0.0) * 100, 2),
        "test_iou_target_pct":  round(test_metrics.get("iou_class_1", 0.0) * 100, 2),
        "test_iou_sediment_pct":round(test_metrics.get("iou_class_2", 0.0) * 100, 2),
        "test_iou_shadow_pct":  round(test_metrics.get("iou_class_3", 0.0) * 100, 2),
        "val_mIoU_pct":         round(val_metrics["mIoU"] * 100, 2),
        "val_pixel_acc_pct":    round(val_metrics["pixel_accuracy"] * 100, 2),
    }

    summary_data = {}
    if os.path.exists(summary_json):
        try:
            with open(summary_json, "r") as f:
                summary_data = json.load(f)
        except Exception:
            summary_data = {}

    summary_data[model_name] = record
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Updated benchmark JSON summary: {summary_json}")

    fieldnames = [
        "model", "feature_dim", "best_epoch",
        "test_mIoU_pct", "test_mDice_pct", "test_pixel_acc_pct",
        "test_iou_ripples_pct", "test_iou_target_pct",
        "test_iou_sediment_pct", "test_iou_shadow_pct",
        "val_mIoU_pct", "val_pixel_acc_pct",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in sorted(summary_data.keys()):
            writer.writerow(summary_data[k])
    print(f"Updated benchmark CSV summary:  {summary_csv}")


# ── Qualitative plot ──────────────────────────────────────────────────────────

def generate_test_prediction_plot(
    extractor: BackboneExtractor,
    trainer: DenseProbeTrainer,
    test_ds: SSSDataset,
    model_name: str,
    output_png: str,
    n_samples: int = 3,
    device: str = "cuda",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3.8 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx in range(n_samples):
        sample   = test_ds[row_idx * 5 % len(test_ds)]
        test_img = sample["image"].unsqueeze(0)
        gt_mask  = sample["mask"].numpy()

        with torch.no_grad():
            test_feat, _, _ = extractor(test_img.to(device))
            pred_mask = trainer.predict_mask(test_feat, out_size=(224, 224))
            pca_map   = compute_pca_feature_map(test_feat.cpu(), out_size=(224, 224))

        gray_tile = (sample["image"][0].numpy() * 0.229 + 0.485).clip(0, 1)

        axes[row_idx, 0].imshow(gray_tile, cmap="copper")
        axes[row_idx, 0].set_title(f"Test Tile {row_idx + 1}\n({sample['filename'][:20]}...)", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=3)
        axes[row_idx, 1].set_title("Ground Truth Mask", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pca_map)
        axes[row_idx, 2].set_title(f"{model_name.upper()} PCA Map\n(Frozen Features)", fontsize=9)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(pred_mask, cmap="tab10", vmin=0, vmax=3)
        axes[row_idx, 3].set_title("1x1 Linear Probe Prediction\n(Trained on split data)", fontsize=9)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved test qualitative comparison to: {output_png}")


# ── Main experiment ───────────────────────────────────────────────────────────

def run_dense_probe_experiment(yaml_path: str) -> Dict[str, Any]:
    cfg        = load_yaml(yaml_path)
    model_name = cfg["model"]["name"]
    device_str = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device     = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"

    output_dir     = cfg["output"].get("results_dir",    "eval/results")
    checkpoint_dir = cfg["output"].get("checkpoint_dir", "eval/checkpoints")
    cache_dir      = cfg["output"].get("cache_dir",      "eval/cache")

    gpu_name = torch.cuda.get_device_name(device) if "cuda" in device and torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(" SSS DENSE LINEAR PROBE  —  YAML-driven")
    print("=" * 70)
    print(f"Config file:       {yaml_path}")
    print(f"Model Backbone:    {model_name}")
    print(f"Inference Device:  {device} ({gpu_name})\n")

    # ── 1. Build SSSDataConfig from YAML ─────────────────────────────────────
    data_cfg  = build_data_config(cfg)
    train_dir = cfg["data"]["train_dir"]
    val_dir   = cfg["data"]["val_dir"]
    test_dir  = cfg["data"]["test_dir"]

    # ── 2. Load pre-split datasets ────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_presplit_dataset(
        train_dir, val_dir, test_dir, config=data_cfg
    )

    # ── 3. Frozen backbone extractor ──────────────────────────────────────────
    extractor = BackboneExtractor(model_name, device=device)
    spec      = SUPPORTED_BACKBONES[model_name]

    # ── 4. Extract or load cached features ───────────────────────────────────
    train_feats, train_masks = get_or_extract_features(
        extractor, train_ds, "train", model_name, cache_dir, device
    )
    val_feats, val_masks = get_or_extract_features(
        extractor, val_ds, "val", model_name, cache_dir, device
    )
    test_feats, test_masks = get_or_extract_features(
        extractor, test_ds, "test", model_name, cache_dir, device
    )

    # ── 5. Build probe + trainer from YAML hyperparams ───────────────────────
    probe_cfg = build_probe_config(cfg, device)
    probe     = DenseLinearProbe(
        in_channels = spec.feature_dim,
        num_classes = cfg["probe"].get("num_classes", 4),
        norm_type   = probe_cfg.norm_type,
    )
    trainer = DenseProbeTrainer(probe, config=probe_cfg)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print(f"--> Training 1x1 Linear Layer for {probe_cfg.epochs} epochs ...")
    history = trainer.fit(train_feats, train_masks, val_feats, val_masks)

    # ── 7. Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_linear_probe.pt")
    trainer.save_checkpoint(
        ckpt_path,
        extra_info={
            "model_name":    model_name,
            "feature_dim":   spec.feature_dim,
            "train_samples": len(train_ds),
            "val_samples":   len(val_ds),
            "test_samples":  len(test_ds),
            "config_file":   yaml_path,
        },
    )

    # ── 8. Training curves ────────────────────────────────────────────────────
    curves_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    trainer.save_training_curves(history, curves_path)

    # ── 9. Evaluate ───────────────────────────────────────────────────────────
    print("\n--> Evaluating on Validation Set ...")
    val_metrics = trainer.evaluate(val_feats, val_masks)

    print("--> Evaluating on Test Set ...")
    test_metrics = trainer.evaluate(test_feats, test_masks)

    print("\n" + "=" * 65)
    print(f" DENSE LINEAR PROBE TEST SET RESULTS ({model_name.upper()})")
    print("=" * 65)
    print(f"• Test Mean IoU (mIoU):         {test_metrics['mIoU'] * 100:.2f}%")
    print(f"• Test Mean Dice Score:         {test_metrics['mDice'] * 100:.2f}%")
    print(f"• Test Overall Pixel Accuracy:  {test_metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"• Class 0 (Ripples) IoU:        {test_metrics.get('iou_class_0', 0.0) * 100:.2f}%")
    print(f"• Class 1 (Target Anomaly) IoU: {test_metrics.get('iou_class_1', 0.0) * 100:.2f}%")
    print(f"• Class 2 (Sediment) IoU:       {test_metrics.get('iou_class_2', 0.0) * 100:.2f}%")
    print(f"• Class 3 (Shadow) IoU:         {test_metrics.get('iou_class_3', 0.0) * 100:.2f}%")
    print("=" * 65)

    # ── 10. Save per-model JSON & cumulative benchmark ────────────────────────
    model_json_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(model_json_path, "w") as f:
        json.dump(
            {
                "model_name":       model_name,
                "config_file":      yaml_path,
                "best_epoch":       trainer.best_epoch,
                "test_metrics":     test_metrics,
                "val_metrics":      val_metrics,
                "training_history": history,
            },
            f,
            indent=2,
        )
    print(f"Saved full model metrics to: {model_json_path}")

    update_benchmark_summary(
        results_dir=output_dir,
        model_name=model_name,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        best_epoch=trainer.best_epoch,
        in_channels=spec.feature_dim,
    )

    # ── 11. Qualitative test plot ─────────────────────────────────────────────
    plot_path = os.path.join(output_dir, f"{model_name}_test_predictions.png")
    generate_test_prediction_plot(extractor, trainer, test_ds, model_name, plot_path, device=device)

    return {"test_metrics": test_metrics, "val_metrics": val_metrics, "history": history}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dense Linear Probe on SSS — all settings loaded from a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/linear_probe.yaml",
        help="Path to the YAML configuration file (default: configs/linear_probe.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    run_dense_probe_experiment(yaml_path=args.config)
