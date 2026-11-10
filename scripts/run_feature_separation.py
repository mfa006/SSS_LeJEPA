#!/usr/bin/env python3
"""
Unsupervised Feature Separation & PCA Benchmark for Side-Scan Sonar (SSS).
Evaluates raw representation quality of pre-trained LeJEPA, ResNet-34, and ConvNeXt-Tiny
WITHOUT any linear probing or fine-tuning (zero-training).
"""

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import matplotlib.pyplot as plt
from typing import List

from sss_probe.config import SSSDataConfig, SUPPORTED_BACKBONES
from sss_probe.dataset import preprocess_sonar_image, load_sonar_mask
from sss_probe.backbones import BackboneExtractor
from sss_probe.metrics import (
    compute_pca_feature_map,
    evaluate_class_separation,
    compute_representation_health,
)


def run_benchmark(
    image_path: str,
    mask_path: str,
    models: List[str] = ["lejepa_vits16", "resnet34", "convnext_tiny"],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_png: str = "eval/results/feature_separation_comparison.png",
):
    gpu_name = torch.cuda.get_device_name(device) if "cuda" in str(device) and torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(" SSS RAW FEATURE SEPARATION BENCHMARK (ZERO-TRAINING, 100% LOCAL GPU)")
    print("=" * 70)
    print(f"Sample Tile:      {os.path.basename(image_path)}")
    print(f"Inference Device: {device} ({gpu_name})")
    print(f"Models:           {models}\n")

    cfg = SSSDataConfig()
    pil_img, img_tensor = preprocess_sonar_image(
        image_path,
        target_size=cfg.target_size,
        enable_contrast_stretch=cfg.enable_contrast_stretch,
    )
    img_tensor = img_tensor.unsqueeze(0).to(device)
    mask_tensor = load_sonar_mask(mask_path, target_size=cfg.target_size)

    pca_maps = {}
    metrics_summary = {}

    for name in models:
        print(f"--> Extracting features with '{name}'...")
        extractor = BackboneExtractor(name, device=device)
        
        with torch.no_grad():
            spatial_feats, global_feat, grid_sz = extractor(img_tensor)

        # 1. Top-3 PCA Heatmap
        pca_map = compute_pca_feature_map(spatial_feats.cpu(), out_size=cfg.target_size)
        pca_maps[name] = pca_map

        # 2. Inter-class separation
        sep = evaluate_class_separation(spatial_feats.cpu(), mask_tensor)

        # 3. Representation health
        health = compute_representation_health(spatial_feats.cpu())

        metrics_summary[name] = {**sep, **health, "grid_size": grid_sz}

        print(f"    Grid Size:          {grid_sz}")
        print(f"    Effective Rank:     {health['effective_rank']:.2f} / {health['max_rank']:.0f}")
        print(f"    Active Dimensions:  {health['active_dims']} ({health['active_dims_pct']:.1f}%)")
        print(f"    Mean Separation:    {sep['mean_separation_distance']:.4f} (Cosine Distance)")
        print(f"    Nearest-Centroid:   {sep['nearest_centroid_acc']:.2f}%\n")

    # Plot comparison
    n_cols = 2 + len(models)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 3.6))

    axes[0].imshow(pil_img, cmap="copper")
    axes[0].set_title(f"Input SSS Tile\n({os.path.basename(image_path)})", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(mask_tensor.numpy(), cmap="tab10", vmin=0, vmax=3)
    axes[1].set_title("Ground Truth Mask", fontsize=9)
    axes[1].axis("off")

    for i, name in enumerate(models):
        axes[2 + i].imshow(pca_maps[name])
        acc = metrics_summary[name]["nearest_centroid_acc"]
        axes[2 + i].set_title(f"{name.upper()} PCA\n(Sep Acc: {acc:.1f}%)", fontsize=9)
        axes[2 + i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Saved visual comparison figure to: {output_png}")
    plt.close()

    print("=" * 70)
    print(" BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate raw feature separation on SSS tile.")
    parser.add_argument("--image", type=str, default="test/images/prefix3_Recon_7_190312155901_xtf_CH34_waterfall_port_0_0_1_2_2.tiff")
    parser.add_argument("--mask", type=str, default="test/masks/prefix3_Recon_7_190312155901_xtf_CH34_waterfall_port_0_0_1_2_2.tiff")
    parser.add_argument("--models", nargs="+", default=["lejepa_vits16", "resnet34", "convnext_tiny"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="eval/results/feature_separation_comparison.png")
    args = parser.parse_args()

    run_benchmark(args.image, args.mask, args.models, args.device, args.output)
