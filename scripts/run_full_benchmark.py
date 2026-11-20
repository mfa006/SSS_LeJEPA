#!/usr/bin/env python3
"""
Full Automated Benchmark Runner for SSS Linear Probing.
Evaluates LeJEPA ViT-S/16, ResNet-34, and ConvNeXt-Tiny on the exact same 10% few-shot split.
Saves:
  - Checkpoints (.pt) for each model in eval/checkpoints/
  - Unified summary leaderboard (.json and .csv) in eval/results/
  - Per-model loss and validation curves (.png) in eval/results/
  - Visual test predictions on unseen test tiles (.png) in eval/results/
"""

import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
import torch

from run_dense_linear_probe import run_dense_probe_experiment


def run_full_benchmark(
    models=("lejepa_vits16", "resnet34", "convnext_tiny"),
    epochs: int = 30,
    lr: float = 1e-3,
    norm_type: str = "batchnorm",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "eval/results",
    checkpoint_dir: str = "eval/checkpoints",
    cache_dir: str = "eval/cache",
):
    print("=" * 75)
    print(" SSS VISION FOUNDATION MODEL BENCHMARK (10% FEW-SHOT DENSE LINEAR PROBE)")
    print("=" * 75)
    print(f"Models to evaluate: {list(models)}")
    print(f"Device:             {device}")
    print(f"Epochs per model:   {epochs}")
    print(f"Results Directory:  {output_dir}")
    print(f"Checkpoints Dir:    {checkpoint_dir}\n")

    for model_name in models:
        print("\n" + "#" * 75)
        print(f" RUNNING BENCHMARK FOR: {model_name.upper()}")
        print("#" * 75)
        run_dense_probe_experiment(
            model_name=model_name,
            epochs=epochs,
            lr=lr,
            norm_type=norm_type,
            device=device,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            cache_dir=cache_dir,
        )

    # Print summary leaderboard
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("\n" + "=" * 75)
        print(" FINAL BENCHMARK LEADERBOARD (UNSEEN 85% TEST SET)")
        print("=" * 75)
        print(df[["model", "feature_dim", "test_mIoU_pct", "test_mDice_pct", "test_pixel_acc_pct", "test_macro_precision_pct", "test_macro_recall_pct", "test_iou_maerl_pct"]].to_string(index=False))
        print("=" * 75)
        print(f"Full CSV results saved at: {csv_path}")
        print(f"Trained models saved at:   {checkpoint_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete benchmark across all supported backbones.")
    parser.add_argument("--models", nargs="+", default=["lejepa_vits16", "resnet34", "convnext_tiny"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--norm_type", type=str, default="batchnorm", choices=["batchnorm", "layernorm", "l2", "none"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="eval/results")
    parser.add_argument("--checkpoint_dir", type=str, default="eval/checkpoints")
    parser.add_argument("--cache_dir", type=str, default="eval/cache")
    args = parser.parse_args()

    run_full_benchmark(
        models=args.models,
        epochs=args.epochs,
        lr=args.lr,
        norm_type=args.norm_type,
        device=args.device,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
    )
