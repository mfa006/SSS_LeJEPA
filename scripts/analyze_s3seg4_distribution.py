#!/usr/bin/env python3
"""Analyze S3SEG4 mask-pixel and image-level class distributions.

The recommended weights use only the selected training split's mask pixels:
    weight[c] = sqrt(max_class_pixel_count / class_pixel_count[c])
This is a moderate inverse-frequency heuristic, not a statistically optimal
weighting guarantee. Validate it against unweighted training and validation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sss_probe.config import ACOUSTIC_CLASSES


DEFAULT_DATA_ROOT = Path(
    "/media/rahul/Apacer_1TB/data/FromHayat/2ndRound/SSS/"
    "s3seg4_256_unique/data"
)
CLASS_NAMES = tuple(ACOUSTIC_CLASSES[index] for index in sorted(ACOUSTIC_CLASSES))
NUM_CLASSES = len(CLASS_NAMES)


def analyze_split(split_dir: Path, image_size: int) -> dict[str, Any]:
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    label_dir = split_dir / "class_labels"

    for directory in (image_dir, mask_dir, label_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")

    images = sorted(image_dir.glob("*.tiff"))
    if not images:
        raise FileNotFoundError(f"No TIFF images found in {image_dir}")

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_presence_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    invalid_pixels: dict[int, int] = {}

    for image_path in images:
        stem = image_path.stem
        mask_path = mask_dir / f"{stem}.tiff"
        label_path = label_dir / f"{stem}.csv"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask paired with {image_path.name}: {mask_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing class-label CSV paired with {image_path.name}: {label_path}")

        with Image.open(mask_path) as mask_image:
            mask = np.asarray(
                mask_image.convert("L").resize(
                    (image_size, image_size), Image.Resampling.NEAREST
                )
            )
        values, counts = np.unique(mask, return_counts=True)
        for value, count in zip(values, counts):
            class_index = int(value)
            if 0 <= class_index < NUM_CLASSES:
                pixel_counts[class_index] += int(count)
            else:
                invalid_pixels[class_index] = invalid_pixels.get(class_index, 0) + int(count)

        present_classes = set()
        with label_path.open(newline="", encoding="utf-8-sig") as label_file:
            reader = csv.DictReader(label_file)
            if not reader.fieldnames or not {"class_label", "presence"}.issubset(reader.fieldnames):
                raise ValueError(f"Expected class_label,presence columns in {label_path}")
            for row in reader:
                class_index = int(row["class_label"])
                if 0 <= class_index < NUM_CLASSES and float(row["presence"]) > 0:
                    present_classes.add(class_index)
        for class_index in present_classes:
            image_presence_counts[class_index] += 1

    image_count = len(images)
    pixel_total = int(pixel_counts.sum())
    if invalid_pixels:
        raise ValueError(f"Mask contains labels outside 0..{NUM_CLASSES - 1}: {invalid_pixels}")
    if pixel_total == 0:
        raise ValueError(f"No valid class pixels found in {mask_dir}")

    return {
        "image_count": image_count,
        "pixel_counts": pixel_counts.tolist(),
        "pixel_percentages": (100 * pixel_counts / pixel_total).tolist(),
        "image_presence_counts": image_presence_counts.tolist(),
        "image_presence_percentages": (100 * image_presence_counts / image_count).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-split", default="liear_prob")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/data_analysis"))
    args = parser.parse_args()

    if args.image_size <= 0:
        parser.error("--image-size must be positive")

    data_root = args.data_root.expanduser().resolve()
    split_names = (args.train_split, args.val_split, args.test_split)
    if len(set(split_names)) != len(split_names):
        parser.error("train, val, and test split names must be distinct")

    split_results = {
        split_name: analyze_split(data_root / split_name, args.image_size)
        for split_name in split_names
    }

    train_counts = np.asarray(split_results[args.train_split]["pixel_counts"], dtype=np.float64)
    if np.any(train_counts == 0):
        missing = [CLASS_NAMES[i] for i, count in enumerate(train_counts) if count == 0]
        raise ValueError(f"Cannot recommend weights; no training-mask pixels for: {missing}")
    recommended_weights = np.sqrt(train_counts.max() / train_counts)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "data_root": str(data_root),
        "mask_resize": {
            "size": [args.image_size, args.image_size],
            "interpolation": "nearest",
        },
        "training_split": args.train_split,
        "classes": {
            str(index): {
                "name": CLASS_NAMES[index],
                "recommended_weight": float(recommended_weights[index]),
            }
            for index in range(NUM_CLASSES)
        },
        "weight_rule": "sqrt(max training-mask pixel count / class training-mask pixel count)",
        "splits": split_results,
    }

    json_path = output_dir / "class_distribution.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    split_labels = {
        args.train_split: f"Train ({args.train_split})",
        args.val_split: f"Validation ({args.val_split})",
        args.test_split: f"Test ({args.test_split})",
    }
    x = np.arange(NUM_CLASSES)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for position, split_name in enumerate(split_names):
        offset = (position - (len(split_names) - 1) / 2) * width
        values = split_results[split_name]
        axes[0].bar(
            x + offset,
            values["pixel_percentages"],
            width,
            label=split_labels[split_name],
        )
        axes[1].bar(
            x + offset,
            values["image_presence_percentages"],
            width,
            label=split_labels[split_name],
        )

    for axis, title, ylabel in (
        (axes[0], "Mask Pixel Distribution", "Share of resized mask pixels (%)"),
        (axes[1], "Class Presence by Image", "Images with class present (%)"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, CLASS_NAMES)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    plot_path = output_dir / "class_distribution.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Analyzed data: {data_root}")
    print(f"Training split for weight recommendation: {args.train_split}")
    print("Class                         Pixels (%)   Images present (%)   Recommended weight")
    train_result = split_results[args.train_split]
    for index, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<28} "
            f"{train_result['pixel_percentages'][index]:>8.2f} "
            f"{train_result['image_presence_percentages'][index]:>18.2f} "
            f"{recommended_weights[index]:>21.3f}"
        )
    print("Validation/test percentages are included in the chart and JSON report.")
    print(f"Recommended YAML class_weights: {[round(float(w), 3) for w in recommended_weights]}")
    print(f"Chart: {plot_path}")
    print(f"Report: {json_path}")
    print("Weights are a square-root inverse-frequency heuristic; compare against unweighted training.")


if __name__ == "__main__":
    main()
