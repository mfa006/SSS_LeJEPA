#!/usr/bin/env python3
"""Split S3SEG4 tiles into train/val/test and make a linear-probe subset.

Default output layout:
  data/train/{images,masks,class labels}/
  data/val/{images,masks,class labels}/
  data/test/{images,masks,class labels}/
  data/liear_prob/{images,masks,class labels}/

The split is deterministic and uses the same random selection for all three
modalities.  The probe subset is copied only from the training split.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path


SPLIT_RATIOS = {"train": 0.85, "val": 0.10, "test": 0.05}
MODALITY_DIRS = ("images", "masks", "class_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split S3SEG4 image/mask/label data")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/media/rahul/Apacer_1TB/data/FromHayat/2ndRound/SSS/s3seg4_256_unique"),
        help="Source dataset root containing images, masks, and class labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output root (default: <source>/data)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--probe-ratio",
        type=float,
        default=0.10,
        help="Fraction of the training split copied to liear_prob",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output root before splitting",
    )
    return parser.parse_args()


def stem_set(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing source directory: {directory}")
    return {path.stem for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix}


def copy_item(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else source / "data")

    if not 0 < args.probe_ratio < 1:
        raise ValueError("--probe-ratio must be between 0 and 1")

    image_dir = source / "images"
    mask_dir = source / "masks"
    label_dir = source / "class labels"

    image_stems = stem_set(image_dir, ".tiff")
    mask_stems = stem_set(mask_dir, ".tiff")
    label_stems = stem_set(label_dir, ".csv")

    complete_stems = sorted(image_stems & mask_stems & label_stems)
    missing_masks = len(image_stems - mask_stems)
    missing_labels = len(image_stems - label_stems)

    if not complete_stems:
        raise RuntimeError("No complete image/mask/label triplets were found")
    if missing_masks or missing_labels:
        print(f"Warning: ignoring {missing_masks} unmatched masks and {missing_labels} unmatched labels")

    rng = random.Random(args.seed)
    rng.shuffle(complete_stems)

    total = len(complete_stems)
    train_count = round(total * SPLIT_RATIOS["train"])
    val_count = round(total * SPLIT_RATIOS["val"])
    test_count = total - train_count - val_count

    split_stems = {
        "train": complete_stems[:train_count],
        "val": complete_stems[train_count:train_count + val_count],
        "test": complete_stems[train_count + val_count:],
    }

    probe_count = round(len(split_stems["train"]) * args.probe_ratio)
    probe_stems = split_stems["train"][:probe_count]

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Use --overwrite to replace it."
        )
    if args.overwrite and output.exists():
        shutil.rmtree(output)

    split_members: dict[str, list[str]] = {}
    for split_name, stems in split_stems.items():
        split_members[split_name] = stems
        for stem in stems:
            copy_item(image_dir / f"{stem}.tiff", output / split_name / "images" / f"{stem}.tiff")
            copy_item(mask_dir / f"{stem}.tiff", output / split_name / "masks" / f"{stem}.tiff")
            copy_item(label_dir / f"{stem}.csv", output / split_name / "class_labels" / f"{stem}.csv")

    split_members["liear_prob"] = probe_stems
    for stem in probe_stems:
        copy_item(image_dir / f"{stem}.tiff", output / "liear_prob" / "images" / f"{stem}.tiff")
        copy_item(mask_dir / f"{stem}.tiff", output / "liear_prob" / "masks" / f"{stem}.tiff")
        copy_item(label_dir / f"{stem}.csv", output / "liear_prob" / "class_labels" / f"{stem}.csv")

    manifest_path = output / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.writer(manifest)
        writer.writerow(["stem", "split"])
        for split_name in ("train", "val", "test", "liear_prob"):
            for stem in split_members[split_name]:
                writer.writerow([stem, split_name])

    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Complete triplets: {total}")
    print(f"Train: {len(split_stems['train'])} (85%)")
    print(f"Val: {len(split_stems['val'])} (10%)")
    print(f"Test: {len(split_stems['test'])} (5%)")
    print(f"Liear probe subset: {len(probe_stems)} (10% of train)")


if __name__ == "__main__":
    main()
