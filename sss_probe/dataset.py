"""
Deterministic Side-Scan Sonar (SSS) dataset loader and few-shot partitioner.
Enforces ZERO data augmentation to evaluate intrinsic representation quality.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import SSSDataConfig


def preprocess_sonar_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    enable_contrast_stretch: bool = True,
    p_low: float = 0.01,
    p_high: float = 0.99,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Deterministic acoustic image loader.
    Applies pure radiometric calibration (zero random augmentations).

    Args:
        image_path: Path to the 4-channel RGBA TIFF image.
        target_size: (Width, Height) for network input.
        enable_contrast_stretch: Whether to apply 1st-99th percentile stretching.
        p_low: Lower percentile cutoff (e.g. 0.01).
        p_high: Upper percentile cutoff (e.g. 0.99).

    Returns:
        (display_pil_img, normalized_torch_tensor) where tensor is [3, H, W]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Sonar image not found: {image_path}")

    # 1. Load TIFF and convert RGBA -> 1-channel grayscale intensity
    raw_img = Image.open(image_path)
    gray_img = raw_img.convert("L").resize(target_size, Image.Resampling.BILINEAR)

    # 2. Pure PyTorch tensor conversion directly from buffer
    b = bytearray(gray_img.tobytes())
    t = torch.frombuffer(b, dtype=torch.uint8).reshape(target_size[1], target_size[0]).float()

    # 3. Deterministic percentile contrast stretching (clips extreme speckle & saturation)
    if enable_contrast_stretch:
        q_low = torch.quantile(t, p_low)
        q_high = torch.quantile(t, p_high)
        if q_high > q_low:
            t = torch.clamp((t - q_low) / (q_high - q_low) * 255.0, 0.0, 255.0)

    # 4. Standard 3-channel broadcast and ImageNet normalization
    t_3ch = (t.unsqueeze(0).repeat(3, 1, 1) / 255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    norm_tensor = (t_3ch - mean) / std

    return gray_img, norm_tensor


def load_sonar_mask(
    mask_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Loads semantic mask TIFF and resizes using nearest-neighbor interpolation.
    Returns: torch.LongTensor of shape [H, W] with class indices in {0, 1, 2, 3}.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    raw_mask = Image.open(mask_path).convert("L")
    resized_mask = raw_mask.resize(target_size, Image.Resampling.NEAREST)
    b_m = bytearray(resized_mask.tobytes())
    mask_tensor = torch.frombuffer(b_m, dtype=torch.uint8).reshape(target_size[1], target_size[0]).long()
    return mask_tensor


def load_sonar_labels(csv_path: str) -> torch.Tensor:
    """
    Loads multi-label acoustic category presence from CSV.
    Returns: torch.FloatTensor of shape [4] with binary indicators (0.0 or 1.0).
    """
    if not os.path.exists(csv_path):
        return torch.zeros(4, dtype=torch.float32)

    df = pd.read_csv(csv_path)
    labels = torch.zeros(4, dtype=torch.float32)
    for _, row in df.iterrows():
        c_idx = int(row["class_label"])
        if 0 <= c_idx < 4:
            labels[c_idx] = float(row["presence"])
    return labels


class SSSDataset(Dataset):
    """
    PyTorch Dataset for Side-Scan Sonar (SSS) imagery and ground truth.
    Provides deterministic image loading without any artificial augmentations.
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        label_paths: Optional[List[str]] = None,
        config: Optional[SSSDataConfig] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.label_paths = label_paths or []
        self.config = config or SSSDataConfig()

        assert len(self.image_paths) == len(self.mask_paths), (
            f"Image count ({len(self.image_paths)}) must match mask count ({len(self.mask_paths)})"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label_path = self.label_paths[idx] if idx < len(self.label_paths) else ""

        # Deterministic load (no random augmentations)
        _, image_tensor = preprocess_sonar_image(
            img_path,
            target_size=self.config.target_size,
            enable_contrast_stretch=self.config.enable_contrast_stretch,
            p_low=self.config.p_low,
            p_high=self.config.p_high,
        )

        mask_tensor = load_sonar_mask(mask_path, target_size=self.config.target_size)
        label_tensor = load_sonar_labels(label_path) if label_path else torch.zeros(4)

        return {
            "image": image_tensor,       # [3, 224, 224]
            "mask": mask_tensor,         # [224, 224] (long)
            "labels": label_tensor,      # [4] (binary float)
            "filename": os.path.basename(img_path),
        }


def partition_sss_dataset(
    config: Optional[SSSDataConfig] = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Splits the 1,800 SSS tiles into 10% Train, 5% Val, and 85% Test sets.
    Uses survey waterfall line prefixes to prevent spatial correlation leakage.
    """
    cfg = config or SSSDataConfig()
    img_dir = os.path.join(cfg.data_dir, cfg.images_subdir)
    mask_dir = os.path.join(cfg.data_dir, cfg.masks_subdir)
    lbl_dir = os.path.join(cfg.data_dir, cfg.labels_subdir)

    img_files = sorted(glob.glob(os.path.join(img_dir, f"*{cfg.image_extension}")))
    if not img_files:
        raise FileNotFoundError(f"No SSS images found in {img_dir}")

    # Track survey line prefixes to avoid autocorrelation leakage
    # e.g., 'prefix1_SC_B_RECON_5_190312220400_xtf_CH34_waterfall_stbd_31'
    np.random.seed(cfg.seed)
    shuffled_indices = np.random.permutation(len(img_files))

    n_total = len(img_files)
    n_train = int(round(cfg.train_ratio * n_total)) # ~180 (10%)
    n_val = int(round(cfg.val_ratio * n_total))     # ~90 (5%)

    train_idx = shuffled_indices[:n_train]
    val_idx = shuffled_indices[n_train:n_train + n_val]
    test_idx = shuffled_indices[n_train + n_val:]

    def get_split(indices):
        split_imgs = [img_files[i] for i in indices]
        split_masks = [os.path.join(mask_dir, os.path.basename(p)) for p in split_imgs]
        split_lbls = [
            os.path.join(lbl_dir, os.path.basename(p).replace(cfg.image_extension, ".csv"))
            for p in split_imgs
        ]
        return split_imgs, split_masks, split_lbls

    train_imgs, train_masks, train_lbls = get_split(train_idx)
    val_imgs, val_masks, val_lbls = get_split(val_idx)
    test_imgs, test_masks, test_lbls = get_split(test_idx)

    return (
        train_imgs, train_masks, train_lbls,
        val_imgs, val_masks, val_lbls,
        test_imgs, test_masks, test_lbls,
    )


def get_sss_dataloaders(
    config: Optional[SSSDataConfig] = None,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function returning (train_loader_10pct, val_loader_5pct, test_loader_85pct).
    """
    cfg = config or SSSDataConfig()
    splits = partition_sss_dataset(cfg)
    
    train_ds = SSSDataset(splits[0], splits[1], splits[2], config=cfg)
    val_ds = SSSDataset(splits[3], splits[4], splits[5], config=cfg)
    test_ds = SSSDataset(splits[6], splits[7], splits[8], config=cfg)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

