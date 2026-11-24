"""
Evaluation metrics for Side-Scan Sonar (SSS) representations.
Includes unsupervised PCA mapping, class separation, and segmentation mIoU.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .config import ACOUSTIC_CLASSES, NUM_ACOUSTIC_CLASSES


def compute_pca_feature_map(
    spatial_feats: torch.Tensor,
    out_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Projects dense spatial token embeddings to top-3 Principal Components (RGB).
    Uses pure PyTorch operations with 1st-99th percentile contrast normalization.

    Args:
        spatial_feats: Tensor of shape [1, D, H_g, W_g] or [D, H_g, W_g]
        out_size: (Height, Width) for output RGB heatmap

    Returns:
        RGB image as np.ndarray of shape [H, W, 3] with values in [0.0, 1.0]
    """
    if spatial_feats.ndim == 4:
        feats = spatial_feats[0] # [D, H_g, W_g]
    else:
        feats = spatial_feats

    D, H_g, W_g = feats.shape
    tokens = feats.reshape(D, H_g * W_g).T.float() # [N, D] where N = H_g * W_g

    # Center tokens
    centered = tokens - tokens.mean(dim=0, keepdim=True)

    # Top-3 PCA via SVD
    _, _, V = torch.pca_lowrank(centered, q=3, center=False)
    proj = torch.matmul(centered, V[:, :3]) # [N, 3]

    # Robust percentile normalization (1st to 99th percentile per channel)
    p1 = torch.quantile(proj, 0.01, dim=0, keepdim=True)
    p99 = torch.quantile(proj, 0.99, dim=0, keepdim=True)
    norm_proj = torch.clamp((proj - p1) / (p99 - p1 + 1e-8), 0.0, 1.0) # [N, 3]

    # Reshape back to spatial grid: [1, 3, H_g, W_g]
    grid = norm_proj.reshape(H_g, W_g, 3).permute(2, 0, 1).unsqueeze(0)

    # Bilinear upsample to target resolution
    upsampled = F.interpolate(grid, size=out_size, mode="bilinear", align_corners=False)
    pca_img = upsampled[0].permute(1, 2, 0).cpu().numpy()
    return pca_img


def evaluate_class_separation(
    spatial_feats: torch.Tensor,
    mask_224: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluates how cleanly frozen feature tokens separate ground-truth acoustic classes.
    Downsamples ground-truth mask to match the feature grid and measures
    pairwise cosine distance and nearest-centroid accuracy.

    Returns:
        dict containing 'mean_separation_distance', 'nearest_centroid_acc', etc.
    """
    if spatial_feats.ndim == 4:
        feats = spatial_feats[0] # [D, H_g, W_g]
    else:
        feats = spatial_feats

    D, H_g, W_g = feats.shape
    tokens = feats.reshape(D, H_g * W_g).T.float() # [N, D]
    norm_tokens = F.normalize(tokens, p=2, dim=-1)

    # Downsample ground-truth mask to feature grid via nearest neighbor
    mask_t = mask_224.float().unsqueeze(0).unsqueeze(0)
    down_mask = F.interpolate(mask_t, size=(H_g, W_g), mode="nearest").squeeze().long()
    mask_grid = down_mask.flatten().cpu().numpy()

    unique_classes = sorted(list(set(mask_grid)))
    if len(unique_classes) < 2:
        return {"mean_separation_distance": 0.0, "nearest_centroid_acc": 100.0}

    # Compute class centroids
    centroids = {}
    for c in unique_classes:
        mask_c = (mask_grid == c)
        c_tokens = norm_tokens[mask_c]
        centroids[c] = F.normalize(c_tokens.mean(dim=0, keepdim=True), p=2, dim=-1)

    # Pairwise distances
    distances = []
    for i, c1 in enumerate(unique_classes):
        for j, c2 in enumerate(unique_classes):
            if j > i:
                sim = torch.dot(centroids[c1].squeeze(), centroids[c2].squeeze()).item()
                distances.append(1.0 - sim)

    # Nearest-centroid accuracy
    all_c = torch.cat([centroids[c] for c in unique_classes], dim=0) # [C, D]
    logits = torch.matmul(norm_tokens, all_c.T) # [N, C]
    preds = np.array([unique_classes[idx] for idx in logits.argmax(dim=-1).cpu().numpy()])
    acc = float(100.0 * np.mean(preds == mask_grid))

    return {
        "mean_separation_distance": float(np.mean(distances)) if distances else 0.0,
        "nearest_centroid_acc": acc,
        "classes_present": unique_classes,
    }


def compute_segmentation_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    num_classes: int = NUM_ACOUSTIC_CLASSES,
) -> Dict[str, float]:
    """
    Computes Intersection-over-Union (IoU), Mean IoU, Dice Score, and Pixel Accuracy.

    Args:
        pred_masks: [N, H, W] integer array of predicted class indices
        gt_masks:   [N, H, W] integer array of ground truth class indices

    Returns:
        Dict mapping metric names to values
    """
    pred_flat = pred_masks.reshape(-1).astype(np.int64, copy=False)
    gt_flat = gt_masks.reshape(-1).astype(np.int64, copy=False)
    valid = (
        (pred_flat >= 0)
        & (pred_flat < num_classes)
        & (gt_flat >= 0)
        & (gt_flat < num_classes)
    )
    encoded = gt_flat[valid] * num_classes + pred_flat[valid]
    confusion = np.bincount(encoded, minlength=num_classes * num_classes)
    confusion = confusion.reshape(num_classes, num_classes)
    return compute_segmentation_metrics_from_confusion(confusion)


def compute_segmentation_metrics_from_confusion(
    confusion: np.ndarray,
) -> Dict[str, float]:
    """Compute segmentation metrics from a matrix with true rows and predicted columns."""
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")

    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    unions = true_support + predicted_support - true_positive
    dice_denominators = true_support + predicted_support

    ious = np.divide(
        true_positive, unions,
        out=np.full(unions.shape, np.nan, dtype=np.float64),
        where=unions != 0,
    )
    dices = np.divide(
        2 * true_positive, dice_denominators,
        out=np.full(dice_denominators.shape, np.nan, dtype=np.float64),
        where=dice_denominators != 0,
    )
    precisions = np.divide(
        true_positive, predicted_support,
        out=np.full(predicted_support.shape, np.nan, dtype=np.float64),
        where=predicted_support != 0,
    )
    recalls = np.divide(
        true_positive, true_support,
        out=np.full(true_support.shape, np.nan, dtype=np.float64),
        where=true_support != 0,
    )

    mean_valid = lambda values: float(np.nanmean(values)) if np.any(~np.isnan(values)) else 0.0
    total = int(confusion.sum())
    mean_dice = mean_valid(dices)
    results = {
        "mIoU": mean_valid(ious),
        "mDice": mean_dice,
        "macro_f1": mean_dice,
        "macro_precision": mean_valid(precisions),
        "macro_recall": mean_valid(recalls),
        "pixel_accuracy": float(true_positive.sum() / total) if total else 0.0,
    }
    for class_index in range(confusion.shape[0]):
        results[f"iou_class_{class_index}"] = float(ious[class_index])
        results[f"dice_class_{class_index}"] = float(dices[class_index])
        results[f"precision_class_{class_index}"] = float(precisions[class_index])
        results[f"recall_class_{class_index}"] = float(recalls[class_index])
        results[f"support_class_{class_index}"] = int(true_support[class_index])
        results[f"predicted_support_class_{class_index}"] = int(predicted_support[class_index])
    return results


def compute_representation_health(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Measures representation health metrics:
      - Effective Rank: Entropy of singular values (detects dimensional collapse)
      - Active Dimensions: Percentage of singular values > 1e-4 * sigma_max
    """
    flat = embeddings.flatten(0, -2).float() # [N_samples, D]
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    
    # Singular values
    _, S, _ = torch.linalg.svd(flat_centered, full_matrices=False)
    p = S / (S.sum() + 1e-12)
    # Shannon entropy
    entropy = -(p * torch.log(p + 1e-12)).sum().item()
    eff_rank = float(np.exp(entropy))

    active_dims = int((S > 1e-4 * S[0]).sum().item())
    active_pct = float(100.0 * active_dims / len(S))

    return {
        "effective_rank": eff_rank,
        "max_rank": float(len(S)),
        "active_dims": active_dims,
        "active_dims_pct": active_pct,
    }

