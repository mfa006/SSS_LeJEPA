"""BenthicDINO fused-feature linear probe on pre-split SSS datasets."""

import importlib
import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ACOUSTIC_CLASSES
from .dataset import SSSDataset
from .metrics import (
    compute_pca_feature_map,
    compute_segmentation_metrics,
    compute_segmentation_metrics_from_confusion,
)


_STAGE_NAMES = ("stage0", "stage1", "stage2", "stage3", "stage4_fused")
_DEFAULT_FEW_SHOT = (1.00,)


def _benthic_utils():
    benthic_root = "/home/rahul/BenthicDINO"
    if benthic_root not in sys.path:
        sys.path.insert(0, benthic_root)
    return importlib.import_module("src.utils")


def _downsample_masks(
    masks: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    source_height, source_width = masks.shape[1:]
    row_indices = ((np.arange(height) + 0.5) * (source_height / height)).astype(int)
    col_indices = ((np.arange(width) + 0.5) * (source_width / width)).astype(int)
    return masks[:, row_indices[:, None], col_indices]


def _resolve_stage_index(feature_variant: str) -> int:
    if feature_variant == "fused":
        return _STAGE_NAMES.index("stage4_fused")
    if feature_variant in _STAGE_NAMES:
        return _STAGE_NAMES.index(feature_variant)
    raise ValueError(
        f"Unsupported BenthicDINO feature variant {feature_variant!r}; "
        f"choose 'fused' or one of {_STAGE_NAMES}."
    )


def _make_loader(
    dataset: SSSDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 2
        if device.type == "cuda":
            loader_options["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **loader_options)


def _extract_tile_samples(
    model: torch.nn.Module,
    dataset: SSSDataset,
    inference_fn,
    device: torch.device,
    batch_size: int,
    pixels_per_tile: int,
    seed: int,
    num_workers: int = 0,
    stage_indices: Sequence[int] = tuple(range(len(_STAGE_NAMES))),
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    selected_stages = tuple(stage_indices)
    if not selected_stages or any(
        stage_index < 0 or stage_index >= len(_STAGE_NAMES)
        for stage_index in selected_stages
    ):
        raise ValueError(f"Invalid BenthicDINO feature-stage indices: {selected_stages}.")

    loader = _make_loader(dataset, batch_size, device, num_workers)
    feature_samples: Dict[int, np.ndarray] = {}
    label_samples: Dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)

    for batch_start, batch in enumerate(
        tqdm(loader, desc="Sampling train pixels", unit="batch")
    ):
        _, _, stage_outputs = inference_fn(
            model,
            batch["image"],
            device=device,
            normalize=False,
            return_tensors=True,
            include_embeddings=False,
            feature_indices=selected_stages,
        )
        masks = batch["mask"].numpy()
        if len(stage_outputs) != len(selected_stages):
            raise RuntimeError(
                f"BenthicDINO returned {len(stage_outputs)} feature stages; "
                f"expected {len(selected_stages)}."
            )

        start = batch_start * batch_size
        batch_count = len(batch["image"])
        for stage_index, stage_batch in zip(selected_stages, stage_outputs):
            if stage_batch.ndim != 4 or stage_batch.shape[0] != batch_count:
                raise RuntimeError(
                    f"Unexpected { _STAGE_NAMES[stage_index] } output shape: "
                    f"{stage_batch.shape}"
                )
            _, channels, height, width = stage_batch.shape
            sample_count = min(pixels_per_tile, height * width)
            if stage_index not in feature_samples:
                feature_samples[stage_index] = np.empty(
                    (len(dataset), sample_count, channels), dtype=np.float16
                )
                label_samples[stage_index] = np.empty(
                    (len(dataset), sample_count), dtype=np.uint8
                )
            if feature_samples[stage_index].shape[1:] != (
                sample_count,
                channels,
            ):
                raise RuntimeError(
                    f"Inconsistent { _STAGE_NAMES[stage_index] } dimensions across batches."
                )

            downsampled_masks = _downsample_masks(masks, height, width)
            pixel_indices = np.empty((batch_count, sample_count), dtype=np.int64)
            for item_index in range(batch_count):
                pixel_indices[item_index] = rng.choice(
                    height * width, size=sample_count, replace=False
                )
                tile_index = start + item_index
                label_samples[stage_index][tile_index] = (
                    downsampled_masks[item_index]
                    .reshape(-1)[pixel_indices[item_index]]
                )

            indices = torch.as_tensor(pixel_indices, device=stage_batch.device)
            batch_indices = torch.arange(batch_count, device=stage_batch.device)[:, None]
            flat_features = stage_batch.flatten(2).transpose(1, 2)
            sampled_features = flat_features[batch_indices, indices]
            feature_samples[stage_index][start : start + batch_count] = (
                sampled_features.to(device="cpu", dtype=torch.float16).numpy()
            )

    if set(feature_samples) != set(selected_stages):
        raise RuntimeError(
            "Did not extract all selected BenthicDINO training feature stages."
        )
    return feature_samples, label_samples


class _PixelLinearProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_scale", feature_scale)
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear((features - self.feature_mean) / self.feature_scale)


def _feature_normalization(
    features: torch.Tensor,
    chunk_size: int = 65536,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_dim = features.shape[-1]
    device = features.device
    count = 0
    mean = torch.zeros(feature_dim, dtype=torch.float32, device=device)
    sum_squared_deviations = torch.zeros_like(mean)

    with torch.no_grad():
        for start in range(0, len(features), chunk_size):
            chunk = features[start : start + chunk_size].float()
            chunk_count = len(chunk)
            chunk_mean = chunk.mean(dim=0)
            chunk_variance = chunk.var(dim=0, unbiased=False)
            combined_count = count + chunk_count
            delta = chunk_mean - mean
            sum_squared_deviations += (
                chunk_variance * chunk_count
                + delta.square() * (count * chunk_count / combined_count)
            )
            mean += delta * (chunk_count / combined_count)
            count = combined_count

    scale = torch.sqrt(sum_squared_deviations / count).clamp_min(1e-6)
    return mean, scale


def _move_sampled_features(
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_dim = features.shape[-1]
    feature_tensor = torch.from_numpy(
        np.ascontiguousarray(features.reshape(-1, feature_dim))
    ).to(device=device, dtype=torch.float16, non_blocking=device.type == "cuda")
    label_tensor = torch.from_numpy(
        np.ascontiguousarray(labels.reshape(-1), dtype=np.int64)
    ).to(device=device, non_blocking=device.type == "cuda")
    return feature_tensor, label_tensor


@torch.no_grad()
def _evaluate_sampled_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    probe.eval()
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    sample_count = 0

    for start in range(0, len(features), batch_size):
        end = start + batch_size
        feature_batch = features[start:end].float()
        label_batch = labels[start:end]
        logits = probe(feature_batch)
        loss = criterion(logits, label_batch)
        predictions = logits.argmax(dim=1)
        encoded = label_batch * num_classes + predictions
        confusion += torch.bincount(
            encoded, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)
        total_loss += loss.detach() * len(label_batch)
        sample_count += len(label_batch)

    metrics = compute_segmentation_metrics_from_confusion(confusion.cpu().numpy())
    metrics["loss"] = float(total_loss.item() / sample_count)
    return metrics


def _fit_linear_probes(
    feature_samples: Dict[int, np.ndarray],
    label_samples: Dict[int, np.ndarray],
    val_feature_samples: Dict[int, np.ndarray],
    val_label_samples: Dict[int, np.ndarray],
    n_train_tiles: int,
    few_shot: Sequence[float],
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    lr_scheduler: str,
    early_stopping_patience: int,
) -> Tuple[
    Dict[Tuple[int, float], nn.Module],
    Dict[Tuple[int, float], Dict[str, Any]],
]:
    tile_order = np.random.default_rng(seed).permutation(n_train_tiles)
    probes: Dict[Tuple[int, float], nn.Module] = {}
    histories: Dict[Tuple[int, float], Dict[str, Any]] = {}

    for stage_index, stage_features in feature_samples.items():
        stage_labels = label_samples[stage_index]
        channels = stage_features.shape[-1]
        if stage_index not in val_feature_samples or stage_index not in val_label_samples:
            raise ValueError(f"Missing validation samples for {_STAGE_NAMES[stage_index]}.")

        for shot in few_shot:
            if not 0 < shot <= 1:
                raise ValueError(f"Few-shot fraction must be in (0, 1], got {shot}.")
            tile_count = int(shot * n_train_tiles)
            if tile_count < 1:
                raise ValueError(
                    f"Few-shot fraction {shot:.0%} selects no train tiles."
                )
            if tile_count == n_train_tiles:
                x_train = stage_features
                y_train = stage_labels
            else:
                selected_tiles = tile_order[:tile_count]
                x_train = stage_features[selected_tiles]
                y_train = stage_labels[selected_tiles]
            class_counts = np.bincount(
                y_train.reshape(-1), minlength=len(ACOUSTIC_CLASSES)
            )
            present_classes = class_counts > 0
            if present_classes.sum() < 2:
                raise ValueError(
                    f"{_STAGE_NAMES[stage_index]} at {shot:.0%} shot contains "
                    "fewer than two classes in the sampled pixels."
                )

            train_features_device, train_labels_device = _move_sampled_features(
                x_train, y_train, device
            )
            val_features_device, val_labels_device = _move_sampled_features(
                val_feature_samples[stage_index],
                val_label_samples[stage_index],
                device,
            )
            torch.manual_seed(seed)
            feature_mean, feature_scale = _feature_normalization(
                train_features_device
            )
            probe = _PixelLinearProbe(
                channels,
                len(ACOUSTIC_CLASSES),
                feature_mean,
                feature_scale,
            ).to(device)

            class_weights = np.zeros(len(ACOUSTIC_CLASSES), dtype=np.float32)
            class_weights[present_classes] = (
                class_counts.sum()
                / (present_classes.sum() * class_counts[present_classes])
            )
            criterion = nn.CrossEntropyLoss(
                weight=torch.as_tensor(class_weights, device=device)
            )
            optimizer = optim.AdamW(
                probe.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if lr_scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6
                )
            elif lr_scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=max(1, epochs // 3), gamma=0.1
                )
            elif lr_scheduler in ("none", None):
                scheduler = None
            else:
                raise ValueError(f"Unsupported learning-rate scheduler: {lr_scheduler}")

            best_state = None
            best_val_miou = -1.0
            best_epoch = 0
            patience_count = 0
            history: Dict[str, Any] = {
                "train_loss": [],
                "val_loss": [],
                "val_miou": [],
            }
            epoch_bar = tqdm(
                range(1, epochs + 1),
                desc=f"Training {_STAGE_NAMES[stage_index]} linear layer on {device}",
                unit="epoch",
            )
            for epoch in epoch_bar:
                probe.train()
                total_loss = torch.zeros((), dtype=torch.float32, device=device)
                sample_count = 0
                permutation = torch.randperm(
                    len(train_features_device), device=device
                )
                for start in range(0, len(train_features_device), batch_size):
                    indices = permutation[start : start + batch_size]
                    features = train_features_device.index_select(0, indices).float()
                    labels = train_labels_device.index_select(0, indices)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(probe(features), labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach() * len(labels)
                    sample_count += len(labels)

                if scheduler is not None:
                    scheduler.step()
                val_metrics = _evaluate_sampled_probe(
                    probe,
                    val_features_device,
                    val_labels_device,
                    criterion,
                    batch_size,
                    device,
                    len(ACOUSTIC_CLASSES),
                )
                train_loss = float(total_loss.item() / sample_count)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_miou"].append(val_metrics["mIoU"])
                epoch_bar.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_mIoU=f"{val_metrics['mIoU']:.4f}",
                )
                if val_metrics["mIoU"] > best_val_miou:
                    best_val_miou = val_metrics["mIoU"]
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().clone()
                        for name, value in probe.state_dict().items()
                    }
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= early_stopping_patience:
                        print(
                            f"Early stopping at epoch {epoch}; best sampled-val "
                            f"mIoU={best_val_miou:.4f}."
                        )
                        break

            if best_state is not None:
                probe.load_state_dict(best_state)
            probe.eval()
            probes[(stage_index, shot)] = probe
            history["best_epoch"] = best_epoch
            history["best_val_miou"] = best_val_miou
            histories[(stage_index, shot)] = history
            print(
                f"Fitted one {_STAGE_NAMES[stage_index]} linear layer on {device} "
                f"using {shot:.0%} of the train split."
            )

    return probes, histories


def _evaluate_split(
    model: torch.nn.Module,
    dataset: SSSDataset,
    split_name: str,
    classifiers: Dict[Tuple[int, float], nn.Module],
    inference_fn,
    report_fn,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    stage_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    loader = _make_loader(dataset, batch_size, device, num_workers)
    stage_truth: Dict[int, np.ndarray] = {}
    predictions: Dict[Tuple[int, float], np.ndarray] = {}
    stage_shapes: Dict[int, Tuple[int, int]] = {}
    for classifier in classifiers.values():
        classifier.eval()

    for batch_start, batch in enumerate(
        tqdm(loader, desc=f"Evaluating {split_name} on {device}", unit="batch")
    ):
        _, _, stage_outputs = inference_fn(
            model,
            batch["image"],
            device=device,
            normalize=False,
            return_tensors=True,
            include_embeddings=False,
            feature_indices=stage_indices,
        )
        masks = batch["mask"].numpy()
        start = batch_start * batch_size
        end = start + len(batch["image"])

        for stage_index, stage_batch in zip(stage_indices, stage_outputs):
            _, channels, height, width = stage_batch.shape
            mask_grid = _downsample_masks(masks, height, width)
            if stage_index not in stage_truth:
                stage_truth[stage_index] = np.empty(
                    (len(dataset), height, width), dtype=np.uint8
                )
                stage_shapes[stage_index] = (height, width)
                for classifier_stage, shot in classifiers:
                    if classifier_stage == stage_index:
                        predictions[(stage_index, shot)] = np.empty(
                            (len(dataset), height, width), dtype=np.uint8
                        )
            elif stage_shapes[stage_index] != (height, width):
                raise RuntimeError(
                    f"Inconsistent { _STAGE_NAMES[stage_index] } grid across batches."
                )

            stage_truth[stage_index][start:end] = mask_grid
            x_eval = (
                stage_batch.flatten(2)
                .transpose(1, 2)
                .reshape(-1, channels)
                .to(dtype=torch.float32)
            )
            for (classifier_stage, shot), classifier in classifiers.items():
                if classifier_stage != stage_index:
                    continue
                with torch.no_grad():
                    predicted = classifier(x_eval).argmax(dim=1)
                predictions[(stage_index, shot)][start:end] = (
                    predicted.reshape(len(batch["image"]), height, width)
                    .to(device="cpu", dtype=torch.uint8)
                    .numpy()
                )

    records: List[Dict[str, Any]] = []
    labels = sorted(ACOUSTIC_CLASSES)
    for (stage_index, shot), predicted in predictions.items():
        truth = stage_truth[stage_index]
        report, _ = report_fn(
            truth.reshape(-1), predicted.reshape(-1), labels=labels
        )
        metric_values = compute_segmentation_metrics(
            predicted, truth, num_classes=len(labels)
        )
        record = {
            "split": split_name,
            "stage": _STAGE_NAMES[stage_index],
            "method": "torch_linear",
            "few_shot": shot,
            **metric_values,
            "class_iou": {
                str(label): metric_values[f"iou_class_{label}"] for label in labels
            },
            "class_support": {
                str(label): metric_values[f"support_class_{label}"] for label in labels
            },
            "report": report,
        }
        records.append(record)
        print(
            f"{split_name} {record['stage']} torch_linear {shot:.0%}: "
            f"pixel acc={record['pixel_accuracy']:.4f}, mIoU={record['mIoU']:.4f}"
        )

    return records


def _save_training_curves(
    history: Dict[str, Any],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history["best_epoch"]

    fig, (loss_axis, miou_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
    loss_axis.plot(
        epochs, history["val_loss"], label="Sampled Val Loss", color="tab:orange"
    )
    loss_axis.axvline(
        best_epoch, color="gray", linestyle="--", alpha=0.7,
        label=f"Best Epoch ({best_epoch})",
    )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross Entropy Loss")
    loss_axis.set_title("Training and Sampled Validation Loss")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)

    miou_axis.plot(
        epochs,
        [value * 100 for value in history["val_miou"]],
        label="Sampled Val mIoU (%)",
        color="tab:green",
    )
    miou_axis.axvline(
        best_epoch, color="gray", linestyle="--", alpha=0.7,
        label=f"Best Epoch ({best_epoch})",
    )
    miou_axis.set_xlabel("Epoch")
    miou_axis.set_ylabel("mIoU (%)")
    miou_axis.set_title("Sampled Validation mIoU")
    miou_axis.legend()
    miou_axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BenthicDINO training curves to: {output_path}")


def _save_test_prediction_plot(
    model: torch.nn.Module,
    dataset: SSSDataset,
    classifier: nn.Module,
    inference_fn,
    device: torch.device,
    stage_index: int,
    output_path: str,
    n_samples: int = 3,
) -> None:
    if len(dataset) == 0:
        raise ValueError("Cannot create test prediction plots from an empty dataset.")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sample_count = min(n_samples, len(dataset))
    fig, axes = plt.subplots(sample_count, 4, figsize=(16, 3.8 * sample_count))
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)

    classifier.eval()
    for row_index in range(sample_count):
        tile_index = row_index * 5 % len(dataset)
        sample = dataset[tile_index]
        image = sample["image"].unsqueeze(0).to(device)
        ground_truth = sample["mask"].numpy()

        with torch.no_grad():
            _, _, stage_outputs = inference_fn(
                model,
                image,
                device=device,
                normalize=False,
                return_tensors=True,
                include_embeddings=False,
                feature_indices=(stage_index,),
            )
            stage_features = stage_outputs[0]
            _, channels, height, width = stage_features.shape
            logits = classifier(
                stage_features.flatten(2).transpose(1, 2).reshape(-1, channels).float()
            )
            prediction_grid = logits.argmax(dim=1).reshape(1, height, width)
            prediction = F.interpolate(
                prediction_grid[:, None].float(),
                size=ground_truth.shape,
                mode="nearest",
            )[0, 0].to(torch.uint8).cpu().numpy()
            pca_map = compute_pca_feature_map(
                stage_features.cpu(), out_size=ground_truth.shape
            )

        config = getattr(dataset, "config", None)
        mean = float(getattr(config, "benthic_mean", 0.0))
        std = float(getattr(config, "benthic_std", 1.0))
        gray_tile = (sample["image"][0].numpy() * std + mean).clip(0, 1)
        filename = sample.get("filename", f"tile {tile_index}")
        panels = (
            (gray_tile, "copper", f"Test Tile\n{filename}"),
            (ground_truth, "tab10", "Ground Truth Mask"),
            (pca_map, None, "BenthicDINO PCA Features"),
            (prediction, "tab10", "Linear Probe Prediction"),
        )
        for axis, (image_data, color_map, title) in zip(axes[row_index], panels):
            axis.imshow(
                image_data,
                cmap=color_map,
                vmin=0 if color_map == "tab10" else None,
                vmax=len(ACOUSTIC_CLASSES) - 1 if color_map == "tab10" else None,
            )
            axis.set_title(title, fontsize=9)
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BenthicDINO test predictions to: {output_path}")


def run_benthic_s3seg_benchmark(
    model: torch.nn.Module,
    train_dataset: SSSDataset,
    val_dataset: SSSDataset,
    test_dataset: SSSDataset,
    device: str,
    results_dir: str,
    checkpoint_dir: str = "eval/checkpoints",
    config_file: str = "",
    batch_size: int = 8,
    pixels_per_tile: int = 256,
    few_shot: Sequence[float] = _DEFAULT_FEW_SHOT,
    feature_variant: str = "fused",
    epochs: int = 30,
    classifier_batch_size: int = 8192,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_scheduler: str = "cosine",
    early_stopping_patience: int = 10,
    seed: int = 42,
    num_workers: int = 0,
) -> Dict[str, Any]:
    if batch_size < 1:
        raise ValueError(f"Feature batch size must be positive, got {batch_size}.")
    if pixels_per_tile < 1:
        raise ValueError(f"Pixels per tile must be positive, got {pixels_per_tile}.")
    if epochs < 1:
        raise ValueError(f"Classifier epochs must be positive, got {epochs}.")
    if classifier_batch_size < 1:
        raise ValueError(
            f"Classifier batch size must be positive, got {classifier_batch_size}."
        )
    if early_stopping_patience < 1:
        raise ValueError(
            "Early-stopping patience must be positive, "
            f"got {early_stopping_patience}."
        )
    if len(few_shot) != 1:
        raise ValueError(
            "The BenthicDINO artifact pipeline currently supports exactly one "
            "few-shot fraction."
        )
    if num_workers < 0:
        raise ValueError(f"DataLoader worker count cannot be negative, got {num_workers}.")

    stage_index = _resolve_stage_index(feature_variant)
    stage_indices = (stage_index,)
    utils = _benthic_utils()
    torch_device = torch.device(device)
    model.eval()
    print(
        f"Training pool: {len(train_dataset)} tiles; few-shot tile counts "
        f"(fractions of this supplied split): "
        + ", ".join(
            f"{shot:.0%}={int(shot * len(train_dataset))}"
            for shot in few_shot
        )
    )

    train_features, train_labels = _extract_tile_samples(
        model,
        train_dataset,
        utils.run_inference,
        torch_device,
        batch_size,
        pixels_per_tile,
        seed,
        num_workers,
        stage_indices,
    )
    val_features, val_labels = _extract_tile_samples(
        model,
        val_dataset,
        utils.run_inference,
        torch_device,
        batch_size,
        pixels_per_tile,
        seed + 1,
        num_workers,
        stage_indices,
    )
    classifiers, histories = _fit_linear_probes(
        train_features,
        train_labels,
        val_features,
        val_labels,
        len(train_dataset),
        few_shot,
        seed,
        torch_device,
        epochs,
        classifier_batch_size,
        learning_rate,
        weight_decay,
        lr_scheduler,
        early_stopping_patience,
    )
    del train_features, train_labels, val_features, val_labels

    records = []
    records.extend(
        _evaluate_split(
            model,
            val_dataset,
            "val",
            classifiers,
            utils.run_inference,
            utils.generate_evaluation_report,
            torch_device,
            batch_size,
            num_workers,
            stage_indices,
        )
    )
    records.extend(
        _evaluate_split(
            model,
            test_dataset,
            "test",
            classifiers,
            utils.run_inference,
            utils.generate_evaluation_report,
            torch_device,
            batch_size,
            num_workers,
            stage_indices,
        )
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    classifier_key = (stage_index, float(few_shot[0]))
    classifier = classifiers[classifier_key]
    history = histories[classifier_key]
    best_epoch = int(history["best_epoch"])
    checkpoint_path = os.path.join(
        checkpoint_dir, "benthic_dino_linear_probe.pt"
    )
    torch.save(
        {
            "model_name": "benthic_dino",
            "feature_variant": feature_variant,
            "stage": _STAGE_NAMES[stage_index],
            "method": "torch_linear",
            "state_dict": classifier.state_dict(),
            "best_epoch": best_epoch,
            "best_val_miou": float(history["best_val_miou"]),
            "config_file": config_file,
            "training_history": history,
        },
        checkpoint_path,
    )

    val_metrics = next(record for record in records if record["split"] == "val")
    test_metrics = next(record for record in records if record["split"] == "test")
    training_curves_path = os.path.join(
        results_dir, "benthic_dino_training_curves.png"
    )
    _save_training_curves(history, training_curves_path)
    prediction_plot_path = os.path.join(
        results_dir, "benthic_dino_test_predictions.png"
    )
    _save_test_prediction_plot(
        model,
        test_dataset,
        classifier,
        utils.run_inference,
        torch_device,
        stage_index,
        prediction_plot_path,
    )

    metrics_path = os.path.join(results_dir, "benthic_dino_metrics.json")
    with open(metrics_path, "w") as metrics_file:
        json.dump(
            {
                "model_name": "benthic_dino",
                "class_names": {
                    str(index): name for index, name in ACOUSTIC_CLASSES.items()
                },
                "config_file": config_file,
                "best_epoch": best_epoch,
                "test_metrics": test_metrics,
                "val_metrics": val_metrics,
                "training_history": history,
                "artifacts": {
                    "checkpoint": checkpoint_path,
                    "training_curves": training_curves_path,
                    "test_predictions": prediction_plot_path,
                },
            },
            metrics_file,
            indent=2,
        )
    print(f"Saved full BenthicDINO metrics to: {metrics_path}")

    output_path = os.path.join(results_dir, "benthic_dino_linear_probe.json")
    with open(output_path, "w") as result_file:
        json.dump(
            {
                "benchmark": "BenthicDINO fused-feature PyTorch linear probe",
                "data_protocol": "pre-split train/val/test",
                "train_tile_count": len(train_dataset),
                "val_tile_count": len(val_dataset),
                "test_tile_count": len(test_dataset),
                "few_shot_tile_counts": [
                    {
                        "fraction": float(shot),
                        "tiles": int(shot * len(train_dataset)),
                    }
                    for shot in few_shot
                ],
                "few_shot_fraction_reference": "supplied pre-split train_dataset",
                "pixels_sampled_per_train_tile": pixels_per_tile,
                "feature_batch_size": batch_size,
                "data_loader_workers": num_workers,
                "feature_variant": feature_variant,
                "methods": ["torch_linear"],
                "few_shot_fractions": list(few_shot),
                "classifier_device": str(torch_device),
                "classifier_epochs": epochs,
                "classifier_batch_size": classifier_batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "lr_scheduler": lr_scheduler,
                "early_stopping_patience": early_stopping_patience,
                "seed": seed,
                "best_epoch": best_epoch,
                "artifacts": {
                    "metrics": metrics_path,
                    "checkpoint": checkpoint_path,
                    "training_curves": training_curves_path,
                    "test_predictions": prediction_plot_path,
                },
                "results": records,
            },
            result_file,
            indent=2,
        )
    print(f"Saved BenthicDINO S3Seg benchmark to: {output_path}")
    return {
        "results": records,
        "output_path": output_path,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "history": history,
    }
