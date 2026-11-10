"""
Training, checkpointing, and evaluation engine for Dense 1x1 Linear Probing on Side-Scan Sonar.
Supports offline feature caching, learning rate scheduling, model checkpoint saving,
and test set evaluation.
"""

import os
import copy
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from .config import LinearProbeConfig, NUM_ACOUSTIC_CLASSES
from .probe import DenseLinearProbe
from .metrics import compute_segmentation_metrics


class DenseProbeTrainer:
    """
    Trains, checkpoints, and evaluates a single 1x1 linear layer on frozen spatial feature maps.
    Tracks validation mIoU, restores the best checkpoint, and persists weights to disk.
    """
    def __init__(
        self,
        probe: DenseLinearProbe,
        config: Optional[LinearProbeConfig] = None,
    ):
        self.config = config or LinearProbeConfig()
        self.device = torch.device(self.config.device)
        self.probe = probe.to(self.device)
        self.best_state_dict = None
        self.best_val_miou = -1.0
        self.best_epoch = 0

        # Set up loss with optional inverse frequency class weights
        if self.config.use_class_weights and self.config.class_weights:
            weights = torch.tensor(self.config.class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimize only the 1x1 linear layer weights
        self.optimizer = optim.AdamW(
            self.probe.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6,
            )
        else:
            self.scheduler = None

    def fit(
        self,
        train_features: torch.Tensor,
        train_masks: torch.Tensor,
        val_features: Optional[torch.Tensor] = None,
        val_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Trains the 1x1 linear layer on the 10% few-shot dataset.
        Automatically saves and restores the weights that achieved the highest validation mIoU.

        Args:
            train_features: [N_train, D, H_g, W_g] cached spatial features
            train_masks:    [N_train, H, W] ground truth masks
            val_features:   [N_val, D, H_g, W_g] validation features
            val_masks:      [N_val, H, W] validation masks
        """
        train_ds = TensorDataset(train_features, train_masks)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": [],
            "learning_rates": [],
            "best_epoch": 1,
            "best_val_miou": 0.0,
        }

        self.best_val_miou = -1.0
        self.best_epoch = 1
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            self.probe.train()
            epoch_loss = 0.0

            for feats, masks in train_loader:
                feats = feats.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                logits = self.probe(feats, out_size=(masks.shape[1], masks.shape[2]))
                loss = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * feats.size(0)

            epoch_loss /= len(train_ds)
            history["train_loss"].append(epoch_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)

            if self.scheduler:
                self.scheduler.step()

            # Validation evaluation
            if val_features is not None and val_masks is not None:
                val_metrics = self.evaluate(val_features, val_masks)
                val_loss = val_metrics.get("val_loss", 0.0)
                val_miou = val_metrics.get("mIoU", 0.0)

                history["val_loss"].append(val_loss)
                history["val_miou"].append(val_miou)

                if val_miou > self.best_val_miou:
                    self.best_val_miou = val_miou
                    self.best_epoch = epoch
                    self.best_state_dict = copy.deepcopy(self.probe.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch} (Best Val mIoU: {self.best_val_miou * 100:.2f}% at epoch {self.best_epoch})")
                        break

        # Restore the best performing model weights
        if self.best_state_dict is not None:
            self.probe.load_state_dict(self.best_state_dict)
            print(f"Restored best probe weights from epoch {self.best_epoch} (Val mIoU: {self.best_val_miou * 100:.2f}%)")

        history["best_epoch"] = self.best_epoch
        history["best_val_miou"] = self.best_val_miou
        return history

    @torch.no_grad()
    def evaluate(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluates dense segmentation accuracy, mIoU, and per-class IoU.
        """
        self.probe.eval()
        ds = TensorDataset(features, masks)
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=False)

        total_loss = 0.0
        all_preds = []
        all_gts = []

        for feats, m_batch in loader:
            feats = feats.to(self.device)
            m_batch = m_batch.to(self.device)

            logits = self.probe(feats, out_size=(m_batch.shape[1], m_batch.shape[2]))
            loss = self.criterion(logits, m_batch)
            total_loss += loss.item() * feats.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(m_batch.cpu().numpy())

        total_loss /= len(ds)
        pred_arr = np.concatenate(all_preds, axis=0) # [N, H, W]
        gt_arr = np.concatenate(all_gts, axis=0)     # [N, H, W]

        metrics = compute_segmentation_metrics(pred_arr, gt_arr)
        metrics["val_loss"] = total_loss
        return metrics

    def save_checkpoint(
        self,
        filepath: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Saves the trained linear probe weights, architecture parameters,
        and evaluation metadata to disk (.pt file).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "state_dict": self.probe.state_dict(),
            "in_channels": self.probe.in_channels,
            "num_classes": self.probe.num_classes,
            "norm_type": getattr(self.probe, "norm_type", "batchnorm"),
            "best_epoch": self.best_epoch,
            "best_val_miou": self.best_val_miou,
            "extra_info": extra_info or {},
        }
        torch.save(checkpoint, filepath)
        print(f"Saved trained linear probe checkpoint to: {filepath}")
        return filepath

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Loads trained weights from checkpoint into the current probe."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.probe.load_state_dict(checkpoint["state_dict"])
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_val_miou = checkpoint.get("best_val_miou", 0.0)
        print(f"Loaded linear probe checkpoint from: {filepath} (Best Val mIoU: {self.best_val_miou * 100:.2f}%)")
        return checkpoint

    @torch.no_grad()
    def predict_mask(self, spatial_feats: torch.Tensor, out_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Generates predicted class mask for a single image feature map."""
        self.probe.eval()
        feats = spatial_feats.to(self.device)
        if feats.ndim == 3:
            feats = feats.unsqueeze(0)
        logits = self.probe(feats, out_size=out_size)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        return pred

    @staticmethod
    def save_training_curves(history: Dict[str, Any], filepath: str) -> None:
        """Plots and saves loss and mIoU training curves across epochs."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)
        best_ep = history.get("best_epoch", 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # Loss curve
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", lw=2)
        if history["val_loss"]:
            ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange", lw=2)
        ax1.axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_ep})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation mIoU curve
        if history["val_miou"]:
            ax2.plot(epochs, [m * 100 for m in history["val_miou"]], label="Val mIoU (%)", color="tab:green", lw=2)
            ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_ep})")
            ax2.axhline(history.get("best_val_miou", 0.0) * 100, color="tab:red", linestyle=":", alpha=0.7, label=f"Peak: {history.get('best_val_miou', 0.0) * 100:.1f}%")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Mean IoU (%)")
            ax2.set_title("Validation mIoU")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved training curves to: {filepath}")
