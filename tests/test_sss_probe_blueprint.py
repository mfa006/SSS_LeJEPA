"""
Unit tests verifying the sss_probe blueprint: dataset, models, probe, trainer, checkpointing, and metrics.
"""

import os
import pytest
import torch
import numpy as np

import sss_probe
from sss_probe.config import SSSDataConfig, LinearProbeConfig
from sss_probe.probe import DenseLinearProbe, GlobalLinearProbe
from sss_probe.metrics import (
    compute_pca_feature_map,
    evaluate_class_separation,
    compute_segmentation_metrics,
    compute_representation_health,
)
from sss_probe.trainer import DenseProbeTrainer


def test_config_initialization():
    cfg = SSSDataConfig()
    assert cfg.train_ratio == 0.10
    assert cfg.val_ratio == 0.05
    assert cfg.test_ratio == 0.85
    assert sss_probe.ACOUSTIC_CLASSES == {
        0: "Ripples",
        1: "Maerl",
        2: "Rocks",
        3: "Mud",
    }
    assert len(sss_probe.ACOUSTIC_CLASSES) == 4
    assert "resnet34" in sss_probe.SUPPORTED_BACKBONES
    assert "convnext_tiny" in sss_probe.SUPPORTED_BACKBONES
    assert "lejepa_vits16" in sss_probe.SUPPORTED_BACKBONES


def test_dense_linear_probe_forward():
    B, D, H_g, W_g = 2, 384, 14, 14
    spatial_feats = torch.randn(B, D, H_g, W_g)
    
    probe = DenseLinearProbe(in_channels=D, num_classes=4, norm_type="batchnorm")
    logits = probe(spatial_feats, out_size=(224, 224))

    assert logits.shape == (B, 4, 224, 224), f"Expected [2, 4, 224, 224], got {logits.shape}"
    assert probe.classifier.weight.shape == (4, D, 1, 1)


def test_global_linear_probe_forward():
    B, D = 4, 768
    global_feats = torch.randn(B, D)
    
    probe = GlobalLinearProbe(in_features=D, num_classes=4)
    logits = probe(global_feats)
    assert logits.shape == (B, 4)


def test_pca_feature_map_computation():
    D, H_g, W_g = 512, 7, 7
    spatial_feats = torch.randn(1, D, H_g, W_g)
    
    pca_img = compute_pca_feature_map(spatial_feats, out_size=(224, 224))
    assert pca_img.shape == (224, 224, 3)
    assert pca_img.min() >= 0.0 and pca_img.max() <= 1.0


def test_segmentation_metrics():
    gt = np.array([[0, 0, 1, 1], [2, 3, 3, 0]])
    pred = np.array([[0, 1, 1, 1], [2, 2, 3, 0]])

    metrics = compute_segmentation_metrics(pred, gt, num_classes=4)
    assert metrics["pixel_accuracy"] == 6 / 8
    assert metrics["precision_class_0"] == 1.0
    assert metrics["recall_class_0"] == 2 / 3
    assert metrics["precision_class_1"] == 2 / 3
    assert metrics["recall_class_1"] == 1.0
    assert metrics["support_class_2"] == 1
    assert metrics["predicted_support_class_2"] == 2
    assert metrics["macro_precision"] == pytest.approx((1 + 2 / 3 + 1 / 2 + 1) / 4)
    assert metrics["macro_recall"] == pytest.approx((2 / 3 + 1 + 1 + 1 / 2) / 4)


def test_dense_probe_trainer_fit_and_checkpoint(tmp_path):
    device = "cpu"
    probe = DenseLinearProbe(in_channels=64, num_classes=4, norm_type="batchnorm")
    cfg = LinearProbeConfig(epochs=3, batch_size=4, device=device)

    trainer = DenseProbeTrainer(probe, config=cfg)
    
    # Dummy cached feature maps
    N, D, H_g, W_g = 8, 64, 7, 7
    train_feats = torch.randn(N, D, H_g, W_g)
    train_masks = torch.randint(0, 4, (N, 224, 224), dtype=torch.long)
    val_feats = torch.randn(4, D, H_g, W_g)
    val_masks = torch.randint(0, 4, (4, 224, 224), dtype=torch.long)

    history = trainer.fit(train_feats, train_masks, val_feats, val_masks)
    assert "train_loss" in history
    assert "val_miou" in history
    assert len(history["train_loss"]) == 3
    assert history["best_epoch"] >= 1

    # Checkpoint persistence test
    ckpt_file = os.path.join(tmp_path, "probe_test.pt")
    trainer.save_checkpoint(ckpt_file, extra_info={"test_key": 42})
    assert os.path.exists(ckpt_file)

    # Checkpoint loading test
    new_probe = DenseLinearProbe(in_channels=64, num_classes=4, norm_type="batchnorm")
    new_trainer = DenseProbeTrainer(new_probe, config=cfg)
    loaded = new_trainer.load_checkpoint(ckpt_file)
    assert loaded["extra_info"]["test_key"] == 42

    # Prediction test
    pred = trainer.predict_mask(train_feats[0], out_size=(224, 224))
    assert pred.shape == (224, 224)
