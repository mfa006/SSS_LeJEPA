"""
Configuration dataclasses and constants for SSS feature extraction and linear probing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch


# Acoustic semantic classes from ground-truth masks
ACOUSTIC_CLASSES: Dict[int, str] = {
    0: "Seabed Ripples",
    1: "Target Anomaly (Highlight)",
    2: "Smooth Sediment",
    3: "Acoustic Shadow / Nadir",
}

NUM_ACOUSTIC_CLASSES: int = len(ACOUSTIC_CLASSES)

# Inverse class frequency weights (Class 1 is only ~7.2% of pixels)
DEFAULT_CLASS_WEIGHTS: List[float] = [1.0, 13.8, 1.5, 1.8]


@dataclass
class SSSDataConfig:
    """Configuration for SSS dataset loading and few-shot partitioning."""
    data_dir: str = "test"
    images_subdir: str = "images"
    masks_subdir: str = "masks"
    labels_subdir: str = "class_labels"
    image_extension: str = ".tiff"
    target_size: Tuple[int, int] = (224, 224)
    
    # Radiometric contrast calibration (deterministic, zero artificial augmentation)
    enable_contrast_stretch: bool = True
    p_low: float = 0.01    # 1st percentile (clips shadow saturation/zero floor)
    p_high: float = 0.99   # 99th percentile (clips speckle spikes)

    # Few-shot data partitioning
    train_ratio: float = 0.10   # Exactly 10% for linear probing (~180 tiles)
    val_ratio: float = 0.05     # 5% for probe validation & early stopping (~90 tiles)
    test_ratio: float = 0.85    # 85% held-out test evaluation (~1,530 tiles)
    seed: int = 42


@dataclass
class ModelSpec:
    """Specification for vision foundation backbones."""
    name: str
    family: str           # "vit", "resnet", "convnext"
    model_id: str         # HuggingFace repo or timm model name
    feature_dim: int      # Channel dimension D
    spatial_downsample: int # Stride from input image to spatial grid
    param_count_m: float  # Parameters in millions


SUPPORTED_BACKBONES: Dict[str, ModelSpec] = {
    "lejepa_vits16": ModelSpec(
        name="lejepa_vits16",
        family="vit",
        model_id="OK-AI/lejepa-vits16-pretrain-in1k",
        feature_dim=384,
        spatial_downsample=16, # 14x14 grid for 224x224 input
        param_count_m=21.66,
    ),
    "resnet34": ModelSpec(
        name="resnet34",
        family="resnet",
        model_id="resnet34",
        feature_dim=512,
        spatial_downsample=32, # 7x7 grid (or 14x14 with dilation)
        param_count_m=21.28,
    ),
    "convnext_tiny": ModelSpec(
        name="convnext_tiny",
        family="convnext",
        model_id="convnext_tiny",
        feature_dim=768,
        spatial_downsample=32, # 7x7 grid (or 14x14 with dilation)
        param_count_m=27.82,
    ),
    "lejepa_vitb16": ModelSpec(
        name="lejepa_vitb16",
        family="vit",
        model_id="OK-AI/lejepa-vitb16-pretrain-in1k",
        feature_dim=768,
        spatial_downsample=16,
        param_count_m=86.57,
    ),
}


@dataclass
class LinearProbeConfig:
    """Hyperparameters for dense 1x1 linear probe training."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    norm_type: str = "batchnorm" # "batchnorm", "layernorm", "l2", "none"
    use_class_weights: bool = True
    lr_scheduler: str = "cosine" # "cosine", "step", "none"
    cache_dir: str = "eval/cache"
    checkpoint_dir: str = "eval/checkpoints"
    results_dir: str = "eval/results"
    class_weights: List[float] = field(default_factory=lambda: list(DEFAULT_CLASS_WEIGHTS))
