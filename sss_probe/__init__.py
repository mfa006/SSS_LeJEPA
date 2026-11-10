"""
SSS-LinearProbe: Side-Scan Sonar Representation Benchmarking Suite.
Compares LeJEPA, ResNet, and ConvNeXt using deterministic preprocessing and dense linear probing.
"""

from .config import (
    ACOUSTIC_CLASSES,
    NUM_ACOUSTIC_CLASSES,
    DEFAULT_CLASS_WEIGHTS,
    SSSDataConfig,
    ModelSpec,
    SUPPORTED_BACKBONES,
    LinearProbeConfig,
)
from .dataset import (
    preprocess_sonar_image,
    load_sonar_mask,
    load_sonar_labels,
    SSSDataset,
    partition_sss_dataset,
    get_sss_dataloaders,
)
from .backbones import BackboneExtractor, load_lejepa_model
from .probe import DenseLinearProbe, GlobalLinearProbe
from .metrics import (
    compute_pca_feature_map,
    evaluate_class_separation,
    compute_segmentation_metrics,
    compute_representation_health,
)
from .trainer import DenseProbeTrainer

__all__ = [
    "ACOUSTIC_CLASSES",
    "NUM_ACOUSTIC_CLASSES",
    "DEFAULT_CLASS_WEIGHTS",
    "SSSDataConfig",
    "ModelSpec",
    "SUPPORTED_BACKBONES",
    "LinearProbeConfig",
    "preprocess_sonar_image",
    "load_sonar_mask",
    "load_sonar_labels",
    "SSSDataset",
    "partition_sss_dataset",
    "get_sss_dataloaders",
    "BackboneExtractor",
    "load_lejepa_model",
    "DenseLinearProbe",
    "GlobalLinearProbe",
    "compute_pca_feature_map",
    "evaluate_class_separation",
    "compute_segmentation_metrics",
    "compute_representation_health",
    "DenseProbeTrainer",
]

