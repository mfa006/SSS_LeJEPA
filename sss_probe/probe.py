"""
Linear Probe Architectures for Side-Scan Sonar (SSS) representations.
Implements single 1x1 linear layer for dense segmentation masks and global classification.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_ACOUSTIC_CLASSES


class DenseLinearProbe(nn.Module):
    """
    Dense Linear Probe for Semantic Segmentation.
    Strips all downstream heads and attaches solely a single 1x1 convolution
    (linear projection per spatial grid token) followed by bilinear upsampling.
    
    Zero non-linearities, zero hidden layers, strictly 1 linear operation.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int = NUM_ACOUSTIC_CLASSES,
        norm_type: str = "batchnorm", # "batchnorm", "layernorm", "l2", "none"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_type = norm_type

        # Feature normalization layer (stabilizes cross-backbone scale differences)
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm2d(in_channels, affine=True)
        elif norm_type == "layernorm":
            self.norm = nn.GroupNorm(1, in_channels, affine=True)
        else:
            self.norm = nn.Identity()

        # Single 1x1 linear layer: maps D channels -> num_classes logits
        self.classifier = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True,
        )

        # Standard normal initialization
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        spatial_feats: torch.Tensor,
        out_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Args:
            spatial_feats: [B, D, H_grid, W_grid] feature map from frozen backbone
            out_size: Target image resolution (H, W) for output mask logits

        Returns:
            logits: [B, num_classes, H, W] dense class logits
        """
        if self.norm_type == "l2":
            spatial_feats = F.normalize(spatial_feats, p=2, dim=1)
        else:
            spatial_feats = self.norm(spatial_feats)

        # 1x1 linear projection: [B, D, H_g, W_g] -> [B, C, H_g, W_g]
        token_logits = self.classifier(spatial_feats)

        # Bilinear upsample to input image resolution
        dense_logits = F.interpolate(
            token_logits,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )
        return dense_logits


class GlobalLinearProbe(nn.Module):
    """
    Global Linear Probe for image-level acoustic class / target presence.
    Attaches a single nn.Linear layer on pooled representation [B, D].
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_ACOUSTIC_CLASSES,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.normalize_features = normalize_features

        self.classifier = nn.Linear(in_features, num_classes, bias=True)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: [B, D] pooled embedding vector
        Returns:
            logits: [B, num_classes] class logits
        """
        if self.normalize_features:
            global_feat = F.normalize(global_feat, p=2, dim=-1)
        return self.classifier(global_feat)

