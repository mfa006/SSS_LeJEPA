"""
Unified Vision Backbone Extractors for LeJEPA, ResNet, and ConvNeXt.
Extracts frozen spatial feature maps [B, D, H_g, W_g] and pooled representations [B, D].
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SUPPORTED_BACKBONES, ModelSpec


def load_lejepa_model(hf_model_string: str = "OK-AI/lejepa-vits16-pretrain-in1k", revision: Optional[str] = None):
    """
    Downloads full Hugging Face repository snapshot so that configuration_vitv2.py
    and the hf_src/ directory are cleanly available on sys.path.
    Suppresses auxiliary pretraining projection head warnings via _keys_to_ignore_on_load_unexpected.
    """
    import re
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    try:
        repo_dir = snapshot_download(repo_id=hf_model_string, revision=revision, local_files_only=True)
    except Exception:
        repo_dir = snapshot_download(repo_id=hf_model_string, revision=revision)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    modelling_path = os.path.join(repo_dir, "modelling_vitv2.py")
    if os.path.exists(modelling_path):
        with open(modelling_path, "r") as f:
            code = f.read()
        patterns = "[r'cva_module_proj', r'norm_patch', r'mask_token']"
        if "_keys_to_ignore_on_load_unexpected" in code:
            code = re.sub(r"_keys_to_ignore_on_load_unexpected = .*", "_keys_to_ignore_on_load_unexpected = " + patterns, code)
        else:
            rep = "config_class = ViTv2Config" + chr(10) + "    _keys_to_ignore_on_load_unexpected = " + patterns
            code = code.replace("config_class = ViTv2Config", rep)
        with open(modelling_path, "w") as f:
            f.write(code)

    return AutoModel.from_pretrained(repo_dir, trust_remote_code=True)


class BackboneExtractor(nn.Module):
    """
    Frozen vision backbone extractor providing both spatial feature maps
    and global pooled embeddings for downstream linear probing and PCA analysis.

    Supported models:
      - 'lejepa_vits16': LeJEPA ViT-S/16 (D=384, 14x14 grid, 21.7M params)
      - 'resnet34': Standard LeJEPA paper ConvNet baseline (D=512, 7x7 grid, 21.3M params)
      - 'convnext_tiny': Modern ConvNet baseline (D=768, 7x7 grid, 27.8M params)
      - 'lejepa_vitb16': LeJEPA ViT-B/16 (D=768, 14x14 grid, 86.6M params)
      - 'benthic_dino': BenthicDINO ConvNeXt-V2 Tiny (D=768, 56x56 fused grid)
    """
    def __init__(
        self,
        model_name: str = "lejepa_vits16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        if model_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported model '{model_name}'. Choose from: {list(SUPPORTED_BACKBONES.keys())}")

        self.spec: ModelSpec = SUPPORTED_BACKBONES[model_name]
        self.family = self.spec.family
        self.feature_dim = self.spec.feature_dim

        # Instantiate backbone
        if self.family == "vit":
            self.model = load_lejepa_model(self.spec.model_id)
        elif self.family in ["resnet", "convnext"]:
            import timm
            self.model = timm.create_model(self.spec.model_id, pretrained=pretrained, num_classes=0)
        elif self.family == "benthic_dino":
            benthic_root = "/home/rahul/BenthicDINO"
            if benthic_root not in sys.path:
                sys.path.insert(0, benthic_root)

            try:
                import cumm.common as cumm_common
                bendino_root = "/home/rahul/miniconda3/envs/bendino"
                cumm_common._CACHED_CUDA_INCLUDE_LIB = (
                    Path(bendino_root + "/include"),
                    Path(bendino_root + "/lib"),
                )
            except ImportError:
                pass
            try:
                import importlib
                benthic_utils = importlib.import_module("src.utils")
                load_backbone = benthic_utils.load_backbone
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "BenthicDINO dependencies are unavailable. Use the bendino Conda environment."
                ) from exc
            self.model = load_backbone(self.spec.model_id, device=torch.device(self.device))
        else:
            raise ValueError(f"Unknown family {self.family}")

        self.model.to(self.device).eval()

        # Completely freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Extracts spatial feature maps and global representation.

        Args:
            x: Input image tensor of shape [B, 3, H, W]

        Returns:
            spatial_feats: Tensor of shape [B, D, H_grid, W_grid]
            global_feat:   Tensor of shape [B, D]
            grid_size:     (H_grid, W_grid) tuple
        """
        x = x.to(self.device)
        B = x.shape[0]

        if self.family == "vit":
            # OK-AI ViTv2 output is a dict: {'latent': [B, D], 'patch_latent': [B, N, D]}
            outputs = self.model(x)
            if isinstance(outputs, dict):
                global_feat = outputs.get("latent", outputs.get("logits"))
                patch_tokens = outputs.get("patch_latent") # [B, N, D]
            else:
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs
                global_feat = hidden[:, 0]
                patch_tokens = hidden[:, 1:]

            N = patch_tokens.shape[1]
            grid_side = int(round(math.sqrt(N))) # e.g. 14 for 196 patches
            # Reshape [B, N, D] -> [B, D, H_g, W_g]
            spatial_feats = patch_tokens.permute(0, 2, 1).reshape(B, self.feature_dim, grid_side, grid_side)
            grid_size = (grid_side, grid_side)

        elif self.family == "benthic_dino":
            if x.shape[1] != 1:
                raise ValueError(
                    f"BenthicDINO requires 1-channel input, received shape {tuple(x.shape)}"
                )
            stages = self.model._inference(x)
            spatial_feats = self.model.fusion(stages)
            if spatial_feats.shape[1] != self.feature_dim:
                raise RuntimeError(
                    f"BenthicDINO fused features have {spatial_feats.shape[1]} channels; "
                    f"expected {self.feature_dim}"
                )
            global_feat = F.adaptive_avg_pool2d(spatial_feats, (1, 1)).flatten(1)
            grid_size = (spatial_feats.shape[2], spatial_feats.shape[3])

        elif self.family in ["resnet", "convnext"]:
            # timm forward_features returns [B, D, H_g, W_g]
            feats = self.model.forward_features(x)
            if feats.ndim == 4:
                # ConvNeXt / ResNet: [B, D, H_g, W_g] or [B, H_g, W_g, D]
                if feats.shape[1] != self.feature_dim and feats.shape[-1] == self.feature_dim:
                    # Channels-last (some ConvNeXt versions): [B, H_g, W_g, D] -> [B, D, H_g, W_g]
                    feats = feats.permute(0, 3, 1, 2).contiguous()

                spatial_feats = feats
                global_feat = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1) # [B, D]
                grid_size = (feats.shape[2], feats.shape[3]) # e.g. (7, 7)
            else:
                spatial_feats = None
                global_feat = feats
                grid_size = (1, 1)

        return spatial_feats, global_feat, grid_size

