"""Image encoder E_I: a ViT over the stacked pre+post 6-channel chip.

``config='tiny'`` builds a small randomly-initialized ViT (CPU-friendly, no
download) directly via ``timm.models.vision_transformer.VisionTransformer``.
``config='vit_b16'`` uses a timm ``vit_base_patch16_224`` pretrained
checkpoint adapted to 6 input channels and a 256x256 image (timm adapts the
patch-embed conv for the channel-count change and interpolates the position
embedding for the image-size change).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

IMG_SIZE = 256
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 256


class ImageEncoder(nn.Module):
    def __init__(
        self,
        variant: str = "tiny",
        in_chans: int = 6,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.variant = variant
        if variant == "tiny":
            from timm.models.vision_transformer import VisionTransformer

            self.backbone = VisionTransformer(
                img_size=IMG_SIZE,
                patch_size=PATCH_SIZE,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                num_classes=0,
                class_token=False,
                global_pool="avg",
            )
            self.embed_dim = embed_dim
            self.num_prefix_tokens = 0
        elif variant == "vit_b16":
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                in_chans=in_chans,
                img_size=IMG_SIZE,
                num_classes=0,
            )
            self.embed_dim = self.backbone.embed_dim
            self.num_prefix_tokens = getattr(self.backbone, "num_prefix_tokens", 1)
        else:
            raise ValueError(f"Unknown image encoder variant: {variant}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, in_chans, 256, 256) -> (patch_tokens (B, 256, D), pooled (B, D))."""
        tokens = self.backbone.forward_features(x)
        if self.num_prefix_tokens > 0:
            pooled = tokens[:, 0, :]
            patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        else:
            patch_tokens = tokens
            pooled = patch_tokens.mean(dim=1)
        return patch_tokens, pooled
