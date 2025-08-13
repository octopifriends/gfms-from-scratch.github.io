# geogfm.models.gfm_vit — GeoViT backbone (Week 3). PatchEmbedding + sinusoidal positional encodings + TransformerBlock stack.
# Tangled on 2025-08-12T17:08:41

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from geogfm.modules.embeddings.patch_embedding import PatchEmbedding, PatchEmbedConfig
from geogfm.modules.embeddings.positional_encoding import sinusoidal_positional_encoding
from geogfm.modules.blocks.transformer_block import TransformerBlock

@dataclass
class ViTBackboneConfig:
    in_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0

class GeoViTBackbone(nn.Module):
    def __init__(self, cfg: ViTBackboneConfig):
        super().__init__()
        self.cfg = cfg
        # Tokenization: Conv2d-based patchify + linear projection
        self.patch_embed = PatchEmbedding(PatchEmbedConfig(cfg.in_channels, cfg.embed_dim, cfg.patch_size))
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        # Fixed positional encodings for stability and speed in the session
        self.pos_embed = nn.Parameter(sinusoidal_positional_encoding(num_patches, cfg.embed_dim), requires_grad=False)
        # Encoder: stack of PreNorm Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent token sequence of shape (batch, num_tokens, embed_dim)."""
        tokens = self.patch_embed(x)  # (B, N, D)
        tokens = tokens + self.pos_embed.unsqueeze(0)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens  # (B, N, D)
