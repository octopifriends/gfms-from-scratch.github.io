# Tangled on 2025-08-12T17:13:44

# Patch embedding layer (Week 2)
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

@dataclass
class PatchEmbedConfig:
    in_channels: int = 3
    embed_dim: int = 256
    patch_size: int = 16

class PatchEmbedding(nn.Module):
    """Conv2d-based patchifier producing token embeddings.
    Input:  (B, C, H, W)
    Output: (B, N, D) where N = (H/ps)*(W/ps), D = embed_dim
    """
    def __init__(self, cfg: PatchEmbedConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(cfg.in_channels, cfg.embed_dim,
                              kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D, H/ps, W/ps) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        b, d, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x
