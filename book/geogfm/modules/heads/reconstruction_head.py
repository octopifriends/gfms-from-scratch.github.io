# Generated from book/course-materials/c03-complete-gfm-architecture.qmd
from __future__ import annotations
import torch
import torch.nn as nn

class ReconstructionHead(nn.Module):
    """Token-wise MLP to reconstruct patch pixels from latent tokens."""
    def __init__(self, embed_dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels * patch_size * patch_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D) -> (B, N, C*P*P) -> (B, N, C, P, P)
        b, n, d = tokens.shape
        x = self.linear(tokens)
        x = x.view(b, n, self.out_channels, self.patch_size, self.patch_size)
        return x
