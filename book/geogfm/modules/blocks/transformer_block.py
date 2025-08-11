# Generated from book/course-materials/c02-spatial-temporal-attention-mechanisms.qmd
from __future__ import annotations
import torch
import torch.nn as nn
from geogfm.modules.attention.multihead_attention import MultiheadSelfAttention
from geogfm.modules.blocks.mlp import MLP

class TransformerBlock(nn.Module):
    """PreNorm Transformer block: LN → MHA → Residual → LN → MLP → Residual."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(embed_dim, num_heads, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
