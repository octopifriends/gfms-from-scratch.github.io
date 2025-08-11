# Generated from book/course-materials/c02-spatial-temporal-attention-mechanisms.qmd
from __future__ import annotations
from typing import Optional
import math
import torch
import torch.nn as nn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, Hd)
        out = out.transpose(1, 2).reshape(bsz, num_tokens, dim)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
