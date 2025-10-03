# Tangled on 2025-08-12T18:59:10

# Sinusoidal positional encodings (Week 2)
from __future__ import annotations
import math
import torch

def sinusoidal_positional_encoding(seq_len: int, dim: int, device: torch.device | None = None) -> torch.Tensor:
    """Return (seq_len, dim) sinusoidal positional encodings."""
    device = device or torch.device("cpu")
    pe = torch.zeros(seq_len, dim, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def sinusoidal_positional_encoding_2d(height: int, width: int, dim: int, device: torch.device | None = None) -> torch.Tensor:
    """Return (height*width, dim) 2D positional encodings by concatenating two 1D encodings.
    dim must be even.
    """
    assert dim % 2 == 0, "dim must be even for 2D positional encoding"
    device = device or torch.device("cpu")
    pe_h = sinusoidal_positional_encoding(height, dim // 2, device)  # (H, D/2)
    pe_w = sinusoidal_positional_encoding(width, dim // 2, device)   # (W, D/2)
    pe_h = pe_h.unsqueeze(1).expand(height, width, dim // 2)
    pe_w = pe_w.unsqueeze(0).expand(height, width, dim // 2)
    pe = torch.cat([pe_h, pe_w], dim=-1).reshape(height * width, dim)
    return pe
