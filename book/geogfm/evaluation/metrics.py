# Generated from book/course-materials/c06-model-evaluation-analysis.qmd
from __future__ import annotations
import torch

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio over reconstructed patches.
    pred/target: (B, N, C, P, P) in [0, max_val]
    """
    mse = (pred - target) ** 2
    mse = mse.mean(dim=(-1, -2, -3, -4))  # per-sample
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse + eps)
