# Generated from book/course-materials/c04-pretraining-implementation.qmd
from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE only over masked tokens.
    pred/target: (B, N, C, P, P)
    mask: (B, N) with 1 for masked, 0 for visible tokens
    """
    b, n, c, p, q = pred.shape
    pred = pred.reshape(b, n, -1)
    target = target.reshape(b, n, -1)
    mask = mask.bool().unsqueeze(-1)  # (B, N, 1)
    diff = pred[mask] - target[mask]
    if diff.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return (diff ** 2).mean()
