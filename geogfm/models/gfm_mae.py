# geogfm.models.gfm_mae — Masked Autoencoder wrapper (Week 4): random masking + reconstruction.
# Tangled on 2025-08-12T17:08:53

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

from geogfm.models.gfm_vit import GeoViTBackbone, ViTBackboneConfig
from geogfm.modules.heads.reconstruction_head import ReconstructionHead

@dataclass
class MAEConfig:
    vit: ViTBackboneConfig
    out_channels: int = 3
    patch_size: int = 16
    mask_ratio: float = 0.75

class MaskedAutoencoder(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg
        # Encoder backbone and reconstruction head
        self.encoder = GeoViTBackbone(cfg.vit)
        self.head = ReconstructionHead(cfg.vit.embed_dim, cfg.out_channels, cfg.patch_size)

    @torch.no_grad()
    def _random_mask(self, num_tokens: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return keep_indices and mask matrix (B, N) with 1 for masked tokens."""
        n_keep = int(round(num_tokens * (1.0 - self.cfg.mask_ratio)))
        idx = torch.stack([torch.randperm(num_tokens, device=device) for _ in range(batch_size)], dim=0)  # (B, N)
        keep = idx[:, :n_keep]
        mask = torch.ones(batch_size, num_tokens, device=device)
        mask.scatter_(1, keep, 0.0)
        return keep, mask

    def forward(self, images: torch.Tensor) -> dict:
        """Forward MAE.
        images: (B, C, H, W)
        Returns dict with: latent, reconstructions, mask
        """
        tokens = self.encoder(images)  # (B, N, D)
        b, n, d = tokens.shape
        keep, mask = self._random_mask(n, b, tokens.device)
        # Gather visible tokens
        batch_indices = torch.arange(b, device=tokens.device).unsqueeze(-1).expand(b, keep.shape[1])
        visible_tokens = tokens[batch_indices, keep]
        # For simplicity, decode all tokens by placing zeros for masked ones, then adding decoded visible back
        decoded_all = torch.zeros(b, n, self.cfg.out_channels, self.cfg.patch_size, self.cfg.patch_size, device=tokens.device)
        decoded_visible = self.head(visible_tokens)  # (B, N_keep, C, P, P)
        decoded_all[batch_indices, keep] = decoded_visible
        return {"latent": tokens, "reconstructions": decoded_all, "mask": mask}
