# geogfm.evaluation.visualization — Reconstruction visualization utilities (Week 6).
# Tangled on 2025-08-12T17:08:11

from __future__ import annotations
import torch
import matplotlib.pyplot as plt

def show_reconstruction_grid(images: torch.Tensor, recon_patches: torch.Tensor, max_items: int = 4) -> None:
    """Show input images and reconstructed images side-by-side.
    images: (B, C, H, W)
    recon_patches: (B, N, C, P, P) reconstructed patches; will be reassembled as a naive grid.
    """
    images = images.detach().cpu()
    recon_patches = recon_patches.detach().cpu()
    b, c, h, w = images.shape
    p = recon_patches.shape[-1]
    grid_h = h // p
    grid_w = w // p

    def assemble(patches: torch.Tensor) -> torch.Tensor:
        # patches: (N, C, P, P)
        rows = []
        for r in range(grid_h):
            row = torch.cat([patches[r * grid_w + cidx] for cidx in range(grid_w)], dim=-1)
            rows.append(row)
        full = torch.cat(rows, dim=-2)
        return full

    num = min(max_items, b)
    fig, axes = plt.subplots(num, 2, figsize=(8, 4 * num))
    if num == 1:
        axes = [axes]
    for i in range(num):
        recon_full = assemble(recon_patches[i])  # (C, H, W)
        axes[i][0].imshow(images[i][0], cmap="viridis")
        axes[i][0].set_title("Input (band 1)")
        axes[i][0].axis("off")
        axes[i][1].imshow(recon_full[0], cmap="viridis")
        axes[i][1].set_title("Reconstruction (band 1)")
        axes[i][1].axis("off")
    plt.tight_layout()
    plt.show()
