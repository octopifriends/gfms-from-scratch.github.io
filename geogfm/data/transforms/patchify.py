# Generated from book/course-materials/c01-geospatial-data-foundations.qmd
from __future__ import annotations
import numpy as np
from typing import Tuple

Array = np.ndarray

def crop_to_patches(data: Array, patch_size: int, stride: int) -> Array:
    bands, height, width = data.shape
    patches_h = (height - patch_size) // stride + 1
    patches_w = (width - patch_size) // stride + 1
    new_height = (patches_h - 1) * stride + patch_size
    new_width = (patches_w - 1) * stride + patch_size
    return data[:, :new_height, :new_width]


def extract_patches(data: Array, patch_size: int, stride: int) -> Array:
    bands, height, width = data.shape
    patches = []
    for r in range(0, height - patch_size + 1, stride):
        for c in range(0, width - patch_size + 1, stride):
            patches.append(data[:, r:r+patch_size, c:c+patch_size])
    return np.stack(patches, axis=0)


def reconstruct_from_patches(patches: Array, height: int, width: int, patch_size: int) -> Array:
    # naive grid reassembly (assumes non-overlapping P stride)
    bands = patches.shape[1]
    grid_h = height // patch_size
    grid_w = width // patch_size
    out = np.zeros((bands, height, width), dtype=patches.dtype)
    idx = 0
    for r in range(grid_h):
        for c in range(grid_w):
            out[:, r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = patches[idx]
            idx += 1
    return out
