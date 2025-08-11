# Generated from book/course-materials/c09-model-implementation-deployment.qmd
from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple

Array = np.ndarray

def window_slices(height: int, width: int, patch_size: int, stride: int) -> Iterator[Tuple[slice, slice]]:
    for r in range(0, height - patch_size + 1, stride):
        for c in range(0, width - patch_size + 1, stride):
            yield slice(r, r + patch_size), slice(c, c + patch_size)
