# Tangled on 2025-10-10T09:56:51

"""Utilities for working with TerraTorch models."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np


def load_pretrained_model(
    model_name: str,
    num_classes: Optional[int] = None,
    task: str = "classification",
    device: str = "cpu"
) -> nn.Module:
    """
    Load a pretrained model.

    Args:
        model_name: Name of the model (e.g., 'prithvi_100m', 'clay_v1')
        num_classes: Number of output classes (for classification/segmentation)
        task: Task type ('classification', 'segmentation', 'embedding')
        device: Device to load model on

    Returns:
        Loaded model

    Note:
        This function requires terratorch to be installed.
        For this demo, we use simplified models instead.
    """
    try:
        from terratorch.models import get_model

        model_config = {
            'backbone': model_name,
            'task': task,
        }

        if num_classes is not None:
            model_config['num_classes'] = num_classes

        model = get_model(**model_config)
        model = model.to(device)
        model.eval()

        return model
    except ImportError:
        raise ImportError(
            "terratorch is required for loading foundation models. "
            "Install with: pip install terratorch"
        )


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    layer: str = "last",
    device: str = "cpu"
) -> torch.Tensor:
    """
    Extract features from a model.

    Args:
        model: PyTorch model
        images: Input images (B, C, H, W)
        layer: Which layer to extract from
        device: Device to use

    Returns:
        Feature tensor
    """
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        if hasattr(model, 'encode'):
            features = model.encode(images)
        else:
            features = model(images)

    return features


def get_model_info(model: nn.Module) -> Dict:
    """
    Get information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_type': type(model).__name__
    }


def prepare_satellite_image(
    image_path: str,
    target_bands: Optional[List[int]] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load and prepare a satellite image for model input.

    Args:
        image_path: Path to image file
        target_bands: Band indices to use (None = all)
        normalize: Whether to normalize to [0, 1]

    Returns:
        Tensor of shape (1, C, H, W)
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required for image loading. Install with: pip install rasterio")

    with rasterio.open(image_path) as src:
        if target_bands is None:
            data = src.read()
        else:
            data = src.read(target_bands)

    # Convert to float and normalize
    data = data.astype(np.float32)

    if normalize:
        # Handle NaN values
        valid_mask = ~np.isnan(data)
        if valid_mask.any():
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            data = (data - data_min) / (data_max - data_min + 1e-8)
            data = np.nan_to_num(data, nan=0.0)

    # Add batch dimension and convert to tensor
    tensor = torch.from_numpy(data).unsqueeze(0)

    return tensor
