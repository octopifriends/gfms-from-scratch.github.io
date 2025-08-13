# geogfm.training.optimizer — Optimizer builder (Week 5): AdamW by default, Adam optional.
# Tangled on 2025-08-12T17:19:40

from __future__ import annotations
from typing import Dict, Any
import torch

def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = (cfg.get("name") or "adamw").lower()
    lr = float(cfg.get("lr", 2e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
