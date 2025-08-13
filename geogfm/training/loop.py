# geogfm.training.loop — Minimal fit/evaluate loop (Week 5): logging and patch-based loss.
# Tangled on 2025-08-12T17:08:04

from __future__ import annotations
from typing import Tuple, Callable, Optional
import time
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        outputs = model(images)
        if isinstance(outputs, dict) and "reconstructions" in outputs:
            preds = outputs["reconstructions"]
            # Target as non-overlapping patches (assumes square patches and stride=patch)
            b, c, h, w = images.shape
            p = preds.shape[-1]
            target = images.unfold(2, p, p).unfold(3, p, p).contiguous().view(b, -1, c, p, p)
            try:
                loss = loss_fn(preds, target, outputs.get("mask"))
            except TypeError:
                loss = loss_fn(preds, target)
        else:
            raise RuntimeError("Model output not supported for evaluation")
        total_loss += float(loss) * images.size(0)
        count += images.size(0)
    return total_loss / max(1, count)


def fit(model: torch.nn.Module,
        loaders: Tuple[DataLoader, Optional[DataLoader]],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        epochs: int = 1,
        device: torch.device | str = "cpu") -> None:
    device = torch.device(device)
    model.to(device)

    train_loader, val_loader = loaders
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running_loss, count = 0.0, 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if isinstance(outputs, dict) and "reconstructions" in outputs:
                preds = outputs["reconstructions"]
                b, c, h, w = images.shape
                p = preds.shape[-1]
                target = images.unfold(2, p, p).unfold(3, p, p).contiguous().view(b, -1, c, p, p)
                try:
                    loss = loss_fn(preds, target, outputs.get("mask"))
                except TypeError:
                    loss = loss_fn(preds, target)
            else:
                raise RuntimeError("Model output not supported for training")
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * images.size(0)
            count += images.size(0)
        train_loss = running_loss / max(1, count)

        msg = f"Epoch {epoch:03d} | train_loss={train_loss:.4f}"
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn)
            msg += f" | val_loss={val_loss:.4f}"
        elapsed = time.time() - start
        print(msg + f" | time={elapsed:.1f}s")
