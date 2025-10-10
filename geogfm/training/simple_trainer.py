# Tangled on 2025-10-10T09:56:51

"""Simple training utilities for classification tasks."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple
import numpy as np


class SimpleDataset(Dataset):
    """Simple dataset for image patches and labels."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu"
) -> Dict:
    """
    Train a classifier.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        lr: Learning rate
        device: Device to use

    Returns:
        Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}")
            if val_loader is not None:
                print(f"  Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")

    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Evaluate a model.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use

    Returns:
        Test loss and accuracy
    """
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total

    return test_loss, test_acc
