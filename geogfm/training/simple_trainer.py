# Tangled on 2025-10-14T22:09:28

import torch
import torch.nn as nn

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates
    device : torch.device
        Device to run on

    Returns
    -------
    tuple
        (average_loss, accuracy)
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # TerraTorch models return ModelOutput object
        # Extract the tensor
        if hasattr(outputs, 'output'):
            outputs = outputs.output

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Parameters
    ----------
    model : nn.Module
        The model to validate
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to run on

    Returns
    -------
    tuple
        (average_loss, accuracy)
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Extract tensor from ModelOutput
            if hasattr(outputs, 'output'):
                outputs = outputs.output

            # Compute loss
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """
    Full training loop.

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data
    val_loader : DataLoader
        Validation data
    epochs : int
        Number of epochs
    lr : float
        Learning rate
    device : torch.device
        Device to use

    Returns
    -------
    dict
        Training history with losses and accuracies
    """
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    print(f"Training for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")
    print()

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print()

    return history
