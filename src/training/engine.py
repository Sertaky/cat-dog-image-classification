"""Single-epoch training and validation functions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


Batch = tuple[Tensor, Tensor] | list[Tensor]


class EpochMetrics(TypedDict):
    """Aggregate loss and accuracy returned for one epoch."""

    loss: float
    accuracy: float


def _finalize_metrics(
    total_loss: float,
    total_correct: int,
    total_samples: int,
) -> EpochMetrics:
    """Convert accumulated sample totals into epoch-level metrics."""
    if total_samples == 0:
        raise ValueError("The dataloader yielded no samples")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
) -> EpochMetrics:
    """Train for one pass over a dataloader and return sample-weighted metrics."""
    model.train()
    target_device = torch.device(device)
    non_blocking = target_device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(target_device, non_blocking=non_blocking)
        labels = labels.to(target_device, non_blocking=non_blocking)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return _finalize_metrics(total_loss, total_correct, total_samples)


def validate_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device | str,
) -> EpochMetrics:
    """Evaluate one dataloader pass without gradients or parameter updates."""
    model.eval()
    target_device = torch.device(device)
    non_blocking = target_device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(target_device, non_blocking=non_blocking)
            labels = labels.to(target_device, non_blocking=non_blocking)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    return _finalize_metrics(total_loss, total_correct, total_samples)
