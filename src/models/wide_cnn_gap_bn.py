"""Wider BatchNorm CNN with global average pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WideCNNGAPBatchNorm(nn.Module):
    """Classify RGB images using wider normalized convolution blocks."""

    def __init__(self) -> None:
        """Define 32-64-128 feature blocks and a compact GAP classifier."""
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw Cat/Dog logits for a batch of RGB images."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
