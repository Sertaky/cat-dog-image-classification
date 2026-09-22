"""Residual convolutional network for binary image classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .residual_block import ResidualBlock


class ResidualCNN(nn.Module):
    """Classify RGB images using four residual feature-extraction stages."""

    def __init__(self) -> None:
        """Define the convolutional stem, residual stages, and linear head."""
        super().__init__()

        self.stem_conv = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU()

        self.stage1 = ResidualBlock(
            in_channels=32,
            out_channels=32,
            stride=1,
        )
        self.stage2 = ResidualBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
        )
        self.stage3 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            stride=2,
        )
        self.stage4 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            stride=2,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw Cat/Dog logits for a batch of RGB images."""
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
