"""ResNet-style convolutional network with early spatial downsampling."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .residual_block import ResidualBlock


class ResNetStyleCNN(nn.Module):
    """Classify RGB images using a downsampling stem and residual stages."""

    def __init__(self) -> None:
        """Define the ResNet-style stem, four residual stages, and head."""
        super().__init__()

        self.stem_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU()
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = ResidualBlock(
            in_channels=64,
            out_channels=64,
            stride=1,
        )
        self.stage2 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            stride=2,
        )
        self.stage3 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            stride=2,
        )
        self.stage4 = ResidualBlock(
            in_channels=256,
            out_channels=512,
            stride=2,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw Cat/Dog logits for a batch of RGB images."""
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
