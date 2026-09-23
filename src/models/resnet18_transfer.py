"""Transfer-learning wrapper for a torchvision ResNet18 model."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Transfer(nn.Module):
    """Classify Cat/Dog images with an ImageNet-pretrained ResNet18."""

    def __init__(self, freeze_backbone: bool = True) -> None:
        """Load pretrained weights and replace the classifier with two outputs."""
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw Cat/Dog logits for a batch of RGB images."""
        return self.resnet(x)
