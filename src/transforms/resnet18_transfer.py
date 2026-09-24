"""ImageNet preprocessing pipelines for ResNet18 transfer learning."""

from torchvision import transforms


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18_train_transform() -> transforms.Compose:
    """Build the stochastic ImageNet-normalized training pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_resnet18_eval_transform() -> transforms.Compose:
    """Build the deterministic ImageNet-normalized evaluation pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
