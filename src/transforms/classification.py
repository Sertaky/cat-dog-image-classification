"""Torchvision transforms for cat-and-dog image classification."""

from torchvision import transforms


_RGB_MEAN = (0.4873452324, 0.4544898525, 0.4166837849)
_RGB_STD = (0.2603815151, 0.2535908669, 0.2564516297)


def build_train_transform() -> transforms.Compose:
    """Build the stochastic preprocessing pipeline used for training."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_RGB_MEAN, std=_RGB_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Build the deterministic preprocessing pipeline used for evaluation."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_RGB_MEAN, std=_RGB_STD),
        ]
    )
