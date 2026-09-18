"""DataLoader construction for cat-and-dog image classification."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.datasets import CatDogDataset
from src.transforms import build_eval_transform, build_train_transform


def build_classification_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Build train, validation, and test DataLoaders from split manifests."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be an integer")
    if num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to zero")
    if not isinstance(pin_memory, bool):
        raise TypeError("pin_memory must be a boolean")

    train_dataset = CatDogDataset(
        "splits/train.csv",
        transform=build_train_transform(),
    )
    val_dataset = CatDogDataset(
        "splits/val.csv",
        transform=build_eval_transform(),
    )
    test_dataset = CatDogDataset(
        "splits/test.csv",
        transform=build_eval_transform(),
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return train_loader, val_loader, test_loader


def _print_first_batch(name: str, loader: DataLoader[Any]) -> None:
    """Print tensor details for one smoke-test batch."""
    images, labels = next(iter(loader))
    print(f"{name} first batch:")
    print(f"  images shape: {tuple(images.shape)}")
    print(f"  images dtype: {images.dtype}")
    print(f"  labels shape: {tuple(labels.shape)}")
    print(f"  labels dtype: {labels.dtype}")


def main() -> None:
    """Run a Windows-safe smoke test for the default loader configuration."""
    batch_size = 32
    num_workers = 4
    pin_memory = True
    train_loader, val_loader, test_loader = build_classification_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"Configuration: batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}"
    )
    print(
        "Dataset lengths: "
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )
    print(
        "Loader batches: "
        f"train={len(train_loader)}, "
        f"val={len(val_loader)}, "
        f"test={len(test_loader)}"
    )

    _print_first_batch("Train", train_loader)
    _print_first_batch("Validation", val_loader)
    _print_first_batch("Test", test_loader)


if __name__ == "__main__":
    main()
