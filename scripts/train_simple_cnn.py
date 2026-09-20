"""Train the SimpleCNN baseline on the manifest-backed Cat/Dog dataset."""

from __future__ import annotations

import argparse
import random
from itertools import islice
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.optim import Adam

from src.dataloaders import build_classification_dataloaders
from src.models import SimpleCNN
from src.training import train_one_epoch, validate_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "simple_cnn_best.pt"


def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Adam,
    val_accuracy: float,
    val_loss: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    random_seed: int,
) -> None:
    """Save a complete best-model training checkpoint."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "training_configuration": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "random_seed": random_seed,
            },
        },
        CHECKPOINT_PATH,
    )


def run_smoke_verification(
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    random_seed: int,
) -> None:
    """Verify imports, loaders, CUDA transfer, optimization, and output setup."""
    set_random_seed(random_seed)
    pin_memory = device.type == "cuda"
    train_loader, val_loader, _ = build_classification_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_one_epoch(
        model,
        islice(train_loader, 1),
        criterion,
        optimizer,
        device,
    )

    print(f"Selected device: {device}")
    print(f"Model device: {next(model.parameters()).device}")
    print(
        "Loader lengths: "
        f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}"
    )
    print(
        "One training batch completed: "
        f"loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.2%}"
    )
    print(f"Checkpoint directory ready: {CHECKPOINT_PATH.parent}")


def run_training(
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    epochs: int,
    random_seed: int,
) -> None:
    """Run the complete training and validation experiment."""
    set_random_seed(random_seed)
    pin_memory = device.type == "cuda"
    train_loader, val_loader, _ = build_classification_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_val_accuracy = float("-inf")
    best_epoch = 0
    started_at = perf_counter()

    print(f"Selected device: {device}")
    print(
        f"Configuration: epochs={epochs}, batch_size={batch_size}, "
        f"num_workers={num_workers}, learning_rate={learning_rate}, "
        f"pin_memory={pin_memory}, random_seed={random_seed}"
    )

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_accuracy=val_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                random_seed=random_seed,
            )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Accuracy: {train_metrics['accuracy']:.2%} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.2%}",
            flush=True,
        )

    runtime_seconds = perf_counter() - started_at
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Total training runtime: {runtime_seconds:.2f} seconds")


def parse_args() -> argparse.Namespace:
    """Parse the optional smoke-verification flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one training batch without starting the full experiment",
    )
    return parser.parse_args()


def main() -> None:
    """Configure and run smoke verification or the full baseline experiment."""
    random_seed = 42
    batch_size = 32
    num_workers = 4
    learning_rate = 0.001
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    if args.smoke_test:
        run_smoke_verification(
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            learning_rate=learning_rate,
            random_seed=random_seed,
        )
        return

    run_training(
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        epochs=epochs,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    main()
