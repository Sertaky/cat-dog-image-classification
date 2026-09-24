"""Fine-tune all parameters of an ImageNet-pretrained ResNet18 classifier."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.optim import Adam

from src.dataloaders import build_resnet18_transfer_dataloaders
from src.models import ResNet18Transfer
from src.training import train_one_epoch, validate_one_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_finetune_best.pt"
EXPECTED_TOTAL_PARAMETERS = 11_177_538


def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable, total - trainable


def verify_finetuning_configuration(
    model: ResNet18Transfer,
) -> tuple[int, int, int]:
    """Validate parameter counts and trainability for full fine-tuning."""
    total, trainable, frozen = count_parameters(model)
    if total != EXPECTED_TOTAL_PARAMETERS:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_PARAMETERS:,} total parameters, got {total:,}"
        )
    if trainable != EXPECTED_TOTAL_PARAMETERS or frozen != 0:
        raise AssertionError(
            "Full fine-tuning requires every model parameter to be trainable"
        )
    if not model.resnet.conv1.weight.requires_grad:
        raise AssertionError("conv1.weight must be trainable")
    if not model.resnet.fc.weight.requires_grad:
        raise AssertionError("fc.weight must be trainable")
    if not model.resnet.fc.bias.requires_grad:
        raise AssertionError("fc.bias must be trainable")
    return total, trainable, frozen


def classifier_changed(
    model: ResNet18Transfer,
    initial_weight: Tensor,
    initial_bias: Tensor,
) -> bool:
    """Return whether either final-classifier parameter changed."""
    return not (
        torch.equal(model.resnet.fc.weight.detach(), initial_weight)
        and torch.equal(model.resnet.fc.bias.detach(), initial_bias)
    )


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
    """Save the best fully fine-tuned model and training state."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_name": "ResNet18Transfer",
            "freeze_backbone": False,
            "weights": "ResNet18_Weights.DEFAULT",
            "training_configuration": {
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
    """Verify full fine-tuning with one optimization batch."""
    set_random_seed(random_seed)
    pin_memory = device.type == "cuda"
    train_loader, val_loader, _ = build_resnet18_transfer_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model = ResNet18Transfer(freeze_backbone=False).to(device)
    total, trainable, frozen = verify_finetuning_configuration(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    initial_conv1 = model.resnet.conv1.weight.detach().clone()
    initial_fc_weight = model.resnet.fc.weight.detach().clone()
    initial_fc_bias = model.resnet.fc.bias.detach().clone()
    batch = next(iter(train_loader))
    images, labels = batch

    with torch.no_grad():
        logits = model(images.to(device, non_blocking=pin_memory))
        smoke_loss = criterion(logits, labels.to(device, non_blocking=pin_memory))
    expected_shape = (labels.size(0), 2)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(
            f"Expected logits shape {expected_shape}, received {tuple(logits.shape)}"
        )

    metrics = train_one_epoch(model, [batch], criterion, optimizer, device)
    backbone_was_updated = not torch.equal(
        model.resnet.conv1.weight.detach(), initial_conv1
    )
    classifier_was_updated = classifier_changed(
        model,
        initial_fc_weight,
        initial_fc_bias,
    )
    if not backbone_was_updated:
        raise AssertionError("conv1.weight did not change during smoke verification")
    if not classifier_was_updated:
        raise AssertionError("Classifier parameters did not change")

    print(f"Selected device: {device}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")
    print(
        "Loader lengths: "
        f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}"
    )
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"CrossEntropyLoss accepted logits: {smoke_loss.item():.4f}")
    print(
        "One training batch completed: "
        f"loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.2%}"
    )
    print(f"Backbone conv1.weight changed: {backbone_was_updated}")
    print(f"Classifier parameters changed: {classifier_was_updated}")
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
    """Run the complete full fine-tuning and validation experiment."""
    set_random_seed(random_seed)
    pin_memory = device.type == "cuda"
    train_loader, val_loader, _ = build_resnet18_transfer_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model = ResNet18Transfer(freeze_backbone=False).to(device)
    total, trainable, frozen = verify_finetuning_configuration(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    initial_conv1 = model.resnet.conv1.weight.detach().clone()
    initial_fc_weight = model.resnet.fc.weight.detach().clone()
    initial_fc_bias = model.resnet.fc.bias.detach().clone()
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    started_at = perf_counter()

    print(f"Selected device: {device}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")
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
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_accuracy=best_val_accuracy,
                val_loss=best_val_loss,
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
    backbone_was_updated = not torch.equal(
        model.resnet.conv1.weight.detach(), initial_conv1
    )
    classifier_was_updated = classifier_changed(
        model,
        initial_fc_weight,
        initial_fc_bias,
    )
    if not backbone_was_updated:
        raise AssertionError("conv1.weight did not change during fine-tuning")
    if not classifier_was_updated:
        raise AssertionError("Classifier parameters did not change during fine-tuning")

    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Backbone conv1.weight changed: {backbone_was_updated}")
    print(f"Classifier parameters changed: {classifier_was_updated}")
    print(f"Checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoint size: {CHECKPOINT_PATH.stat().st_size:,} bytes")
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
    """Configure and run smoke verification or full fine-tuning."""
    random_seed = 42
    batch_size = 32
    num_workers = 4
    learning_rate = 0.0001
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
