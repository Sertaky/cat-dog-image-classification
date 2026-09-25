"""Evaluate the selected fine-tuned ResNet18 checkpoint on the test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.dataloaders import build_resnet18_transfer_dataloaders
from src.models import ResNet18Transfer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "resnet18_finetune_best.pt"
REPORT_PATH = PROJECT_ROOT / "reports" / "resnet18_finetune_test.json"
EXPECTED_CHECKPOINT_EPOCH = 7
EXPECTED_TEST_SAMPLES = 3_750


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Divide two values, returning zero when the denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


def precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """Calculate precision, recall, and F1 from class-specific counts."""
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def load_selected_model(device: torch.device) -> tuple[ResNet18Transfer, int]:
    """Load and validate the selected epoch-7 fine-tuned checkpoint."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    checkpoint_epoch = int(checkpoint["epoch"])
    if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(
            f"Expected checkpoint epoch {EXPECTED_CHECKPOINT_EPOCH}, "
            f"received {checkpoint_epoch}"
        )
    if checkpoint.get("freeze_backbone") is not False:
        raise ValueError("Selected checkpoint is not a full fine-tuning checkpoint")

    model = ResNet18Transfer(freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint_epoch


def evaluate_test_split(
    model: nn.Module,
    test_loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, int, int, torch.Tensor]:
    """Evaluate every test sample and accumulate loss and confusion counts."""
    non_blocking = device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion_matrix = torch.zeros((2, 2), dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            logits = model(images)
            expected_shape = (labels.size(0), 2)
            if tuple(logits.shape) != expected_shape:
                raise AssertionError(
                    f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}"
                )

            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            encoded = labels * 2 + predictions
            confusion_matrix += (
                torch.bincount(encoded, minlength=4).reshape(2, 2).cpu()
            )

    if total_samples == 0:
        raise ValueError("Test DataLoader yielded no samples")
    return total_loss / total_samples, total_correct, total_samples, confusion_matrix


def build_report(
    *,
    checkpoint_epoch: int,
    loss: float,
    correct: int,
    total: int,
    confusion_matrix: torch.Tensor,
) -> dict[str, Any]:
    """Build and validate the deterministic test-evaluation report."""
    true_cat_pred_cat = int(confusion_matrix[0, 0].item())
    true_cat_pred_dog = int(confusion_matrix[0, 1].item())
    true_dog_pred_cat = int(confusion_matrix[1, 0].item())
    true_dog_pred_dog = int(confusion_matrix[1, 1].item())

    confusion_total = int(confusion_matrix.sum().item())
    confusion_correct = true_cat_pred_cat + true_dog_pred_dog
    if total != EXPECTED_TEST_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_TEST_SAMPLES} test predictions, received {total}"
        )
    if confusion_total != total:
        raise AssertionError("Confusion-matrix counts do not equal the test total")
    if confusion_correct != correct:
        raise AssertionError("Correct count does not match the confusion-matrix diagonal")

    accuracy = safe_divide(correct, total)
    cat_metrics = precision_recall_f1(
        true_positive=true_cat_pred_cat,
        false_positive=true_dog_pred_cat,
        false_negative=true_cat_pred_dog,
    )
    dog_metrics = precision_recall_f1(
        true_positive=true_dog_pred_dog,
        false_positive=true_cat_pred_dog,
        false_negative=true_dog_pred_cat,
    )

    return {
        "metadata": {
            "model": "ResNet18Transfer",
            "checkpoint": "checkpoints/resnet18_finetune_best.pt",
            "checkpoint_epoch": checkpoint_epoch,
            "selection_basis": "best validation accuracy",
            "test_samples": total,
        },
        "overall": {
            "loss": round(loss, 6),
            "accuracy": round(accuracy, 6),
            "correct": correct,
            "total": total,
        },
        "confusion_matrix": {
            "true_cat_pred_cat": true_cat_pred_cat,
            "true_cat_pred_dog": true_cat_pred_dog,
            "true_dog_pred_cat": true_dog_pred_cat,
            "true_dog_pred_dog": true_dog_pred_dog,
        },
        "per_class": {
            "Cat": cat_metrics,
            "Dog": dog_metrics,
        },
        "macro": {
            "precision": round(
                (cat_metrics["precision"] + dog_metrics["precision"]) / 2.0,
                6,
            ),
            "recall": round(
                (cat_metrics["recall"] + dog_metrics["recall"]) / 2.0,
                6,
            ),
            "f1": round(
                (cat_metrics["f1"] + dog_metrics["f1"]) / 2.0,
                6,
            ),
        },
    }


def main() -> None:
    """Run the one-time final test evaluation and write its JSON report."""
    batch_size = 32
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    model, checkpoint_epoch = load_selected_model(device)
    _, _, test_loader = build_resnet18_transfer_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if len(test_loader.dataset) != EXPECTED_TEST_SAMPLES:
        raise AssertionError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, "
            f"received {len(test_loader.dataset)}"
        )

    loss, correct, total, confusion_matrix = evaluate_test_split(
        model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
    )
    report = build_report(
        checkpoint_epoch=checkpoint_epoch,
        loss=loss,
        correct=correct,
        total=total,
        confusion_matrix=confusion_matrix,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    print(f"Selected device: {device}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Test samples: {total}")
    print(f"Test loss: {report['overall']['loss']:.6f}")
    print(f"Test accuracy: {report['overall']['accuracy']:.2%}")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Confusion matrix: {confusion_matrix.tolist()}")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
