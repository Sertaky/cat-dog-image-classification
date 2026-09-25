"""Benchmark forward-pass inference for representative classification models."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

from src.models import DeepCNNGAPBatchNorm, ResNet18Transfer, ResNetStyleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "inference_benchmark.json"
INPUT_SHAPE = (3, 224, 224)
BATCH_SIZES = (1, 8, 32)
WARMUP_ITERATIONS = 20
TIMED_ITERATIONS = 100


@dataclass(frozen=True)
class ModelSpec:
    """Describe one architecture and its selected checkpoint."""

    name: str
    factory: Callable[[], nn.Module]
    checkpoint_path: Path
    expected_parameters: int


MODEL_SPECS = (
    ModelSpec(
        name="DeepCNNGAPBatchNorm",
        factory=DeepCNNGAPBatchNorm,
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "deep_cnn_gap_bn_best.pt",
        expected_parameters=422_530,
    ),
    ModelSpec(
        name="ResNetStyleCNN",
        factory=ResNetStyleCNN,
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "resnet_style_cnn_best.pt",
        expected_parameters=4_906_818,
    ),
    ModelSpec(
        name="ResNet18Transfer",
        factory=lambda: ResNet18Transfer(freeze_backbone=False),
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "resnet18_finetune_best.pt",
        expected_parameters=11_177_538,
    ),
)


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work when benchmarking on a GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(spec: ModelSpec, device: torch.device) -> nn.Module:
    """Instantiate a model and load its selected checkpoint outside timing."""
    if not spec.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint_path}")

    checkpoint = torch.load(
        spec.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = spec.factory()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and currently trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def benchmark_case(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
) -> dict[str, int | float | None]:
    """Benchmark one model and batch size using synchronized forward passes."""
    if device.type == "cuda":
        torch.cuda.empty_cache()

    inputs = torch.randn(
        (batch_size, *INPUT_SHAPE),
        dtype=torch.float32,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        output = model(inputs)
        for _ in range(WARMUP_ITERATIONS - 1):
            output = model(inputs)
        synchronize(device)

        expected_shape = (batch_size, 2)
        if tuple(output.shape) != expected_shape:
            raise AssertionError(
                f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
            )

        latencies_ms: list[float] = []
        for _ in range(TIMED_ITERATIONS):
            synchronize(device)
            started_at = perf_counter()
            output = model(inputs)
            synchronize(device)
            latencies_ms.append((perf_counter() - started_at) * 1_000.0)

    if tuple(output.shape) != (batch_size, 2):
        raise AssertionError("Model output shape changed during timed inference")

    mean_batch_latency_ms = statistics.fmean(latencies_ms)
    median_batch_latency_ms = statistics.median(latencies_ms)
    latency_per_image_ms = mean_batch_latency_ms / batch_size
    throughput = batch_size / (mean_batch_latency_ms / 1_000.0)

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    return {
        "batch_size": batch_size,
        "mean_batch_latency_ms": round(mean_batch_latency_ms, 4),
        "median_batch_latency_ms": round(median_batch_latency_ms, 4),
        "latency_per_image_ms": round(latency_per_image_ms, 4),
        "throughput_images_per_second": round(throughput, 2),
        "peak_allocated_memory_bytes": peak_allocated,
        "peak_reserved_memory_bytes": peak_reserved,
    }


def benchmark_model(spec: ModelSpec, device: torch.device) -> dict[str, Any]:
    """Load and benchmark one model across every configured batch size."""
    model = load_model(spec, device)
    total_parameters, trainable_parameters = count_parameters(model)
    if total_parameters != spec.expected_parameters:
        raise AssertionError(
            f"{spec.name} expected {spec.expected_parameters:,} parameters, "
            f"got {total_parameters:,}"
        )

    benchmarks = [
        benchmark_case(model, batch_size, device) for batch_size in BATCH_SIZES
    ]
    result = {
        "model": spec.name,
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint": spec.checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_size_bytes": spec.checkpoint_path.stat().st_size,
        "benchmarks": benchmarks,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_metadata(device: torch.device) -> dict[str, Any]:
    """Describe the benchmark method and execution environment."""
    device_name = None
    cuda_version = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        cuda_version = torch.version.cuda

    return {
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "dtype": "float32",
        "input_shape": list(INPUT_SHAPE),
        "batch_sizes": list(BATCH_SIZES),
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "timing_scope": "model forward pass only",
        "memory_scope": "runtime device memory during inference",
    }


def main() -> None:
    """Run every inference benchmark and write the JSON report."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    model_results = []
    for spec in MODEL_SPECS:
        print(f"Benchmarking {spec.name}...", flush=True)
        result = benchmark_model(spec, device)
        model_results.append(result)
        for benchmark in result["benchmarks"]:
            print(
                f"  batch={benchmark['batch_size']:>2} | "
                f"mean={benchmark['mean_batch_latency_ms']:.4f} ms | "
                f"per-image={benchmark['latency_per_image_ms']:.4f} ms | "
                f"throughput={benchmark['throughput_images_per_second']:.2f} img/s",
                flush=True,
            )

    report = {
        "metadata": build_metadata(device),
        "models": model_results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
