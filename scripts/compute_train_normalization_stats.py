"""Compute streaming RGB normalization statistics from the training split."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = PROJECT_ROOT / "splits" / "train.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "train_normalization.json"
RESIZE_SIZE = (224, 224)
RESAMPLE_FILTER = Image.Resampling.BILINEAR
CHANNEL_NAMES = ("red", "green", "blue")
PIXEL_MAX_VALUE = 255


@dataclass
class ChannelAccumulator:
    """Accumulate exact channel sums and squared sums from image histograms."""

    sums: list[int] = field(default_factory=lambda: [0, 0, 0])
    squared_sums: list[int] = field(default_factory=lambda: [0, 0, 0])
    pixel_count: int = 0
    image_count: int = 0

    def update(self, histogram: list[int], pixels_per_channel: int) -> None:
        """Add one RGB image histogram to the dataset-wide accumulators."""
        expected_bins = len(CHANNEL_NAMES) * (PIXEL_MAX_VALUE + 1)
        if len(histogram) != expected_bins:
            raise ValueError(
                f"Expected an RGB histogram with {expected_bins} bins, "
                f"received {len(histogram)}"
            )

        bins_per_channel = PIXEL_MAX_VALUE + 1
        for channel_index in range(len(CHANNEL_NAMES)):
            start = channel_index * bins_per_channel
            channel_histogram = histogram[start : start + bins_per_channel]
            self.sums[channel_index] += sum(
                value * frequency
                for value, frequency in enumerate(channel_histogram)
            )
            self.squared_sums[channel_index] += sum(
                value * value * frequency
                for value, frequency in enumerate(channel_histogram)
            )

        self.pixel_count += pixels_per_channel
        self.image_count += 1

    def finalize(self) -> tuple[list[float], list[float]]:
        """Return population mean and standard deviation scaled to [0, 1]."""
        if self.pixel_count == 0:
            raise RuntimeError("Cannot compute statistics for an empty manifest")

        means: list[float] = []
        standard_deviations: list[float] = []
        for channel_sum, channel_squared_sum in zip(
            self.sums,
            self.squared_sums,
            strict=True,
        ):
            raw_mean = channel_sum / self.pixel_count
            raw_second_moment = channel_squared_sum / self.pixel_count
            raw_variance = max(raw_second_moment - (raw_mean * raw_mean), 0.0)
            means.append(raw_mean / PIXEL_MAX_VALUE)
            standard_deviations.append(
                math.sqrt(raw_variance) / PIXEL_MAX_VALUE
            )

        return means, standard_deviations


def iter_training_paths(manifest_path: Path) -> Iterator[str]:
    """Yield unique project-relative paths from the training manifest only."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    seen_paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "path" not in (reader.fieldnames or ()):
            raise ValueError(
                f"Training manifest is missing required column 'path': {manifest_path}"
            )

        for row_number, row in enumerate(reader, start=2):
            relative_path = (row["path"] or "").strip()
            if not relative_path:
                raise ValueError(f"Empty path at training manifest row {row_number}")
            if relative_path in seen_paths:
                raise ValueError(
                    f"Duplicate path at training manifest row {row_number}: "
                    f"{relative_path}"
                )
            seen_paths.add(relative_path)
            yield relative_path


def resolve_image_path(relative_path: str) -> Path:
    """Resolve and validate a project-relative training image path."""
    image_path = (PROJECT_ROOT / Path(relative_path)).resolve()
    try:
        image_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Training image path escapes the project root: {relative_path}"
        ) from exc

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Training image does not exist: {relative_path} "
            f"(resolved to {image_path})"
        )
    return image_path


def compute_statistics(manifest_path: Path) -> tuple[int, list[float], list[float]]:
    """Stream training images and compute dataset-wide RGB moments."""
    accumulator = ChannelAccumulator()
    pixels_per_image = RESIZE_SIZE[0] * RESIZE_SIZE[1]

    for relative_path in iter_training_paths(manifest_path):
        image_path = resolve_image_path(relative_path)
        try:
            with Image.open(image_path) as source_image:
                with source_image.convert("RGB") as rgb_image:
                    with rgb_image.resize(
                        RESIZE_SIZE,
                        resample=RESAMPLE_FILTER,
                    ) as resized_image:
                        accumulator.update(
                            resized_image.histogram(),
                            pixels_per_image,
                        )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to process training image: {relative_path}"
            ) from exc

    means, standard_deviations = accumulator.finalize()
    return accumulator.image_count, means, standard_deviations


def build_report(
    image_count: int,
    means: list[float],
    standard_deviations: list[float],
) -> dict[str, object]:
    """Build deterministic metadata and normalization output."""
    return {
        "metadata": {
            "source_manifest": TRAIN_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "image_count": image_count,
            "resize": list(RESIZE_SIZE),
            "resize_resampling": "bilinear",
            "pixel_scale": "[0, 1]",
            "channel_order": list(CHANNEL_NAMES),
            "standard_deviation": "population",
            "calculation": "dataset-wide pixel sums and squared sums",
        },
        "normalization": {
            "mean": [round(value, 10) for value in means],
            "std": [round(value, 10) for value in standard_deviations],
        },
    }


def write_report(report: dict[str, object]) -> None:
    """Write the deterministic normalization report as formatted JSON."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")


def print_summary(report: dict[str, object], runtime_seconds: float) -> None:
    """Print the training-only normalization result."""
    metadata = report["metadata"]
    normalization = report["normalization"]
    if not isinstance(metadata, dict) or not isinstance(normalization, dict):
        raise TypeError("Unexpected report structure")

    print(f"Training images processed: {metadata['image_count']}")
    print(f"Resize: {metadata['resize'][0]} x {metadata['resize'][1]}")
    print(f"RGB mean: {normalization['mean']}")
    print(f"RGB std: {normalization['std']}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Compute, save, and print training-split normalization statistics."""
    started_at = perf_counter()
    image_count, means, standard_deviations = compute_statistics(TRAIN_MANIFEST)
    report = build_report(image_count, means, standard_deviations)
    write_report(report)
    print_summary(report, perf_counter() - started_at)


if __name__ == "__main__":
    main()
