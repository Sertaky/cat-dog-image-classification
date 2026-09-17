"""Create deterministic, stratified CSV manifests for the Cat/Dog dataset."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "PetImages"
SPLITS_DIRECTORY = PROJECT_ROOT / "splits"
METADATA_PATH = SPLITS_DIRECTORY / "split_metadata.json"

CLASS_LABELS = {"Cat": 0, "Dog": 1}
SPLIT_RATIOS = {
    "train": Fraction(70, 100),
    "val": Fraction(15, 100),
    "test": Fraction(15, 100),
}
RANDOM_SEED = 42
EXPECTED_TOTAL_SAMPLE_COUNT = 24_998
CSV_FIELDNAMES = ("path", "class_name", "label", "width", "height")


@dataclass(frozen=True)
class Sample:
    """Metadata needed to reference one immutable source image."""

    path: str
    class_name: str
    label: int
    width: int
    height: int

    def to_csv_row(self) -> dict[str, str | int]:
        """Return a dictionary matching the manifest CSV schema."""
        return {
            "path": self.path,
            "class_name": self.class_name,
            "label": self.label,
            "width": self.width,
            "height": self.height,
        }


def project_relative_path(path: Path) -> str:
    """Return a portable path relative to the repository root."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def find_class_files(class_directory: Path) -> list[Path]:
    """Find class files in a stable order before randomized assignment."""
    return sorted(
        (path for path in class_directory.rglob("*") if path.is_file()),
        key=lambda path: project_relative_path(path).casefold(),
    )


def read_sample(path: Path, class_name: str, label: int) -> Sample:
    """Read source dimensions without modifying or fully decoding the image."""
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        relative_path = project_relative_path(path)
        raise RuntimeError(f"Unable to read audited image: {relative_path}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive image dimensions: {project_relative_path(path)}")

    return Sample(
        path=project_relative_path(path),
        class_name=class_name,
        label=label,
        width=width,
        height=height,
    )


def collect_samples_by_class() -> dict[str, list[Sample]]:
    """Collect manifest metadata for every audited source image."""
    samples_by_class: dict[str, list[Sample]] = {}

    for class_name, label in CLASS_LABELS.items():
        class_directory = DATASET_ROOT / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Dataset class directory not found: {class_directory}")

        files = find_class_files(class_directory)
        if not files:
            raise RuntimeError(f"No images found for class: {class_name}")

        samples_by_class[class_name] = [
            read_sample(path, class_name, label) for path in files
        ]

    return samples_by_class


def allocate_counts(total: int) -> dict[str, int]:
    """Allocate integer split counts using the largest-remainder method."""
    counts = {
        split_name: (total * ratio.numerator) // ratio.denominator
        for split_name, ratio in SPLIT_RATIOS.items()
    }
    remaining = total - sum(counts.values())
    split_order = {name: index for index, name in enumerate(SPLIT_RATIOS)}

    ranked_remainders = sorted(
        SPLIT_RATIOS,
        key=lambda split_name: (
            -(
                (total * SPLIT_RATIOS[split_name].numerator)
                % SPLIT_RATIOS[split_name].denominator
            )
            / SPLIT_RATIOS[split_name].denominator,
            split_order[split_name],
        ),
    )

    for split_name in ranked_remainders[:remaining]:
        counts[split_name] += 1

    return counts


def assign_splits(
    samples_by_class: dict[str, list[Sample]],
) -> dict[str, list[Sample]]:
    """Shuffle each class reproducibly and assign its samples to each split."""
    rng = random.Random(RANDOM_SEED)
    splits: dict[str, list[Sample]] = {name: [] for name in SPLIT_RATIOS}

    for class_name in CLASS_LABELS:
        shuffled_samples = list(samples_by_class[class_name])
        rng.shuffle(shuffled_samples)
        counts = allocate_counts(len(shuffled_samples))
        start = 0

        for split_name in SPLIT_RATIOS:
            end = start + counts[split_name]
            splits[split_name].extend(shuffled_samples[start:end])
            start = end

        if start != len(shuffled_samples):
            raise AssertionError(f"Incomplete allocation for class: {class_name}")

    for records in splits.values():
        records.sort(key=lambda sample: sample.path.casefold())

    return splits


def validate_splits(
    samples_by_class: dict[str, list[Sample]],
    splits: dict[str, list[Sample]],
) -> None:
    """Validate completeness, exclusivity, files, labels, and class balance."""
    source_samples = [
        sample
        for class_name in CLASS_LABELS
        for sample in samples_by_class[class_name]
    ]
    all_split_samples = [sample for records in splits.values() for sample in records]

    if len(source_samples) != EXPECTED_TOTAL_SAMPLE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_SAMPLE_COUNT} source images, "
            f"found {len(source_samples)}"
        )
    if len(all_split_samples) != EXPECTED_TOTAL_SAMPLE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_SAMPLE_COUNT} manifest rows, "
            f"found {len(all_split_samples)}"
        )

    source_paths = {sample.path for sample in source_samples}
    split_paths = [sample.path for sample in all_split_samples]
    if len(source_paths) != len(source_samples):
        raise AssertionError("Duplicate paths exist in the source scan")
    if len(set(split_paths)) != len(split_paths):
        raise AssertionError("A path appears in more than one split")
    if set(split_paths) != source_paths:
        raise AssertionError("Split paths do not exactly match the source dataset")

    for sample in all_split_samples:
        if sample.class_name not in CLASS_LABELS:
            raise AssertionError(f"Unexpected class: {sample.class_name}")
        if sample.label != CLASS_LABELS[sample.class_name]:
            raise AssertionError(f"Incorrect label for: {sample.path}")
        if not (PROJECT_ROOT / Path(sample.path)).is_file():
            raise AssertionError(f"Referenced file does not exist: {sample.path}")

    for split_name, records in splits.items():
        class_counts = Counter(sample.class_name for sample in records)
        if set(class_counts) != set(CLASS_LABELS):
            raise AssertionError(f"Missing expected class in split: {split_name}")
        if max(class_counts.values()) - min(class_counts.values()) > 1:
            raise AssertionError(f"Class balance not preserved in split: {split_name}")

        for class_name in CLASS_LABELS:
            expected_counts = allocate_counts(len(samples_by_class[class_name]))
            if class_counts[class_name] != expected_counts[split_name]:
                raise AssertionError(
                    f"Non-optimal allocation for {class_name} in {split_name}"
                )


def build_metadata(splits: dict[str, list[Sample]]) -> dict[str, Any]:
    """Build deterministic metadata describing the generated manifests."""
    return {
        "schema_version": 1,
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            name: float(ratio) for name, ratio in SPLIT_RATIOS.items()
        },
        "total_sample_count": sum(len(records) for records in splits.values()),
        "class_labels": CLASS_LABELS,
        "counts_per_split": {
            name: len(records) for name, records in splits.items()
        },
        "counts_per_class_per_split": {
            split_name: {
                class_name: sum(
                    sample.class_name == class_name for sample in records
                )
                for class_name in CLASS_LABELS
            }
            for split_name, records in splits.items()
        },
        "generation_method": {
            "strategy": "stratified per-class shuffle with largest-remainder allocation",
            "random_number_generator": "random.Random(42)",
            "source_order": "case-insensitive project-relative path",
            "manifest_row_order": "case-insensitive project-relative path",
            "quality_filtering": "none",
        },
        "source_dataset_root": project_relative_path(DATASET_ROOT),
    }


def write_csv_manifest(path: Path, records: list[Sample]) -> None:
    """Write one manifest with a stable schema, row order, and line endings."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sample.to_csv_row() for sample in records)


def write_outputs(splits: dict[str, list[Sample]], metadata: dict[str, Any]) -> None:
    """Write all validated manifests and their deterministic metadata."""
    SPLITS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        write_csv_manifest(SPLITS_DIRECTORY / f"{split_name}.csv", records)

    with METADATA_PATH.open("w", encoding="utf-8", newline="\n") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
        metadata_file.write("\n")


def print_summary(metadata: dict[str, Any]) -> None:
    """Print split totals, class counts, and realized percentages."""
    total = metadata["total_sample_count"]
    print(f"Dataset splits created in: {SPLITS_DIRECTORY.relative_to(PROJECT_ROOT)}")
    print(f"Random seed: {metadata['random_seed']}")
    for split_name in SPLIT_RATIOS:
        count = metadata["counts_per_split"][split_name]
        class_counts = metadata["counts_per_class_per_split"][split_name]
        print(
            f"{split_name}: {count} rows ({count / total:.2%}) | "
            f"Cat={class_counts['Cat']} Dog={class_counts['Dog']}"
        )
    print(f"Total: {total} rows; invariants validated")


def main() -> None:
    """Collect, split, validate, and write deterministic manifests."""
    samples_by_class = collect_samples_by_class()
    splits = assign_splits(samples_by_class)
    validate_splits(samples_by_class, splits)
    metadata = build_metadata(splits)
    write_outputs(splits, metadata)
    print_summary(metadata)


if __name__ == "__main__":
    main()
