"""Audit the immutable Cat/Dog image dataset and write a JSON report."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "PetImages"
REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_audit.json"
CLASS_LABELS = {"Cat": 0, "Dog": 1}


def project_relative_path(path: Path) -> str:
    """Return a portable path relative to the project root."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def find_candidate_files(class_directory: Path) -> list[Path]:
    """Return all regular files below a class directory in stable order."""
    return sorted(
        (path for path in class_directory.rglob("*") if path.is_file()),
        key=lambda path: project_relative_path(path).casefold(),
    )


def audit_image(path: Path, class_name: str, label: int) -> dict[str, Any]:
    """Validate one image without modifying it and return its audit record."""
    record: dict[str, Any] = {
        "path": project_relative_path(path),
        "class_name": class_name,
        "label": label,
        "extension": path.suffix.lower(),
        "width": None,
        "height": None,
        "valid": False,
    }

    try:
        # verify() checks file integrity without decoding and changing image data.
        with Image.open(path) as image:
            image.verify()

        # Reopen because verify() leaves the image object unusable. load() forces
        # pixel decoding so truncated or otherwise unreadable data is detected.
        with Image.open(path) as image:
            image.load()
            width, height = image.size

        record.update(width=width, height=height, valid=True)
    except Exception as exc:  # Pillow can raise several format-specific errors.
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def dimension_statistics(values: list[int]) -> tuple[int | None, int | None, float | None]:
    """Return minimum, maximum, and average values, or null equivalents."""
    if not values:
        return None, None, None
    return min(values), max(values), round(fmean(values), 2)


def build_report() -> dict[str, Any]:
    """Scan the configured classes and assemble the complete audit report."""
    records: list[dict[str, Any]] = []
    extension_counts: Counter[str] = Counter()

    for class_name, label in CLASS_LABELS.items():
        class_directory = DATASET_ROOT / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Dataset class directory not found: {class_directory}")

        for path in find_candidate_files(class_directory):
            record = audit_image(path, class_name, label)
            records.append(record)
            extension_counts[record["extension"]] += 1

    valid_records = [record for record in records if record["valid"]]
    invalid_records = [record for record in records if not record["valid"]]
    widths = [record["width"] for record in valid_records]
    heights = [record["height"] for record in valid_records]
    min_width, max_width, average_width = dimension_statistics(widths)
    min_height, max_height, average_height = dimension_statistics(heights)

    valid_count_per_class = {
        class_name: sum(
            record["valid"] and record["class_name"] == class_name for record in records
        )
        for class_name in CLASS_LABELS
    }
    invalid_count_per_class = {
        class_name: sum(
            not record["valid"] and record["class_name"] == class_name
            for record in records
        )
        for class_name in CLASS_LABELS
    }

    return {
        "metadata": {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": project_relative_path(DATASET_ROOT),
            "class_labels": CLASS_LABELS,
            "validation_method": "Pillow verify followed by reopen and full pixel load",
            "dimension_statistics_scope": "valid images only",
        },
        "summary": {
            "total_files_scanned": len(records),
            "total_valid_images": len(valid_records),
            "total_invalid_images": len(invalid_records),
            "valid_count_per_class": valid_count_per_class,
            "invalid_count_per_class": invalid_count_per_class,
            "image_extension_counts": dict(sorted(extension_counts.items())),
            "minimum_width": min_width,
            "maximum_width": max_width,
            "average_width": average_width,
            "minimum_height": min_height,
            "maximum_height": max_height,
            "average_height": average_height,
        },
        "invalid_image_paths": [record["path"] for record in invalid_records],
    }


def write_report(report: dict[str, Any]) -> None:
    """Write the report as human-readable, machine-parseable JSON."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")


def print_summary(report: dict[str, Any]) -> None:
    """Print a concise audit summary and any invalid image paths."""
    summary = report["summary"]
    print(f"Dataset audit complete: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Files scanned: {summary['total_files_scanned']}")
    print(f"Valid images: {summary['total_valid_images']}")
    print(f"Invalid images: {summary['total_invalid_images']}")
    print(f"Valid per class: {summary['valid_count_per_class']}")
    print(f"Invalid per class: {summary['invalid_count_per_class']}")

    if report["invalid_image_paths"]:
        print("Invalid files:")
        for path in report["invalid_image_paths"]:
            print(f"  - {path}")


def main() -> None:
    """Run the dataset audit, persist its results, and print a summary."""
    report = build_report()
    write_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
