"""Rank dimension outliers in the immutable Cat/Dog image dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "PetImages"
REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_outliers.json"
CLASS_LABELS = {"Cat": 0, "Dog": 1}
RANKED_LIST_SIZE = 30
TERMINAL_LIST_SIZE = 10


@dataclass(frozen=True)
class ImageDimensions:
    """Dimension metadata collected for one image."""

    path: str
    class_name: str
    label: int
    width: int
    height: int

    @property
    def pixel_area(self) -> int:
        """Return the image area in pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Return width divided by height."""
        return self.width / self.height

    @property
    def aspect_ratio_extremeness(self) -> float:
        """Return a symmetric measure of portrait or landscape elongation."""
        return max(self.width / self.height, self.height / self.width)

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-serializable candidate record."""
        return {
            "path": self.path,
            "class_name": self.class_name,
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "pixel_area": self.pixel_area,
            "aspect_ratio": round(self.aspect_ratio, 6),
            "aspect_ratio_extremeness": round(
                self.aspect_ratio_extremeness, 6
            ),
        }


def project_relative_path(path: Path) -> str:
    """Return a portable path relative to the project root."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def find_image_files(class_directory: Path) -> list[Path]:
    """Return all class files in deterministic project-relative order."""
    return sorted(
        (path for path in class_directory.rglob("*") if path.is_file()),
        key=lambda path: project_relative_path(path).casefold(),
    )


def read_dimensions(path: Path, class_name: str, label: int) -> ImageDimensions:
    """Read image dimensions without altering or decoding the raw image."""
    with Image.open(path) as image:
        width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError(f"Image has non-positive dimensions: {project_relative_path(path)}")

    return ImageDimensions(
        path=project_relative_path(path),
        class_name=class_name,
        label=label,
        width=width,
        height=height,
    )


def collect_dimensions() -> list[ImageDimensions]:
    """Collect dimension metadata for every image in both classes."""
    records: list[ImageDimensions] = []

    for class_name, label in CLASS_LABELS.items():
        class_directory = DATASET_ROOT / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Dataset class directory not found: {class_directory}")

        records.extend(
            read_dimensions(path, class_name, label)
            for path in find_image_files(class_directory)
        )

    if not records:
        raise RuntimeError(f"No images found under: {DATASET_ROOT}")

    return records


def build_report(records: list[ImageDimensions]) -> dict[str, Any]:
    """Build statistics and threshold-free ranked outlier candidate lists."""
    widths = [record.width for record in records]
    heights = [record.height for record in records]
    areas = [record.pixel_area for record in records]
    aspect_ratios = [record.aspect_ratio for record in records]

    smallest_images = sorted(
        records,
        key=lambda record: (record.pixel_area, record.path.casefold()),
    )[:RANKED_LIST_SIZE]
    most_extreme_aspect_ratios = sorted(
        records,
        key=lambda record: (-record.aspect_ratio_extremeness, record.path.casefold()),
    )[:RANKED_LIST_SIZE]

    return {
        "metadata": {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": project_relative_path(DATASET_ROOT),
            "class_labels": CLASS_LABELS,
            "ranked_candidates_per_list": RANKED_LIST_SIZE,
            "aspect_ratio_definition": "width / height",
            "aspect_ratio_extremeness_definition": (
                "max(width / height, height / width)"
            ),
            "selection_policy": "ranking only; no exclusion threshold applied",
        },
        "summary": {
            "total_images_analyzed": len(records),
            "minimum_width": min(widths),
            "maximum_width": max(widths),
            "average_width": round(fmean(widths), 2),
            "minimum_height": min(heights),
            "maximum_height": max(heights),
            "average_height": round(fmean(heights), 2),
            "minimum_area": min(areas),
            "maximum_area": max(areas),
            "average_area": round(fmean(areas), 2),
            "minimum_aspect_ratio": round(min(aspect_ratios), 6),
            "maximum_aspect_ratio_extremeness": round(
                max(record.aspect_ratio_extremeness for record in records), 6
            ),
        },
        "smallest_images": [record.to_dict() for record in smallest_images],
        "most_extreme_aspect_ratios": [
            record.to_dict() for record in most_extreme_aspect_ratios
        ],
    }


def write_report(report: dict[str, Any]) -> None:
    """Write the compact outlier analysis report as formatted JSON."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
        report_file.write("\n")


def print_ranked_samples(title: str, records: list[dict[str, Any]]) -> None:
    """Print the leading ranked candidates from one report list."""
    print(title)
    for rank, record in enumerate(records[:TERMINAL_LIST_SIZE], start=1):
        print(
            f"  {rank:2}. {record['path']} | "
            f"{record['width']}x{record['height']} | "
            f"area={record['pixel_area']} | "
            f"aspect_extremeness={record['aspect_ratio_extremeness']:.6f}"
        )


def print_summary(report: dict[str, Any]) -> None:
    """Print concise statistics and leading manual-inspection candidates."""
    summary = report["summary"]
    print(f"Dataset outlier analysis complete: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Images analyzed: {summary['total_images_analyzed']}")
    print(
        "Dimensions: "
        f"width {summary['minimum_width']}..{summary['maximum_width']} "
        f"(average {summary['average_width']}), "
        f"height {summary['minimum_height']}..{summary['maximum_height']} "
        f"(average {summary['average_height']})"
    )
    print(
        "Area: "
        f"{summary['minimum_area']}..{summary['maximum_area']} "
        f"(average {summary['average_area']})"
    )
    print_ranked_samples("10 smallest images:", report["smallest_images"])
    print_ranked_samples(
        "10 most extreme aspect-ratio images:",
        report["most_extreme_aspect_ratios"],
    )


def main() -> None:
    """Run outlier analysis, write the report, and print a summary."""
    records = collect_dimensions()
    report = build_report(records)
    write_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
