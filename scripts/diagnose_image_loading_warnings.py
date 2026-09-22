"""Diagnose per-image Pillow and transform warnings from split manifests."""

from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.transforms import build_eval_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "image_loading_warnings.json"
SPLIT_MANIFESTS = {
    "train": PROJECT_ROOT / "splits" / "train.csv",
    "val": PROJECT_ROOT / "splits" / "val.csv",
    "test": PROJECT_ROOT / "splits" / "test.csv",
}
REQUIRED_COLUMNS = frozenset({"path", "class_name", "label"})


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load and validate the diagnostic fields from one split manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Manifest is missing required column(s): {missing}. "
                f"File: {manifest_path}"
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            record = {
                "path": (row["path"] or "").strip(),
                "class_name": (row["class_name"] or "").strip(),
                "label": (row["label"] or "").strip(),
            }
            if not record["path"]:
                raise ValueError(f"Empty path at manifest row {row_number}")
            if not record["class_name"]:
                raise ValueError(f"Empty class_name at manifest row {row_number}")
            try:
                int(record["label"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label at manifest row {row_number}: "
                    f"{record['label']!r}"
                ) from exc
            rows.append(record)

    return rows


def resolve_image_path(relative_path: str) -> Path:
    """Resolve a project-relative image path without permitting traversal."""
    image_path = (PROJECT_ROOT / Path(relative_path)).resolve()
    try:
        image_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Manifest image path escapes the project root: {relative_path}"
        ) from exc
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image referenced by manifest does not exist: {relative_path}"
        )
    return image_path


def diagnostic_record(
    row: dict[str, str],
    split: str,
    issue_type: str,
    message: str,
) -> dict[str, str | int]:
    """Create one stable report entry for a warning or exception."""
    return {
        "path": Path(row["path"]).as_posix(),
        "split": split,
        "class_name": row["class_name"],
        "label": int(row["label"]),
        "warning_or_error_type": issue_type,
        "message": message,
    }


def inspect_sample(
    row: dict[str, str],
    split: str,
    transform: Any,
) -> tuple[list[dict[str, str | int]], bool, bool]:
    """Load and transform one sample while capturing all emitted warnings."""
    records: list[dict[str, str | int]] = []
    exception: Exception | None = None

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            image_path = resolve_image_path(row["path"])
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            transformed_image = transform(image)
            del transformed_image
        except Exception as exc:  # Continue after any per-sample loading failure.
            exception = exc

    for captured in caught_warnings:
        records.append(
            diagnostic_record(
                row,
                split,
                captured.category.__name__,
                str(captured.message),
            )
        )

    if exception is not None:
        records.append(
            diagnostic_record(
                row,
                split,
                type(exception).__name__,
                str(exception),
            )
        )

    return records, bool(caught_warnings), exception is not None


def build_report() -> dict[str, Any]:
    """Inspect all manifest entries and return a deterministic report."""
    transform = build_eval_transform()
    problematic_samples: list[dict[str, str | int]] = []
    inspected_per_split: dict[str, int] = {}
    samples_with_warnings = 0
    samples_with_exceptions = 0
    path_splits: defaultdict[str, set[str]] = defaultdict(set)

    for split, manifest_path in SPLIT_MANIFESTS.items():
        rows = load_manifest(manifest_path)
        inspected_per_split[split] = len(rows)
        for row in rows:
            records, had_warning, had_exception = inspect_sample(
                row,
                split,
                transform,
            )
            problematic_samples.extend(records)
            normalized_path = Path(row["path"]).as_posix()
            if had_warning:
                samples_with_warnings += 1
            if had_exception:
                samples_with_exceptions += 1
            if records:
                path_splits[normalized_path].add(split)

    problematic_samples.sort(
        key=lambda record: (
            str(record["path"]),
            str(record["split"]),
            str(record["warning_or_error_type"]),
            str(record["message"]),
        )
    )
    unique_problematic_paths = sorted(path_splits)
    repeated_problematic_paths = sorted(
        path for path, splits in path_splits.items() if len(splits) > 1
    )

    return {
        "metadata": {
            "schema_version": 1,
            "source_manifests": [
                manifest.relative_to(PROJECT_ROOT).as_posix()
                for manifest in SPLIT_MANIFESTS.values()
            ],
            "image_loading": "Pillow Image.open followed by convert('RGB')",
            "transform": "existing deterministic evaluation transform",
            "raw_data_policy": "read-only; no files modified or excluded",
        },
        "summary": {
            "total_samples_inspected": sum(inspected_per_split.values()),
            "train_samples_inspected": inspected_per_split["train"],
            "val_samples_inspected": inspected_per_split["val"],
            "test_samples_inspected": inspected_per_split["test"],
            "total_samples_with_warnings": samples_with_warnings,
            "total_samples_with_exceptions": samples_with_exceptions,
            "unique_problematic_paths": unique_problematic_paths,
            "problematic_paths_in_multiple_splits": repeated_problematic_paths,
        },
        "problematic_samples": problematic_samples,
    }


def write_report(report: dict[str, Any]) -> None:
    """Write the diagnostic report using stable JSON formatting."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, sort_keys=True)
        report_file.write("\n")


def main() -> None:
    """Run the diagnostic and print a concise summary."""
    report = build_report()
    write_report(report)
    summary = report["summary"]

    print(f"Samples inspected: {summary['total_samples_inspected']}")
    print(
        "Per split: "
        f"train={summary['train_samples_inspected']}, "
        f"val={summary['val_samples_inspected']}, "
        f"test={summary['test_samples_inspected']}"
    )
    print(f"Samples with warnings: {summary['total_samples_with_warnings']}")
    print(f"Samples with exceptions: {summary['total_samples_with_exceptions']}")
    print(f"Unique problematic paths: {len(summary['unique_problematic_paths'])}")
    for record in report["problematic_samples"]:
        print(
            f"- [{record['split']}] {record['path']}: "
            f"{record['warning_or_error_type']}: {record['message']}"
        )
    print(f"Report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
