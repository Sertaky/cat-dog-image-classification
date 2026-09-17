"""CSV-manifest-backed PyTorch Dataset for cat-and-dog images."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = frozenset({"path", "class_name", "label", "width", "height"})


@dataclass(frozen=True)
class _SampleRecord:
    """Validated metadata for one manifest row."""

    path: str
    class_name: str
    label: int
    width: int
    height: int


Metadata = dict[str, str | int]
Transform = Callable[[Image.Image], Any]
DatasetItem = tuple[Any, int] | tuple[Any, int, Metadata]


class CatDogDataset(Dataset[DatasetItem]):
    """Load cat-and-dog samples from a committed CSV manifest.

    The manifest is read and validated once during initialization. Images are
    loaded lazily, converted to RGB, and optionally passed to an externally
    supplied transform when indexed.

    Args:
        manifest_path: Absolute path or project-relative path to a split CSV.
        transform: Optional callable applied to each RGB Pillow image.
        return_metadata: Include path, class, width, and height in each item.
    """

    def __init__(
        self,
        manifest_path: str | PathLike[str],
        transform: Transform | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.manifest_path = self._resolve_manifest_path(Path(manifest_path))
        self.transform = transform
        self.return_metadata = return_metadata
        self._records = self._load_manifest(self.manifest_path)

    @staticmethod
    def _resolve_manifest_path(manifest_path: Path) -> Path:
        """Resolve a manifest path relative to the repository root."""
        resolved_path = (
            manifest_path.resolve()
            if manifest_path.is_absolute()
            else (PROJECT_ROOT / manifest_path).resolve()
        )
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {resolved_path}")
        return resolved_path

    @staticmethod
    def _parse_integer(value: str | None, column: str, row_number: int) -> int:
        """Parse an integer field and add useful manifest context on failure."""
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid integer in column '{column}' at manifest row "
                f"{row_number}: {value!r}"
            ) from exc

    @classmethod
    def _load_manifest(cls, manifest_path: Path) -> list[_SampleRecord]:
        """Read and validate all manifest rows once."""
        records: list[_SampleRecord] = []

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

            for row_number, row in enumerate(reader, start=2):
                relative_path = (row["path"] or "").strip()
                class_name = (row["class_name"] or "").strip()
                if not relative_path:
                    raise ValueError(f"Empty path at manifest row {row_number}")
                if not class_name:
                    raise ValueError(f"Empty class_name at manifest row {row_number}")

                label = cls._parse_integer(row["label"], "label", row_number)
                width = cls._parse_integer(row["width"], "width", row_number)
                height = cls._parse_integer(row["height"], "height", row_number)
                if width <= 0 or height <= 0:
                    raise ValueError(
                        f"Non-positive dimensions at manifest row {row_number}: "
                        f"{width}x{height}"
                    )

                records.append(
                    _SampleRecord(
                        path=relative_path,
                        class_name=class_name,
                        label=label,
                        width=width,
                        height=height,
                    )
                )

        return records

    @staticmethod
    def _resolve_image_path(relative_path: str) -> Path:
        """Resolve a project-relative image path without allowing traversal."""
        image_path = (PROJECT_ROOT / Path(relative_path)).resolve()
        try:
            image_path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"Manifest image path escapes the project root: {relative_path}"
            ) from exc

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image referenced by manifest does not exist: {relative_path} "
                f"(resolved to {image_path})"
            )
        return image_path

    def __len__(self) -> int:
        """Return the number of samples loaded from the manifest."""
        return len(self._records)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Load one image, apply the optional transform, and return its label."""
        record = self._records[idx]
        image_path = self._resolve_image_path(record.path)

        with Image.open(image_path) as source_image:
            image: Any = source_image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if not self.return_metadata:
            return image, record.label

        metadata: Metadata = {
            "path": record.path,
            "class_name": record.class_name,
            "width": record.width,
            "height": record.height,
        }
        return image, record.label, metadata
