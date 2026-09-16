"""Create contact sheets for dimension-outlier candidates in the audit report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_outliers.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "outlier_inspection"

COLUMNS = 5
TILE_WIDTH = 300
TILE_HEIGHT = 280
IMAGE_SLOT_HEIGHT = 205
TILE_PADDING = 10
GRID_GAP = 12
SHEET_MARGIN = 18
HEADER_HEIGHT = 58

SHEET_BACKGROUND = "#20242b"
TILE_BACKGROUND = "#ffffff"
TILE_BORDER = "#c8ccd2"
PRIMARY_TEXT = "#17191d"
SECONDARY_TEXT = "#505761"
HEADER_TEXT = "#ffffff"


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable bundled/system font, falling back to Pillow's default."""
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_name, size=size)
    except OSError:
        return ImageFont.load_default()


def load_candidates() -> dict[str, list[dict[str, Any]]]:
    """Load the two ranked candidate lists without rescanning the dataset."""
    with REPORT_PATH.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    required_lists = ("smallest_images", "most_extreme_aspect_ratios")
    candidates: dict[str, list[dict[str, Any]]] = {}

    for list_name in required_lists:
        records = report.get(list_name)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Report field must be a non-empty list: {list_name}")
        candidates[list_name] = records

    return candidates


def resolve_image_path(relative_path: str) -> Path:
    """Resolve and validate a report path within the project directory."""
    project_root = PROJECT_ROOT.resolve()
    image_path = (PROJECT_ROOT / Path(relative_path)).resolve()

    try:
        image_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes the project root: {relative_path}") from exc

    if not image_path.is_file():
        raise FileNotFoundError(f"Candidate image not found: {relative_path}")

    return image_path


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    """Shorten text with an ellipsis so it fits inside a tile."""
    if draw.textlength(text, font=font) <= max_width:
        return text

    suffix = "..."
    shortened = text
    while shortened and draw.textlength(shortened + suffix, font=font) > max_width:
        shortened = shortened[:-1]
    return shortened + suffix


def paste_candidate_image(
    sheet: Image.Image,
    image_path: Path,
    tile_x: int,
    tile_y: int,
) -> None:
    """Fit an image into its padded slot without stretching its aspect ratio."""
    max_size = (
        TILE_WIDTH - (2 * TILE_PADDING),
        IMAGE_SLOT_HEIGHT - (2 * TILE_PADDING),
    )

    with Image.open(image_path) as source_image:
        display_image = ImageOps.exif_transpose(source_image).convert("RGB")
        fitted_image = ImageOps.contain(
            display_image,
            max_size,
            method=Image.Resampling.LANCZOS,
        )

    image_x = tile_x + (TILE_WIDTH - fitted_image.width) // 2
    image_y = tile_y + TILE_PADDING + (max_size[1] - fitted_image.height) // 2
    sheet.paste(fitted_image, (image_x, image_y))


def create_contact_sheet(
    records: list[dict[str, Any]],
    title: str,
) -> Image.Image:
    """Render ranked candidates into a labeled, padded contact sheet."""
    rows = math.ceil(len(records) / COLUMNS)
    sheet_width = (
        (2 * SHEET_MARGIN) + (COLUMNS * TILE_WIDTH) + ((COLUMNS - 1) * GRID_GAP)
    )
    sheet_height = (
        HEADER_HEIGHT
        + (2 * SHEET_MARGIN)
        + (rows * TILE_HEIGHT)
        + ((rows - 1) * GRID_GAP)
    )

    sheet = Image.new("RGB", (sheet_width, sheet_height), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(25, bold=True)
    label_font = load_font(15, bold=True)
    detail_font = load_font(14)

    draw.text(
        (SHEET_MARGIN, 15),
        f"{title} ({len(records)} candidates)",
        fill=HEADER_TEXT,
        font=title_font,
    )

    grid_top = HEADER_HEIGHT + SHEET_MARGIN
    for index, record in enumerate(records):
        row, column = divmod(index, COLUMNS)
        tile_x = SHEET_MARGIN + column * (TILE_WIDTH + GRID_GAP)
        tile_y = grid_top + row * (TILE_HEIGHT + GRID_GAP)
        tile_bounds = (
            tile_x,
            tile_y,
            tile_x + TILE_WIDTH - 1,
            tile_y + TILE_HEIGHT - 1,
        )
        draw.rectangle(
            tile_bounds,
            fill=TILE_BACKGROUND,
            outline=TILE_BORDER,
            width=1,
        )

        relative_path = str(record["path"])
        paste_candidate_image(sheet, resolve_image_path(relative_path), tile_x, tile_y)

        path = Path(relative_path)
        compact_path = f"{path.parent.name}/{path.name}"
        label_width = TILE_WIDTH - (2 * TILE_PADDING)
        label_y = tile_y + IMAGE_SLOT_HEIGHT + 7
        path_label = fit_text(
            draw,
            f"#{index + 1:02d}  {compact_path}",
            label_font,
            label_width,
        )
        detail_label = (
            f"Class: {record['class_name']}    "
            f"Size: {record['width']} x {record['height']}"
        )
        draw.text(
            (tile_x + TILE_PADDING, label_y),
            path_label,
            fill=PRIMARY_TEXT,
            font=label_font,
        )
        draw.text(
            (tile_x + TILE_PADDING, label_y + 25),
            detail_label,
            fill=SECONDARY_TEXT,
            font=detail_font,
        )

    return sheet


def save_contact_sheet(
    records: list[dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    """Create and save one high-quality JPEG contact sheet."""
    sheet = create_contact_sheet(records, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=92, subsampling=0, optimize=True)


def main() -> None:
    """Generate both contact sheets from the existing outlier report."""
    candidates = load_candidates()
    outputs = (
        (
            "smallest_images",
            "Smallest Images by Pixel Area",
            OUTPUT_DIRECTORY / "smallest_images.jpg",
        ),
        (
            "most_extreme_aspect_ratios",
            "Most Extreme Aspect Ratios",
            OUTPUT_DIRECTORY / "extreme_aspect_ratios.jpg",
        ),
    )

    for list_name, title, output_path in outputs:
        records = candidates[list_name]
        save_contact_sheet(records, title, output_path)
        print(
            f"Created {output_path.relative_to(PROJECT_ROOT)} "
            f"with {len(records)} samples."
        )


if __name__ == "__main__":
    main()
