"""Create reproducible experiment figures from the summary CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "reports" / "experiment_summary.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SCATTER_PATH = FIGURES_DIR / "validation_accuracy_vs_parameters.png"
BAR_PATH = FIGURES_DIR / "validation_accuracy_by_model.png"
REQUIRED_COLUMNS = frozenset({"model", "parameters", "best_val_accuracy"})


@dataclass(frozen=True)
class ExperimentResult:
    """Plotting fields loaded from one experiment-summary row."""

    model: str
    parameters: int
    best_val_accuracy: float


def load_experiment_results(summary_path: Path) -> list[ExperimentResult]:
    """Load plotting values in their existing chronological CSV order."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Experiment summary not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Experiment summary is missing column(s): {missing}")

        results = [
            ExperimentResult(
                model=(row["model"] or "").strip(),
                parameters=int(row["parameters"]),
                best_val_accuracy=float(row["best_val_accuracy"]),
            )
            for row in reader
        ]

    if not results:
        raise ValueError("Experiment summary contains no experiment rows")
    return results


def configure_plot_style() -> None:
    """Apply fixed, GitHub-friendly Matplotlib styling."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0f172a",
            "axes.titlecolor": "#0f172a",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def plot_accuracy_vs_parameters(results: list[ExperimentResult]) -> None:
    """Create a labeled accuracy-versus-parameter scatter plot."""
    figure, axis = plt.subplots(figsize=(12, 7))
    parameters = [result.parameters for result in results]
    accuracies = [result.best_val_accuracy for result in results]
    colors = plt.cm.viridis([index / max(len(results) - 1, 1) for index in range(len(results))])

    axis.scatter(
        parameters,
        accuracies,
        c=colors,
        s=85,
        edgecolors="#0f172a",
        linewidths=0.6,
        zorder=3,
    )

    vertical_offsets = (10, -16, 11, -18, 10, -15, 10, 10, 12, -14)
    maximum_parameters = max(parameters)
    for index, result in enumerate(results):
        is_rightmost = result.parameters == maximum_parameters
        axis.annotate(
            result.model,
            (result.parameters, result.best_val_accuracy),
            xytext=(-7 if is_rightmost else 7, vertical_offsets[index]),
            textcoords="offset points",
            fontsize=8.5,
            ha="right" if is_rightmost else "left",
            va="center",
        )

    axis.set_xscale("log")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_ylim(min(accuracies) - 0.035, max(accuracies) + 0.035)
    axis.set_title("Validation Accuracy vs. Total Parameters", pad=12, weight="bold")
    axis.set_xlabel("Total parameters (log scale)")
    axis.set_ylabel("Best validation accuracy")
    axis.grid(True, which="major", color="#cbd5e1", linewidth=0.7, alpha=0.7)
    axis.grid(True, which="minor", axis="x", color="#e2e8f0", linewidth=0.5, alpha=0.5)
    figure.tight_layout()
    figure.savefig(SCATTER_PATH, dpi=160)
    plt.close(figure)


def plot_accuracy_by_model(results: list[ExperimentResult]) -> None:
    """Create a chronological validation-accuracy bar chart."""
    figure, axis = plt.subplots(figsize=(14, 7.4))
    models = [result.model for result in results]
    accuracies = [result.best_val_accuracy for result in results]
    colors = plt.cm.viridis([index / max(len(results) - 1, 1) for index in range(len(results))])

    bars = axis.bar(models, accuracies, color=colors, edgecolor="#0f172a", linewidth=0.5)
    axis.bar_label(
        bars,
        labels=[f"{accuracy:.2%}" for accuracy in accuracies],
        padding=3,
        fontsize=8.5,
    )

    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_title("Best Validation Accuracy by Model", pad=12, weight="bold")
    axis.set_xlabel("Model (chronological experiment order)")
    axis.set_ylabel("Best validation accuracy")
    axis.grid(True, axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelrotation=38)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.tight_layout()
    figure.savefig(BAR_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    """Load completed experiments and generate both report figures."""
    results = load_experiment_results(SUMMARY_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    plot_accuracy_vs_parameters(results)
    plot_accuracy_by_model(results)

    print(f"Experiments plotted: {len(results)}")
    print(f"Scatter plot: {SCATTER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Bar chart: {BAR_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
