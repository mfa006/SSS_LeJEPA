#!/usr/bin/env python3
"""Create presentation-ready plots comparing linear-probe benchmarks."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sss_probe.config import SUPPORTED_BACKBONES


MODEL_LABELS = {
    "benthic_dino": "BenthicDINO (ConvNeXt-V2-Tiny)",
    "lejepa_vits16": "LeJEPA ViT-S/16",
    "lejepa_vitb16": "LeJEPA ViT-B/16",
    "resnet34": "ResNet-34",
    "convnext_tiny": "ConvNeXt-Tiny",
}
CLASS_NAMES = ("Ripples", "Maerl", "Rocks", "Mud")
MODEL_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
OVERALL_METRICS = (
    ("test_mIoU_pct", "mIoU"),
    ("test_mDice_pct", "mDice"),
    ("test_macro_f1_pct", "Macro F1"),
    ("test_pixel_acc_pct", "Pixel accuracy"),
    ("test_macro_precision_pct", "Macro precision"),
    ("test_macro_recall_pct", "Macro recall"),
)
LEGACY_METRIC_KEYS = {"test_macro_f1_pct": "test_mDice_pct"}


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark summary not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as summary_file:
        rows = list(csv.DictReader(summary_file))
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")
    if "model" not in rows[0]:
        raise ValueError(f"Expected a 'model' column in {path}")
    return sorted(rows, key=lambda row: float(row.get("test_mIoU_pct") or 0), reverse=True)


def metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        value = row.get(LEGACY_METRIC_KEYS.get(key, ""))
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def model_label(row: dict[str, Any]) -> str:
    model_name = row["model"]
    feature_dim = row.get("feature_dim")
    label = MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())
    details = []
    if feature_dim:
        details.append(f"D={feature_dim}")

    parameter_count = row.get("param_count_m")
    if parameter_count in (None, ""):
        model_spec = SUPPORTED_BACKBONES.get(model_name)
        parameter_count = model_spec.param_count_m if model_spec else None
    if parameter_count not in (None, ""):
        details.append(f"{float(parameter_count):.2f}M params")

    return f"{label} ({', '.join(details)})" if details else label


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def draw_score_matrix(
    axis: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    value_range: tuple[float, float],
) -> Any:
    image = axis.imshow(
        values,
        cmap="RdYlGn",
        vmin=value_range[0],
        vmax=value_range[1],
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(column_labels)), column_labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title, loc="left", weight="bold", pad=12)
    axis.tick_params(length=0)
    axis.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="#202124",
            )
    return image


def add_score_colorbar(
    fig: plt.Figure,
    axis: plt.Axes,
    image: Any,
    value_range: tuple[float, float],
) -> None:
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Score (%)")
    colorbar.set_ticks(np.linspace(value_range[0], value_range[1], 5))


def plot_overall(rows: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = [(key, label) for key, label in OVERALL_METRICS]
    values = np.asarray(
        [[metric_value(row, key) for key, _ in metrics] for row in rows],
        dtype=np.float64,
    )
    value_range = (80.0, 100.0)
    fig, axis = plt.subplots(figsize=(13, 5.5), layout="constrained")
    image = draw_score_matrix(
        axis,
        values,
        [model_label(row) for row in rows],
        [label for _, label in metrics],
        "Overall Test Performance",
        value_range,
    )
    add_score_colorbar(fig, axis, image, value_range)
    fig.text(
        0.01,
        -0.01,
        "Color scale: 80–100%. Macro F1 equals mDice for single-label segmentation.",
        fontsize=9,
        color="#4B5563",
    )
    save_figure(fig, output_dir / "benchmark_overall")


def plot_class_iou(rows: list[dict[str, Any]], output_dir: Path) -> None:
    values = np.asarray(
        [
            [metric_value(row, f"test_iou_{name.lower()}_pct") for name in CLASS_NAMES]
            for row in rows
        ],
        dtype=np.float64,
    )
    value_range = (70.0, 100.0)
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    image = draw_score_matrix(
        axis,
        values,
        [model_label(row) for row in rows],
        list(CLASS_NAMES),
        "Test IoU by Class",
        value_range,
    )
    add_score_colorbar(fig, axis, image, value_range)
    fig.text(
        0.01,
        -0.01,
        "Color scale: 70–100%. Class IDs: 0 Ripples, 1 Maerl, 2 Rocks, 3 Mud.",
        fontsize=9,
        color="#4B5563",
    )
    save_figure(fig, output_dir / "benchmark_class_iou")


def plot_class_precision_and_recall(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    value_range = (80.0, 100.0)
    row_labels = [model_label(row) for row in rows]
    for metric, title in (
        ("precision", "Test Precision by Class"),
        ("recall", "Test Recall by Class"),
    ):
        values = np.asarray(
            [
                [metric_value(row, f"test_{metric}_{name.lower()}_pct") for name in CLASS_NAMES]
                for row in rows
            ],
            dtype=np.float64,
        )
        fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
        image = draw_score_matrix(
            axis, values, row_labels, list(CLASS_NAMES), title, value_range
        )
        add_score_colorbar(fig, axis, image, value_range)
        fig.text(
            0.01,
            -0.01,
            "Color scale: 80–100%. Exact scores are printed in every cell.",
            fontsize=9,
            color="#4B5563",
        )
        save_figure(fig, output_dir / f"benchmark_class_{metric}")

    (output_dir / "benchmark_class_precision_recall.png").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("eval/results/benchmark_summary.csv"),
        help="Benchmark CSV produced by run_dense_linear_probe.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/benchmark_comparison"),
        help="Directory for exported figures",
    )
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = load_results(summary_path)

    missing_metrics = [
        metric
        for metric, _ in OVERALL_METRICS
        if any(metric_value(row, metric) is None for row in rows)
    ]
    if missing_metrics:
        parser.error(
            "Summary is missing required comparison metrics: "
            + ", ".join(missing_metrics)
            + ". Run the updated benchmark script to regenerate it."
        )

    for class_name in CLASS_NAMES:
        required = (
            f"test_iou_{class_name.lower()}_pct",
            f"test_precision_{class_name.lower()}_pct",
            f"test_recall_{class_name.lower()}_pct",
        )
        if any(any(metric_value(row, key) is None for row in rows) for key in required):
            parser.error(
                f"Summary is missing per-class metrics for {class_name}. "
                "Run the updated benchmark script to regenerate it."
            )

    plot_overall(rows, output_dir)
    plot_class_iou(rows, output_dir)
    plot_class_precision_and_recall(rows, output_dir)

    print(f"Loaded {len(rows)} models from {summary_path}")
    print("Ranked by test mIoU:")
    for rank, row in enumerate(rows, start=1):
        print(f"  {rank}. {model_label(row)}: {float(row['test_mIoU_pct']):.2f}%")
    print(f"Saved PNG figures to {output_dir}")


if __name__ == "__main__":
    main()
