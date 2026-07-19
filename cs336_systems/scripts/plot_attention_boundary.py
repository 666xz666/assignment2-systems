"""Plot timing, memory, and OOM boundaries from attention benchmark CSV output.

Example:
    python plot_attention_boundary.py \
        --input output/attention/attention_benchmark.csv \
        --output output/attention/attention_boundary.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def read_results(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(input_file):
            row["d_model"] = int(row["d_model"])
            row["sequence_length"] = int(row["sequence_length"])
            row["status"] = row["status"].upper()
            for key in (
                "forward_ms",
                "backward_ms",
                "memory_before_backward_mb",
            ):
                row[key] = float(row[key]) if row[key] else None
            rows.append(row)
    if not rows:
        raise ValueError(f"no benchmark rows found in {path}")
    return rows


def make_grid(
    rows: list[dict[str, Any]], metric: str
) -> tuple[list[int], list[int], list[list[float | None]]]:
    d_models = sorted({row["d_model"] for row in rows})
    sequence_lengths = sorted({row["sequence_length"] for row in rows})
    values = {
        (row["d_model"], row["sequence_length"]): row[metric] for row in rows
    }
    grid = [
        [values.get((d_model, sequence_length)) for sequence_length in sequence_lengths]
        for d_model in d_models
    ]
    return d_models, sequence_lengths, grid


def draw_heatmap(
    axis: Any,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    color_map: str,
) -> None:
    d_models, sequence_lengths, grid = make_grid(rows, metric)
    numeric_values = [value for line in grid for value in line if value is not None]
    if not numeric_values:
        axis.set_title(f"{title} (no successful configurations)")
        return

    # Use a logarithmic scale because the benchmark spans milliseconds to
    # seconds and small configurations would otherwise be hard to compare.
    image_values = [
        [value if value is not None else min(numeric_values) for value in line]
        for line in grid
    ]
    image = axis.imshow(
        image_values,
        aspect="auto",
        cmap=color_map,
        norm=LogNorm(vmin=min(numeric_values), vmax=max(numeric_values)),
    )
    axis.figure.colorbar(image, ax=axis, shrink=0.85, label=metric.replace("_", " "))

    axis.set_xticks(range(len(sequence_lengths)), [str(value) for value in sequence_lengths])
    axis.set_yticks(range(len(d_models)), [str(value) for value in d_models])
    axis.set_xlabel("sequence length")
    axis.set_ylabel("d_model")
    axis.set_title(title)

    for row_index, d_model in enumerate(d_models):
        for column_index, sequence_length in enumerate(sequence_lengths):
            row = next(
                (
                    item
                    for item in rows
                    if item["d_model"] == d_model
                    and item["sequence_length"] == sequence_length
                ),
                None,
            )
            if row is None:
                continue
            if row["status"] == "OOM":
                axis.text(
                    column_index,
                    row_index,
                    "OOM",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            elif row[metric] is not None:
                axis.text(
                    column_index,
                    row_index,
                    f"{row[metric]:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )


def draw_boundary(axis: Any, rows: list[dict[str, Any]]) -> None:
    d_models = sorted({row["d_model"] for row in rows})
    successful = [row for row in rows if row["status"] == "OK"]
    oom = [row for row in rows if row["status"] == "OOM"]

    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("sequence length")
    axis.set_ylabel("d_model")
    axis.set_title("Attention performance boundary")

    if successful:
        axis.scatter(
            [row["sequence_length"] for row in successful],
            [row["d_model"] for row in successful],
            s=90,
            color="#2ca02c",
            edgecolors="black",
            label="successful",
            zorder=3,
        )
    if oom:
        axis.scatter(
            [row["sequence_length"] for row in oom],
            [row["d_model"] for row in oom],
            s=110,
            marker="X",
            color="#d62728",
            edgecolors="black",
            label="OOM",
            zorder=4,
        )

    # Draw the largest successful sequence length for each d_model. This is
    # the practical boundary users normally need from this experiment.
    for d_model in d_models:
        successful_lengths = [
            row["sequence_length"]
            for row in successful
            if row["d_model"] == d_model
        ]
        if successful_lengths:
            boundary = max(successful_lengths)
            axis.plot(
                [boundary, boundary],
                [d_model / 1.35, d_model * 1.35],
                color="#2ca02c",
                alpha=0.35,
                linewidth=2,
            )
            axis.annotate(
                f"max S={boundary}",
                (boundary, d_model),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_xticks(sorted({row["sequence_length"] for row in rows}))
    axis.set_yticks(d_models)
    axis.get_xaxis().set_major_formatter(lambda value, _: f"{int(value)}")
    axis.get_yaxis().set_major_formatter(lambda value, _: f"{int(value)}")
    axis.grid(True, which="both", linestyle=":", alpha=0.35)
    axis.legend(loc="best")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/attention/attention_benchmark.csv"),
        help="CSV file generated by benchmark_attention.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/attention/attention_boundary.png"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"benchmark CSV does not exist: {args.input}")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")

    rows = read_results(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    draw_heatmap(axes[0, 0], rows, "forward_ms", "Forward latency (ms)", "viridis")
    draw_heatmap(axes[0, 1], rows, "backward_ms", "Backward latency (ms)", "plasma")
    draw_heatmap(
        axes[1, 0],
        rows,
        "memory_before_backward_mb",
        "Memory before backward (MiB)",
        "magma",
    )
    draw_boundary(axes[1, 1], rows)

    figure.suptitle(
        "Naive scaled dot-product attention benchmark\n"
        f"{args.input} | green=success, red=OOM",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
