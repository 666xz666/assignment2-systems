"""Compare eager and torch.compile attention benchmark CSV files.

Example:
    python plot_attention_comparison.py

The figure contains absolute forward/backward latency, memory difference,
forward/backward speedup, and a status boundary.  Speedup is defined as
``eager time / compiled time``; values greater than 1 mean compilation is
faster.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


METRICS = {
    "forward_ms": "Forward latency (ms)",
    "backward_ms": "Backward latency (ms)",
}


def read_results(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        results: dict[tuple[int, int], dict[str, Any]] = {}
        for row in csv.DictReader(input_file):
            key = (int(row["d_model"]), int(row["sequence_length"]))
            row["d_model"] = key[0]
            row["sequence_length"] = key[1]
            row["status"] = row["status"].upper()
            for field in set(METRICS) | {"memory_before_backward_mb"}:
                row[field] = float(row[field]) if row[field] else None
            results[key] = row
    if not results:
        raise ValueError(f"no benchmark rows found in {path}")
    return results


def all_dimensions(
    eager: dict[tuple[int, int], dict[str, Any]],
    compiled: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[int], list[int]]:
    keys = set(eager) | set(compiled)
    return sorted({key[0] for key in keys}), sorted({key[1] for key in keys})


def draw_latency(
    axis: Any,
    eager: dict[tuple[int, int], dict[str, Any]],
    compiled: dict[tuple[int, int], dict[str, Any]],
    d_models: list[int],
    sequence_lengths: list[int],
    metric: str,
) -> None:
    colors = plt.cm.tab10.colors
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("sequence length")
    axis.set_ylabel("milliseconds, log scale")
    axis.set_title(METRICS[metric])
    axis.grid(True, which="both", linestyle=":", alpha=0.35)

    plotted_values: list[float] = []
    for index, d_model in enumerate(d_models):
        color = colors[index % len(colors)]
        for label, data, linestyle, marker in (
            ("eager", eager, "--", "o"),
            ("compiled", compiled, "-", "s"),
        ):
            points = [
                (sequence_length, data[(d_model, sequence_length)][metric])
                for sequence_length in sequence_lengths
                if (d_model, sequence_length) in data
                and data[(d_model, sequence_length)]["status"] == "OK"
                and data[(d_model, sequence_length)][metric] is not None
            ]
            if not points:
                continue
            plotted_values.extend(value for _, value in points if value is not None)
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5,
                label=f"d={d_model} {label}",
            )

    oom_y = max(plotted_values) * 1.35 if plotted_values else 1.0
    for data, color, marker, label in (
        (eager, "#d62728", "X", "eager OOM"),
        (compiled, "#9467bd", "P", "compiled OOM"),
    ):
        points = [
            (row["sequence_length"], d_model)
            for (d_model, _), row in data.items()
            if row["status"] == "OOM"
        ]
        if points:
            axis.scatter(
                [point[0] for point in points],
                [oom_y] * len(points),
                color=color,
                marker=marker,
                s=65,
                label=label,
                zorder=5,
            )
    if plotted_values:
        axis.set_ylim(bottom=min(plotted_values) * 0.75, top=oom_y * 1.25)
    axis.set_xticks(sequence_lengths)
    axis.set_xticklabels([str(value) for value in sequence_lengths])
    axis.legend(fontsize=7, ncol=2)


def draw_speedup_heatmap(
    axis: Any,
    eager: dict[tuple[int, int], dict[str, Any]],
    compiled: dict[tuple[int, int], dict[str, Any]],
    d_models: list[int],
    sequence_lengths: list[int],
    metric: str,
) -> None:
    speedups: list[float] = []
    grid: list[list[float | None]] = []
    for d_model in d_models:
        line: list[float | None] = []
        for sequence_length in sequence_lengths:
            eager_row = eager.get((d_model, sequence_length))
            compiled_row = compiled.get((d_model, sequence_length))
            if (
                eager_row is not None
                and compiled_row is not None
                and eager_row["status"] == "OK"
                and compiled_row["status"] == "OK"
                and eager_row[metric] is not None
                and compiled_row[metric] is not None
                and compiled_row[metric] > 0
            ):
                value = eager_row[metric] / compiled_row[metric]
                line.append(value)
                speedups.append(value)
            else:
                line.append(None)
        grid.append(line)

    image_values = [
        [value if value is not None else 1.0 for value in line] for line in grid
    ]
    if speedups:
        vmin = min(min(speedups), 1.0)
        vmax = max(max(speedups), 1.0)
        if vmin >= 1.0:
            vmin = min(vmin * 0.9, 0.99)
        if vmax <= 1.0:
            vmax = max(vmax * 1.1, 1.01)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=0.9, vcenter=1.0, vmax=1.1)

    image = axis.imshow(
        image_values,
        aspect="auto",
        cmap="RdYlGn",
        norm=norm,
    )
    axis.figure.colorbar(image, ax=axis, shrink=0.85, label="eager / compiled")
    axis.set_xticks(
        range(len(sequence_lengths)), [str(value) for value in sequence_lengths]
    )
    axis.set_yticks(range(len(d_models)), [str(value) for value in d_models])
    axis.set_xlabel("sequence length")
    axis.set_ylabel("d_model")
    axis.set_title(f"{METRICS[metric].split()[0]} speedup (>1 is faster)")

    for row_index, d_model in enumerate(d_models):
        for column_index, sequence_length in enumerate(sequence_lengths):
            value = grid[row_index][column_index]
            if value is None:
                text = "OOM"
                color = "black"
            else:
                text = f"{value:.2f}x"
                color = (
                    "white"
                    if value > (max(speedups) * 0.55 if speedups else 1)
                    else "black"
                )
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if value is None else "normal",
                color=color,
            )


def draw_memory_delta(
    axis: Any,
    eager: dict[tuple[int, int], dict[str, Any]],
    compiled: dict[tuple[int, int], dict[str, Any]],
    d_models: list[int],
    sequence_lengths: list[int],
) -> None:
    grid: list[list[float | None]] = []
    deltas: list[float] = []
    for d_model in d_models:
        line: list[float | None] = []
        for sequence_length in sequence_lengths:
            eager_row = eager.get((d_model, sequence_length))
            compiled_row = compiled.get((d_model, sequence_length))
            if (
                eager_row is not None
                and compiled_row is not None
                and eager_row["memory_before_backward_mb"] is not None
                and compiled_row["memory_before_backward_mb"] is not None
            ):
                value = (
                    compiled_row["memory_before_backward_mb"]
                    - eager_row["memory_before_backward_mb"]
                )
                line.append(value)
                deltas.append(value)
            else:
                line.append(None)
        grid.append(line)

    image_values = [
        [value if value is not None else 0.0 for value in line] for line in grid
    ]
    limit = max(max((abs(value) for value in deltas), default=1.0), 1.0)
    image = axis.imshow(
        image_values,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.figure.colorbar(image, ax=axis, shrink=0.85, label="MiB")
    axis.set_xticks(
        range(len(sequence_lengths)), [str(value) for value in sequence_lengths]
    )
    axis.set_yticks(range(len(d_models)), [str(value) for value in d_models])
    axis.set_xlabel("sequence length")
    axis.set_ylabel("d_model")
    axis.set_title("Memory delta: compiled - eager (MiB)")
    for row_index, line in enumerate(grid):
        for column_index, value in enumerate(line):
            axis.text(
                column_index,
                row_index,
                "OOM" if value is None else f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if value is None else "normal",
            )


def draw_status_boundary(
    axis: Any,
    eager: dict[tuple[int, int], dict[str, Any]],
    compiled: dict[tuple[int, int], dict[str, Any]],
    d_models: list[int],
    sequence_lengths: list[int],
) -> None:
    colors = {
        "OK/OK": "#2ca02c",
        "OOM/OOM": "#d62728",
        "OK/OOM": "#ff7f0e",
        "OOM/OK": "#1f77b4",
    }
    axis.set_facecolor("#eeeeee")
    for row_index, d_model in enumerate(d_models):
        for column_index, sequence_length in enumerate(sequence_lengths):
            eager_status = eager.get((d_model, sequence_length), {}).get("status", "-")
            compiled_status = compiled.get((d_model, sequence_length), {}).get(
                "status", "-"
            )
            status = f"{eager_status}/{compiled_status}"
            color = colors.get(status, "#aaaaaa")
            axis.scatter(
                column_index,
                row_index,
                s=900,
                marker="s",
                color=color,
                edgecolors="black",
            )
            axis.text(
                column_index,
                row_index,
                status,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
    axis.set_xticks(
        range(len(sequence_lengths)), [str(value) for value in sequence_lengths]
    )
    axis.set_yticks(range(len(d_models)), [str(value) for value in d_models])
    axis.set_xlabel("sequence length")
    axis.set_ylabel("d_model")
    axis.set_title("OOM boundary: eager / compiled")
    axis.grid(False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--eager",
        "--baseline",
        dest="eager",
        type=Path,
        default=Path("output/attention/attention_benchmark.csv"),
        help="CSV from the uncompiled benchmark",
    )
    parser.add_argument(
        "--compiled",
        type=Path,
        default=Path(
            "output/attention/"
            "attention_benchmark_compiled-inductor-default_float32_b8_w5_n100_"
            "dm16-32-64-128_s256-1024-4096-8192-16384.csv"
        ),
        help="CSV from the torch.compile benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/attention/attention_eager_vs_compiled.png"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.eager.exists():
        raise FileNotFoundError(f"eager CSV does not exist: {args.eager}")
    if not args.compiled.exists():
        raise FileNotFoundError(f"compiled CSV does not exist: {args.compiled}")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")

    eager = read_results(args.eager)
    compiled = read_results(args.compiled)
    d_models, sequence_lengths = all_dimensions(eager, compiled)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    draw_latency(axes[0, 0], eager, compiled, d_models, sequence_lengths, "forward_ms")
    draw_latency(axes[0, 1], eager, compiled, d_models, sequence_lengths, "backward_ms")
    draw_memory_delta(axes[0, 2], eager, compiled, d_models, sequence_lengths)
    draw_speedup_heatmap(
        axes[1, 0], eager, compiled, d_models, sequence_lengths, "forward_ms"
    )
    draw_speedup_heatmap(
        axes[1, 1], eager, compiled, d_models, sequence_lengths, "backward_ms"
    )
    draw_status_boundary(axes[1, 2], eager, compiled, d_models, sequence_lengths)

    figure.suptitle(
        "Eager attention vs torch.compile attention\n"
        "Speedup = eager time / compiled time; values above 1 are faster",
        fontsize=15,
    )
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    common_success = [
        key
        for key in set(eager) & set(compiled)
        if eager[key]["status"] == "OK" and compiled[key]["status"] == "OK"
    ]
    if common_success:
        forward_speedups = [
            eager[key]["forward_ms"] / compiled[key]["forward_ms"]
            for key in common_success
        ]
        backward_speedups = [
            eager[key]["backward_ms"] / compiled[key]["backward_ms"]
            for key in common_success
        ]
        print(f"Saved comparison plot to {args.output}")
        print(
            f"Forward speedup range: {min(forward_speedups):.2f}x - {max(forward_speedups):.2f}x"
        )
        print(
            f"Backward speedup range: {min(backward_speedups):.2f}x - {max(backward_speedups):.2f}x"
        )
    else:
        print(
            f"Saved comparison plot to {args.output}; no common successful configurations"
        )


if __name__ == "__main__":
    main()
