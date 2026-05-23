from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load the local statistics module by file path. A plain
# `from statistics import load_results` would silently resolve to Python's
# stdlib statistics module (which has no load_results), causing an ImportError
# at runtime when this file is executed as a script. The relative import works
# when used as a package but fails as a script, hence the two-path approach.
try:
    from .statistics import load_results  # package import
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "_utilities_statistics",
        Path(__file__).with_name("statistics.py"),
    )
    assert _spec is not None and _spec.loader is not None
    _stats_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_stats_mod)  # type: ignore[union-attr]
    load_results = _stats_mod.load_results  # type: ignore[attr-defined]


def _load_completed_rows(csv_path: str | Path):
    rows = [r for r in load_results(csv_path) if not r.timed_out]
    if not rows:
        raise ValueError("No completed rows found in CSV.")
    return rows


def _prepare_output_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _instance_labels(rows) -> list[str]:
    return [r.instance_name for r in rows]


def _figure_width(num_rows: int) -> float:
    return max(10, num_rows * 0.6)


def _save_figure(fig: Figure, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_bar_by_instance(
    labels: Sequence[str],
    values: Sequence[float],
    ylabel: str,
    title: str,
    color: str = "#3498db",
) -> Figure:
    fig, ax = plt.subplots(figsize=(_figure_width(len(labels)), 5))
    ax.bar(range(len(labels)), values, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _plot_stacked_bar_by_instance(
    labels: Sequence[str],
    base_values: Sequence[float],
    top_values: Sequence[float],
    ylabel: str,
    title: str,
    base_label: str,
    top_label: str,
    base_color: str = "#2ecc71",
    top_color: str = "#e74c3c",
) -> Figure:
    fig, ax = plt.subplots(figsize=(_figure_width(len(labels)), 5))
    x = range(len(labels))
    ax.bar(x, base_values, label=base_label, color=base_color, edgecolor="white")
    ax.bar(
        x,
        top_values,
        bottom=base_values,
        label=top_label,
        color=top_color,
        edgecolor="white",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_box_with_jitter(
    grouped_data: dict[int, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    yscale: str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = sorted(grouped_data)
    data = [grouped_data[c] for c in categories]
    ax.boxplot(data, tick_labels=[str(c) for c in categories], patch_artist=True)
    rng = np.random.default_rng(seed=0)
    for i, values in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), i + 1) + jitter, values, s=24, zorder=3)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _group_times_by_size(rows) -> dict[int, list[float]]:
    by_size: dict[int, list[float]] = {}
    for r in rows:
        by_size.setdefault(r.num_items, []).append(r.elapsed_time)
    return by_size


def _build_solve_time_chart(rows, dataset_key: str, out_dir: Path) -> Path:
    labels = _instance_labels(rows)
    times = [r.elapsed_time for r in rows]
    fig = _plot_bar_by_instance(
        labels, times, "Solve time (s)", f"Solve Time per Instance — {dataset_key}"
    )
    return _save_figure(fig, out_dir, f"fig1_solve_times_{dataset_key}.png")


def _build_bins_vs_lb_chart(rows, dataset_key: str, out_dir: Path) -> Path:
    labels = _instance_labels(rows)
    lb_vals = [r.lower_bound for r in rows]
    gaps = [r.bins_used - r.lower_bound for r in rows]
    fig = _plot_stacked_bar_by_instance(
        labels,
        lb_vals,
        gaps,
        "Bins",
        f"Bins Used vs Lower Bound — {dataset_key}",
        base_label="Lower Bound",
        top_label="Gap",
    )
    return _save_figure(fig, out_dir, f"fig2_bins_vs_lb_{dataset_key}.png")


def _build_time_by_size_chart(rows, dataset_key: str, out_dir: Path) -> Path | None:
    by_size = _group_times_by_size(rows)
    if len(by_size) <= 1:
        return None

    fig = _plot_box_with_jitter(
        grouped_data=by_size,
        xlabel="Number of items (n)",
        ylabel="Solve time (s) [log scale]",
        title=f"Solve Time Distribution by Instance Size — {dataset_key}",
        yscale="log",
    )
    return _save_figure(fig, out_dir, f"fig3_time_by_size_{dataset_key}.png")


def create_graphs(csv_path: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Generate and save all benchmark graphs for the given CSV.

    Parameters
    ----------
    csv_path:
        Path to a benchmark results CSV produced by benchmarking.py.
    out_dir:
        Directory to write the PNG files into.  Defaults to a ``graphs/``
        subdirectory next to *csv_path*, so output is always co-located with
        the data it describes.

    Returns
    -------
    list[Path]
        Paths of the files written.
    """
    csv_path = Path(csv_path)
    if out_dir is None:
        out_dir = csv_path.parent / "graphs"

    rows = _load_completed_rows(csv_path)
    output = _prepare_output_dir(out_dir)
    dataset_key = rows[0].dataset_key

    paths: list[Path] = []
    paths.append(_build_solve_time_chart(rows, dataset_key, output))
    paths.append(_build_bins_vs_lb_chart(rows, dataset_key, output))
    maybe_path = _build_time_by_size_chart(rows, dataset_key, output)
    if maybe_path is not None:
        paths.append(maybe_path)
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create graphs from benchmark CSV output."
    )
    parser.add_argument("csv", help="Path to benchmark CSV file.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for generated graphs. "
            "Defaults to a graphs/ subdirectory next to the CSV file."
        ),
    )
    args = parser.parse_args()

    files = create_graphs(args.csv, args.out_dir)
    for path in files:
        print(path)
