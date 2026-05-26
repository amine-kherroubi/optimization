from __future__ import annotations

from pathlib import Path
from typing import Sequence
from datetime import datetime

import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display

# Project root used for sensible defaults
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

from bin_packing_optimization.utilities.statistics import (
    filter_rows_by_items,
    load_results,
)


def _load_completed_rows(
    csv_path: str | Path,
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
):
    rows = filter_rows_by_items(
        load_results(csv_path),
        num_items=num_items,
        min_items=min_items,
        max_items=max_items,
    )
    rows = [r for r in rows if not r.timed_out]
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


def _apply_common_style(ax: Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


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
    _apply_common_style(ax)
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
    _apply_common_style(ax)
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
    _apply_common_style(ax)
    fig.tight_layout()
    return fig


def _plot_boxplot_categories(
    grouped_data: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    yscale: str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(max(8, len(grouped_data) * 0.8), 5))
    categories = list(grouped_data.keys())
    data = [grouped_data[c] for c in categories]
    ax.boxplot(data, patch_artist=True)
    rng = np.random.default_rng(seed=0)
    for i, values in enumerate(data):
        if not values:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), i + 1) + jitter, values, s=24, zorder=3)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _apply_common_style(ax)
    fig.tight_layout()
    return fig


def _build_fill_rate_chart(rows, dataset_key: str, out_dir: Path) -> Path:
    labels = _instance_labels(rows)
    fill_rates = [r.total_weight / (r.bins_used * r.bin_capacity) * 100 for r in rows]
    fig = _plot_bar_by_instance(
        labels,
        fill_rates,
        "Fill rate (%)",
        f"Bin Fill Rate per Instance — {dataset_key}",
        color="#1abc9c",
    )
    return _save_figure(fig, out_dir, f"fig4_fill_rate_{dataset_key}.png")


def _build_gap_vs_time_scatter(rows, dataset_key: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    gaps = [r.bins_used - r.lower_bound for r in rows]
    times = [r.elapsed_time for r in rows]
    sizes = [max(25, r.num_items * 0.8) for r in rows]
    sc = ax.scatter(
        gaps, times, s=sizes, alpha=0.8, c=[r.num_items for r in rows], cmap="viridis"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Number of items (n)")
    ax.set_xlabel("Gap (bins_used - lower_bound)")
    ax.set_ylabel("Solve time (s)")
    ax.set_title(f"Gap vs Solve Time — {dataset_key}")
    _apply_common_style(ax)
    fig.tight_layout()
    return _save_figure(fig, out_dir, f"fig5_gap_vs_time_{dataset_key}.png")


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


def create_graphs(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> list[Path]:
    """Generate and save all benchmark graphs for the given CSV.

    Parameters
    ----------
    csv_path:
        Path to a benchmark results CSV produced by benchmarking.py.
    out_dir:
        Directory to write the PNG files into.  Defaults to a ``graphs/``
        subdirectory next to *csv_path*, so output is always co-located with
        the data it describes.
    num_items / min_items / max_items:
        Optional row filters by instance size. Use `num_items` for exact size,
        or `min_items`/`max_items` for an inclusive range.

    Returns
    -------
    list[Path]
        Paths of the files written.
    """
    csv_path = Path(csv_path)
    if out_dir is None:
        out_dir = csv_path.parent / "graphs"

    # Ensure we have a Path so the division operator works predictably.
    out_dir = Path(out_dir)

    rows = _load_completed_rows(
        csv_path,
        num_items=num_items,
        min_items=min_items,
        max_items=max_items,
    )
    # Place graphs alongside the CSV. The CSV is created in a timestamped
    # results folder by the benchmark runner, so adding another timestamp here
    # would be redundant.
    output = _prepare_output_dir(out_dir)
    dataset_key = rows[0].dataset_key

    paths: list[Path] = []
    paths.append(_build_solve_time_chart(rows, dataset_key, output))
    paths.append(_build_bins_vs_lb_chart(rows, dataset_key, output))
    maybe_path = _build_time_by_size_chart(rows, dataset_key, output)
    if maybe_path is not None:
        paths.append(maybe_path)
    paths.append(_build_fill_rate_chart(rows, dataset_key, output))
    paths.append(_build_gap_vs_time_scatter(rows, dataset_key, output))
    return paths


def create_comparison_graphs(
    csv_paths: Sequence[str | Path], out_dir: str | Path | None = None
) -> list[Path]:
    """Create comparative graphs across multiple benchmark CSV files.

    Produces: (1) boxplot of solve times per method and (2) average bins per
    method bar chart.
    """
    if not csv_paths:
        raise ValueError("At least one CSV path must be provided.")

    rows_by_file = {Path(p).stem: load_results(p) for p in csv_paths}

    # group rows by method across all files
    methods: dict[str, list] = {}
    for _k, rows in rows_by_file.items():
        for r in rows:
            methods.setdefault(r.method, []).append(r)

    # prepare aggregated metrics
    times_by_method: dict[str, list[float]] = {
        m: [r.elapsed_time for r in rs if not r.timed_out] for m, rs in methods.items()
    }
    bins_by_method: dict[str, list[int]] = {
        m: [r.bins_used for r in rs if not r.timed_out] for m, rs in methods.items()
    }

    if out_dir is None:
        out_dir = _PROJECT_ROOT / "results" / "comparisons"

    # Ensure we have a Path so the division operator works predictably.
    out_dir = Path(out_dir)

    output = _prepare_output_dir(out_dir / datetime.now().strftime("%Y%m%d_%H%M%S"))
    paths: list[Path] = []

    # boxplot of solve times per method (skip methods with no completed runs)
    times_nonempty = {m: v for m, v in times_by_method.items() if v}
    if times_nonempty:
        fig = _plot_boxplot_categories(
            times_nonempty,
            xlabel="Method",
            ylabel="Solve time (s) [log scale]",
            title="Solve Time Distribution by Method",
            yscale="log",
        )
        paths.append(_save_figure(fig, output, "compare_time_by_method.png"))

    # average bins per method
    avg_bins_labels = []
    avg_bins_vals = []
    for m, vals in bins_by_method.items():
        if vals:
            avg_bins_labels.append(m)
            avg_bins_vals.append(float(np.mean(vals)))
    if avg_bins_labels:
        fig = _plot_bar_by_instance(
            avg_bins_labels,
            avg_bins_vals,
            ylabel="Average bins used",
            title="Average Bins Used by Method",
            color="#9b59b6",
        )
        paths.append(_save_figure(fig, output, "compare_avg_bins_by_method.png"))

    return paths


def _dataset_colors(
    dataset_keys: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    n = len(dataset_keys)
    if n <= 20:
        cmap = plt.get_cmap("tab20", max(1, n))
        return {k: cmap(i) for i, k in enumerate(dataset_keys)}
    cmap = plt.get_cmap("hsv", max(1, n))
    return {k: cmap(i) for i, k in enumerate(dataset_keys)}


def create_multi_dataset_graphs(
    csv_paths: Sequence[str | Path],
    out_dir: str | Path | None = None,
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> list[Path]:
    """Create specialized cross-dataset graphs from many benchmark CSV files.

    The number of datasets is intentionally unbounded; colors are assigned
    dynamically from a colormap for robustness and simplicity.
    """
    if not csv_paths:
        raise ValueError("At least one CSV path must be provided.")

    rows_all = []
    for csv_path in csv_paths:
        rows = filter_rows_by_items(
            load_results(csv_path),
            num_items=num_items,
            min_items=min_items,
            max_items=max_items,
        )
        rows_all.extend(rows)
    completed = [r for r in rows_all if not r.timed_out]
    if not completed:
        raise ValueError("No completed rows found across provided CSV files.")

    if out_dir is None:
        out_dir = _PROJECT_ROOT / "results" / "multi_dataset_graphs"
    output = _prepare_output_dir(
        Path(out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    dataset_keys = sorted({r.dataset_key for r in completed})
    colors = _dataset_colors(dataset_keys)
    paths: list[Path] = []

    # 1) Solve-time distribution by dataset (boxplot + jitter, color-coded).
    fig, ax = plt.subplots(figsize=(max(8, len(dataset_keys) * 0.8), 5))
    data = [
        [r.elapsed_time for r in completed if r.dataset_key == k] for k in dataset_keys
    ]
    bp = ax.boxplot(data, patch_artist=True)
    for patch, key in zip(bp["boxes"], dataset_keys):
        patch.set_facecolor(colors[key])
        patch.set_alpha(0.5)
    rng = np.random.default_rng(seed=0)
    for i, key in enumerate(dataset_keys):
        vals = [r.elapsed_time for r in completed if r.dataset_key == key]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i + 1) + jitter, vals, s=20, color=colors[key], alpha=0.8
        )
    ax.set_xticks(range(1, len(dataset_keys) + 1))
    ax.set_xticklabels(dataset_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Solve time (s) [log scale]")
    ax.set_title("Solve Time Distribution by Dataset")
    _apply_common_style(ax)
    fig.tight_layout()
    paths.append(_save_figure(fig, output, "multi_dataset_time_distribution.png"))

    # 2) Average optimality gap by dataset.
    avg_gap = {
        key: float(
            np.mean(
                [r.bins_used - r.lower_bound for r in completed if r.dataset_key == key]
            )
        )
        for key in dataset_keys
    }
    fig, ax = plt.subplots(figsize=(max(8, len(dataset_keys) * 0.8), 5))
    ax.bar(
        range(len(dataset_keys)),
        [avg_gap[k] for k in dataset_keys],
        color=[colors[k] for k in dataset_keys],
    )
    ax.set_xticks(range(len(dataset_keys)))
    ax.set_xticklabels(dataset_keys, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Average gap (bins_used - lower_bound)")
    ax.set_title("Average Gap by Dataset")
    _apply_common_style(ax)
    fig.tight_layout()
    paths.append(_save_figure(fig, output, "multi_dataset_avg_gap.png"))

    # 3) Size-vs-time scatter colored by dataset.
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in dataset_keys:
        subset = [r for r in completed if r.dataset_key == key]
        ax.scatter(
            [r.num_items for r in subset],
            [r.elapsed_time for r in subset],
            alpha=0.75,
            s=24,
            color=colors[key],
            label=key,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of items (n)")
    ax.set_ylabel("Solve time (s) [log scale]")
    ax.set_title("Instance Size vs Solve Time by Dataset")
    ax.legend(ncol=2, fontsize=8)
    _apply_common_style(ax)
    fig.tight_layout()
    paths.append(_save_figure(fig, output, "multi_dataset_size_vs_time.png"))

    return paths


def _caption_from_graph_name(path: Path) -> str:
    if path.name.startswith("fig1_"):
        return "Fig 1 — Solve time per instance (seconds)"
    if path.name.startswith("fig2_"):
        return "Fig 2 — Bins used vs continuous lower bound"
    if path.name.startswith("fig3_"):
        return "Fig 3 — Solve time distribution by instance size"
    if path.name == "compare_time_by_method.png":
        return "Comparison — Solve time distribution by method"
    if path.name == "compare_avg_bins_by_method.png":
        return "Comparison — Average bins used by method"
    if path.name == "multi_dataset_time_distribution.png":
        return "Cross-dataset — Solve time distribution"
    if path.name == "multi_dataset_avg_gap.png":
        return "Cross-dataset — Average gap"
    if path.name == "multi_dataset_size_vs_time.png":
        return "Cross-dataset — Instance size vs solve time"
    return f"Figure — {path.stem}"


def display_graphs(csv_path: str | Path) -> list[Path]:
    """Create benchmark graphs and display each with an automatic caption."""
    graph_paths = create_graphs(csv_path)
    for path in graph_paths:
        print(_caption_from_graph_name(path))
        display(Image(filename=str(path)))
        print()
    return graph_paths


def display_comparison_graphs(
    csv_paths: Sequence[str | Path],
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Create comparison graphs and display each figure inline."""
    graph_paths = create_comparison_graphs(csv_paths, out_dir=out_dir)
    for path in graph_paths:
        print(_caption_from_graph_name(path))
        display(Image(filename=str(path)))
        print()
    return graph_paths


def display_multi_dataset_graphs(
    csv_paths: Sequence[str | Path],
    out_dir: str | Path | None = None,
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> list[Path]:
    """Create cross-dataset graphs and display each figure inline."""
    graph_paths = create_multi_dataset_graphs(
        csv_paths,
        out_dir=out_dir,
        num_items=num_items,
        min_items=min_items,
        max_items=max_items,
    )
    for path in graph_paths:
        print(_caption_from_graph_name(path))
        display(Image(filename=str(path)))
        print()
    return graph_paths
