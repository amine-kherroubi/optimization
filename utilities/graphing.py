from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .statistics import load_results
except ImportError:  # direct script execution fallback
    from statistics import load_results  # type: ignore[no-redef]


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


def _save_figure(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
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
) -> plt.Figure:
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
) -> plt.Figure:
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
) -> plt.Figure:
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


def _plot_scatter(
    x_values: Sequence[float],
    y_values: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, s=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _plot_heatmap(
    matrix: np.ndarray,
    x_tick_labels: Sequence[str],
    y_tick_labels: Sequence[str],
    title: str,
    colorbar_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * len(y_tick_labels))))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)
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


def _create_graphs_granular(csv_path: str | Path, out_dir: str | Path) -> list[Path]:
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


def _create_graphs_monolithic(csv_path: str | Path, out_dir: str | Path) -> list[Path]:
    rows = [r for r in load_results(csv_path) if not r.timed_out]
    if not rows:
        raise ValueError("No completed rows found in CSV.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_key = rows[0].dataset_key
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 5))
    times = [r.elapsed_time for r in rows]
    ax.bar(range(len(rows)), times, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [r.instance_name for r in rows], rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("Solve time (s)")
    ax.set_title(f"Solve Time per Instance — {dataset_key}")
    fig.tight_layout()
    p = out / f"fig1_solve_times_{dataset_key}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 5))
    x = range(len(rows))
    lb_vals = [r.lower_bound for r in rows]
    gaps = [r.bins_used - r.lower_bound for r in rows]
    ax.bar(x, lb_vals, label="Lower Bound", color="#2ecc71", edgecolor="white")
    ax.bar(x, gaps, bottom=lb_vals, label="Gap", color="#e74c3c", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [r.instance_name for r in rows], rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("Bins")
    ax.set_title(f"Bins Used vs Lower Bound — {dataset_key}")
    ax.legend()
    fig.tight_layout()
    p = out / f"fig2_bins_vs_lb_{dataset_key}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    by_size: dict[int, list[float]] = {}
    for r in rows:
        by_size.setdefault(r.num_items, []).append(r.elapsed_time)
    if len(by_size) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = sorted(by_size)
        data = [by_size[s] for s in sizes]
        ax.boxplot(data, tick_labels=[str(s) for s in sizes], patch_artist=True)
        rng = np.random.default_rng(seed=0)
        for i, times in enumerate(data):
            jitter = rng.uniform(-0.12, 0.12, size=len(times))
            ax.scatter(np.full(len(times), i + 1) + jitter, times, s=24, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("Number of items (n)")
        ax.set_ylabel("Solve time (s) [log scale]")
        ax.set_title(f"Solve Time Distribution by Instance Size — {dataset_key}")
        fig.tight_layout()
        p = out / f"fig3_time_by_size_{dataset_key}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def create_graphs(csv_path: str | Path, out_dir: str | Path) -> list[Path]:
    return _create_graphs_granular(csv_path, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create graphs from benchmark CSV output."
    )
    parser.add_argument("csv", help="Path to benchmark CSV file.")
    parser.add_argument(
        "--out-dir",
        default="results/graphs",
        help="Output directory for generated graphs.",
    )
    args = parser.parse_args()

    files = create_graphs(args.csv, args.out_dir)
    for path in files:
        print(path)
