from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .statistics import load_results
except ImportError:  # direct script execution fallback
    from statistics import load_results  # type: ignore[no-redef]


def create_graphs(csv_path: str | Path, out_dir: str | Path) -> list[Path]:
    rows = [r for r in load_results(csv_path) if not r.timed_out]
    if not rows:
        raise ValueError("No completed rows found in CSV.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_key = rows[0].dataset_key
    paths: list[Path] = []

    # Solve times
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

    # Bins vs lower bound
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

    # time by size groups
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
