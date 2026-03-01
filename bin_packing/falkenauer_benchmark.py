from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# from bin_packing.slow import BinPacking, Solution
from bin_packing.fast import BinPacking, Solution

BENCHMARKS_ROOT: Path = Path(__file__).parent.parent / "benchmarks/Falkenauer"
FALKENAUER_T_DIR: Path = BENCHMARKS_ROOT / "Falkenauer_T"
FALKENAUER_U_DIR: Path = BENCHMARKS_ROOT / "Falkenauer U"

GRAPHS_DIR: Path = Path(__file__).parent / "results"


class FalkenauerVariant(StrEnum):
    """Falkenauer dataset variant.

    T (Triplets): items are large; optimal solutions pack exactly three items per bin.
    U (Uniform): items drawn uniformly at random.
    Both variants use sizes in [20, 100] and bin capacity 150.
    """

    T = "T"
    U = "U"

    @property
    def directory(self) -> Path:
        """Resolve the dataset directory for this variant."""
        return {
            FalkenauerVariant.T: FALKENAUER_T_DIR,
            FalkenauerVariant.U: FALKENAUER_U_DIR,
        }[self]


@dataclass(slots=True)
class BenchmarkResult(object):
    instance_name: str  # Name of the instance file (without extension)
    variant: FalkenauerVariant  # Dataset variant (T or U)
    num_items: int  # Number of items in the instance
    bin_capacity: int  # Capacity of each bin
    bins_used: int  # Number of bins used in the solution
    lower_bound: int  # Theoretical lower bound: ceil(sum(sizes) / capacity)
    total_weight: int  # Sum of all item sizes
    elapsed_time: float  # Wall-clock solve time in seconds
    method: str  # Solving method used


@dataclass(slots=True)
class FalkenauerInstance(object):
    """Parsed Falkenauer instance.

    File format:
        Line 1: number of items
        Line 2: bin capacity
        Lines 3+: one item size per line
    """

    name: str
    variant: FalkenauerVariant
    num_items: int
    bin_capacity: int
    sizes: list[int]

    @classmethod
    def from_file(
        cls, filepath: Path, variant: FalkenauerVariant
    ) -> FalkenauerInstance:
        """Parse an instance file and return a FalkenauerInstance."""
        lines: list[str] = filepath.read_text().splitlines()
        num_items: int = int(lines[0])
        bin_capacity: int = int(lines[1])
        sizes: list[int] = [int(lines[i]) for i in range(2, 2 + num_items)]
        return cls(
            name=filepath.stem,
            variant=variant,
            num_items=num_items,
            bin_capacity=bin_capacity,
            sizes=sizes,
        )


class FalkenauerBenchmark(object):
    """Benchmarks BinPacking instances from the Falkenauer dataset."""

    def __init__(self, variant: FalkenauerVariant = FalkenauerVariant.T) -> None:
        self._variant: FalkenauerVariant = variant
        self._instances_dir: Path = variant.directory
        self._results: list[BenchmarkResult] = []

    def run(
        self,
        method: str = "b&b",
        num_items: int | None = None,
        max_items: int | None = None,
        generate_graphs: bool = True,
    ) -> None:
        """Run the benchmark on a filtered subset of instances.

        num_items: run only instances with exactly this many items.
        max_items: run only instances with at most this many items.
        generate_graphs: if True, produce summary graphs after the run.
        Omit both item filters to run all instances.
        """
        if method != "b&b":
            raise ValueError("This solver version only supports method='b&b'.")
        if num_items is not None and max_items is not None:
            raise ValueError("num_items and max_items are mutually exclusive.")

        instances: list[FalkenauerInstance] = self._load_instances(num_items, max_items)
        if not instances:
            qualifier: str = ""
            if num_items is not None:
                qualifier = f" with exactly {num_items} items"
            elif max_items is not None:
                qualifier = f" with at most {max_items} items"
            raise FileNotFoundError(
                f"No {self._variant}-variant instance files{qualifier} found in '{self._instances_dir}'."
            )

        name_width: int = max(
            max(len(inst.name) for inst in instances), len("Instance")
        )

        self._results.clear()
        self._print_header(name_width)

        for instance in instances:
            result: BenchmarkResult = self._solve(instance, method)
            self._results.append(result)
            self._print_row(result, name_width)

        self._print_footer(name_width)

        if generate_graphs:
            self._generate_graphs()

    def run_instance(
        self, filepath: str | Path, method: str = "b&b"
    ) -> BenchmarkResult:
        """Solve and time a single instance file, appending its result."""
        if method != "b&b":
            raise ValueError("This solver version only supports method='b&b'.")

        instance: FalkenauerInstance = FalkenauerInstance.from_file(
            Path(filepath), self._variant
        )
        result: BenchmarkResult = self._solve(instance, method)
        self._results.append(result)
        return result

    def available_sizes(self) -> list[int]:
        """Return the sorted list of distinct item counts found in the instance directory."""
        sizes: set[int] = set()
        for filepath in self._instances_dir.glob("*.txt"):
            try:
                inst: FalkenauerInstance = FalkenauerInstance.from_file(
                    filepath, self._variant
                )
                sizes.add(inst.num_items)
            except (ValueError, IndexError):
                continue
        return sorted(sizes)

    def get_results(self) -> list[BenchmarkResult]:
        """Return a copy of all collected benchmark results."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()

    def print_summary(self) -> None:
        """Print a formatted summary table of all collected results."""
        if not self._results:
            print("No results available. Run the benchmark first.")
            return

        name_width: int = max(
            max(len(r.instance_name) for r in self._results), len("Instance")
        )
        self._print_header(name_width)
        for result in self._results:
            self._print_row(result, name_width)
        self._print_footer(name_width)

    def _load_instances(
        self, num_items: int | None, max_items: int | None
    ) -> list[FalkenauerInstance]:
        """Load instance files from the variant directory, optionally filtered by item count.

        Instances are sorted by (num_items, name) rather than lexicographically so that
        60-item instances always precede 120-item ones (avoiding '6' > '1' ordering issues).
        """
        instances: list[FalkenauerInstance] = []
        for filepath in self._instances_dir.glob("*.txt"):
            try:
                inst: FalkenauerInstance = FalkenauerInstance.from_file(
                    filepath, self._variant
                )
            except (ValueError, IndexError):
                continue
            if num_items is not None and inst.num_items != num_items:
                continue
            if max_items is not None and inst.num_items > max_items:
                continue
            instances.append(inst)

        # Sort by (num_items, name) so groups stay together in natural numeric order
        instances.sort(key=lambda inst: (inst.num_items, inst.name))
        return instances

    def _solve(self, instance: FalkenauerInstance, method: str) -> BenchmarkResult:
        """Solve a parsed instance and return a timed result."""
        solver: BinPacking = BinPacking(instance.sizes, instance.bin_capacity)

        start: float = time.perf_counter()
        solver.solve()
        elapsed: float = time.perf_counter() - start

        solution: Solution = solver.get_solution()
        lower_bound: int = ceil(sum(instance.sizes) / instance.bin_capacity)

        return BenchmarkResult(
            instance_name=instance.name,
            variant=instance.variant,
            num_items=instance.num_items,
            bin_capacity=instance.bin_capacity,
            bins_used=solution.bins_used,
            lower_bound=lower_bound,
            total_weight=sum(instance.sizes),
            elapsed_time=elapsed,
            method=method,
        )

    def _generate_graphs(self) -> None:
        """Generate and save four summary graphs from the collected results."""
        if not self._results:
            print("No results to plot.")
            return

        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update(
            {
                "figure.facecolor": "#FAFAFA",
                "axes.facecolor": "#F5F5F5",
                "axes.grid": True,
                "grid.color": "white",
                "grid.linewidth": 1.2,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 11,
            }
        )

        self._plot_time_vs_n()
        self._plot_bins_vs_lb()
        self._plot_fill_rate()
        self._plot_time_by_size_group()

    def _plot_time_vs_n(self) -> None:
        """Fig 1 — Scatter plot of solve time versus number of items.

        Each point is one instance, colored by elapsed time. The three slowest
        instances are annotated. A dashed trend line connects points in order of n.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ns: list[int] = [r.num_items for r in self._results]
        times: list[float] = [r.elapsed_time for r in self._results]

        sc = ax.scatter(
            ns,
            times,
            c=times,
            cmap="plasma",
            s=90,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

        # Dashed trend line through points sorted by n
        order: list[int] = list(np.argsort(ns))
        ax.plot(
            [ns[i] for i in order],
            [times[i] for i in order],
            color="steelblue",
            linewidth=1.2,
            alpha=0.45,
            linestyle="--",
        )

        # Annotate the three slowest instances
        for result in sorted(self._results, key=lambda r: r.elapsed_time, reverse=True)[
            :3
        ]:
            ax.annotate(
                result.instance_name,
                xy=(result.num_items, result.elapsed_time),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

        plt.colorbar(sc, ax=ax, label="Time (s)", pad=0.02)
        ax.set_yscale("log")
        ax.set_xlabel("Number of items (n)", fontsize=12)
        ax.set_ylabel("Solve time (s)  [log scale]", fontsize=12)
        ax.set_title(
            f"B&B Solve Time vs Number of Items  —  Variant {self._variant}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        fig.tight_layout()

        path: Path = GRAPHS_DIR / f"fig1_time_vs_n_{self._variant}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Fig 1 → {path}")

    def _plot_bins_vs_lb(self) -> None:
        """Fig 2 — Stacked bar chart comparing bins used against the theoretical lower bound.

        The blue portion represents the lower bound LB = ceil(sum(sizes) / capacity).
        The red portion on top represents the gap (bins_used - LB). A gap of zero means
        the solution is provably optimal.
        """
        fig, ax = plt.subplots(figsize=(max(10, len(self._results) * 0.6), 5))

        x: np.ndarray = np.arange(len(self._results))
        lbs: np.ndarray = np.array([r.lower_bound for r in self._results])
        gaps: np.ndarray = np.array(
            [r.bins_used - r.lower_bound for r in self._results]
        )

        ax.bar(x, lbs, color="#4C72B0", alpha=0.85, label="Lower bound LB = ⌈Σw / C⌉")
        ax.bar(
            x,
            gaps,
            bottom=lbs,
            color="#E74C3C",
            alpha=0.75,
            label="Gap = bins_used − LB",
        )

        for i, result in enumerate(self._results):
            ax.text(
                i,
                result.bins_used + 0.05,
                str(result.bins_used),
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.instance_name for r in self._results],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Number of bins", fontsize=12)
        ax.set_title(
            f"Bins Used vs Theoretical Lower Bound  —  Variant {self._variant}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

        path: Path = GRAPHS_DIR / f"fig2_bins_vs_lb_{self._variant}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Fig 2 → {path}")

    def _plot_fill_rate(self) -> None:
        """Fig 3 — Bar chart of average bin fill rate per instance.

        Fill rate = sum(sizes) / (bins_used * capacity) * 100.
        Bars are colored on a red-yellow-green scale; an orange threshold at 80 % is drawn.
        """
        fig, ax = plt.subplots(figsize=(max(10, len(self._results) * 0.6), 5))

        fill_rates: list[float] = [
            r.total_weight / (r.bins_used * r.bin_capacity) * 100 for r in self._results
        ]

        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(min(fill_rates), 100)
        colors: list = [cmap(norm(v)) for v in fill_rates]

        ax.bar(
            np.arange(len(self._results)), fill_rates, color=colors, edgecolor="white"
        )
        ax.axhline(100, color="black", linewidth=1.2, linestyle="--", label="100 %")
        ax.axhline(
            80, color="orange", linewidth=1.0, linestyle=":", label="Threshold 80 %"
        )

        for i, rate in enumerate(fill_rates):
            ax.text(i, rate + 0.4, f"{rate:.1f}%", ha="center", va="bottom", fontsize=7)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Fill rate (%)", pad=0.02)

        ax.set_xticks(np.arange(len(self._results)))
        ax.set_xticklabels(
            [r.instance_name for r in self._results],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_ylim(0, 115)
        ax.set_ylabel("Average bin fill rate (%)", fontsize=12)
        ax.set_title(
            f"Solution Quality: Average Bin Fill Rate  —  Variant {self._variant}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

        path: Path = GRAPHS_DIR / f"fig3_fill_rate_{self._variant}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Fig 3 → {path}")

    def _plot_time_by_size_group(self) -> None:
        """Fig 4 — Box plot of solve times grouped by instance size (number of items).

        Each box summarises the distribution of elapsed times across all instances
        sharing the same n, making it easy to compare scaling behaviour across sizes.
        Individual data points are overlaid as a strip plot for transparency.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        # Group results by number of items
        size_groups: dict[int, list[float]] = {}
        for result in self._results:
            size_groups.setdefault(result.num_items, []).append(result.elapsed_time)

        sorted_sizes: list[int] = sorted(size_groups.keys())
        group_times: list[list[float]] = [size_groups[n] for n in sorted_sizes]
        labels: list[str] = [str(n) for n in sorted_sizes]

        box = ax.boxplot(
            group_times,
            tick_labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        # Color each box by group using the viridis palette
        palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(sorted_sizes)))
        for patch, color in zip(box["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay individual points with horizontal jitter for readability
        rng = np.random.default_rng(seed=0)
        for group_index, times in enumerate(group_times):
            jitter: np.ndarray = rng.uniform(-0.12, 0.12, size=len(times))
            ax.scatter(
                np.full(len(times), group_index + 1) + jitter,
                times,
                s=28,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
                color=palette[group_index],
            )

        ax.set_yscale("log")
        ax.set_xlabel("Number of items (n)", fontsize=12)
        ax.set_ylabel("Solve time (s)  [log scale]", fontsize=12)
        ax.set_title(
            f"Solve Time Distribution by Instance Size  —  Variant {self._variant}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        fig.tight_layout()

        path: Path = GRAPHS_DIR / f"fig4_time_by_size_{self._variant}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Fig 4 → {path}")

    def _header_and_separator(self, name_width: int) -> tuple[str, str]:
        """Build the table header string and a matching separator line."""
        header: str = (
            f"{'Instance':<{name_width}}  "
            f"{'Items':>5}  "
            f"{'Capacity':>8}  "
            f"{'LB':>4}  "
            f"{'Bins':>4}  "
            f"{'Gap':>4}  "
            f"{'Time (s)':>10}  "
            f"{'Method'}"
        )
        return header, "-" * len(header)

    def _print_header(self, name_width: int) -> None:
        """Print the table header with surrounding separators."""
        header, separator = self._header_and_separator(name_width)
        print(separator)
        print(header)
        print(separator)

    def _print_row(self, result: BenchmarkResult, name_width: int) -> None:
        """Print a single result row."""
        gap: int = result.bins_used - result.lower_bound
        print(
            f"{result.instance_name:<{name_width}}  "
            f"{result.num_items:>5}  "
            f"{result.bin_capacity:>8}  "
            f"{result.lower_bound:>4}  "
            f"{result.bins_used:>4}  "
            f"{gap:>4}  "
            f"{result.elapsed_time:>10.4f}  "
            f"{result.method}"
        )

    def _print_footer(self, name_width: int) -> None:
        """Print the closing separator and aggregate statistics."""
        _, separator = self._header_and_separator(name_width)
        print(separator)
        total_time: float = sum(r.elapsed_time for r in self._results)
        avg_bins: float = sum(r.bins_used for r in self._results) / len(self._results)
        optimal_count: int = sum(
            1 for r in self._results if r.bins_used == r.lower_bound
        )
        print(f"Instances : {len(self._results)}")
        print(f"Total time: {total_time:.4f} s")
        print(f"Avg bins  : {avg_bins:.2f}")
        print(f"Optimal   : {optimal_count}/{len(self._results)}")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the Falkenauer bin-packing benchmark."
    )
    parser.add_argument(
        "--variant",
        choices=[v.value for v in FalkenauerVariant],
        default=FalkenauerVariant.T,
        help="Dataset variant: T (triplets) or U (uniform). Default: T.",
    )
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--num-items",
        type=int,
        default=None,
        metavar="N",
        help="Run only instances with exactly N items.",
    )
    size_group.add_argument(
        "--max-items",
        type=int,
        default=None,
        metavar="N",
        help="Run only instances with at most N items.",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        default=False,
        help="Skip graph generation after the benchmark run.",
    )

    args: argparse.Namespace = parser.parse_args()
    variant: FalkenauerVariant = FalkenauerVariant(args.variant)

    bench: FalkenauerBenchmark = FalkenauerBenchmark(variant)
    bench.run(
        method="b&b",
        num_items=args.num_items,
        max_items=args.max_items,
        generate_graphs=not args.no_graphs,
    )
