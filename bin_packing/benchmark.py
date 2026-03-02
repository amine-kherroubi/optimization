from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


BENCHMARKS_ROOT: Path = Path(__file__).parent.parent / "benchmarks/Falkenauer"
FALKENAUER_T_DIR: Path = BENCHMARKS_ROOT / "Falkenauer_T"
FALKENAUER_U_DIR: Path = BENCHMARKS_ROOT / "Falkenauer U"
GRAPHS_DIR: Path = Path(__file__).parent / "results"


def _solver_worker(
    sizes: list[int],
    bin_capacity: int,
    out_queue: mp.Queue[tuple[int | None, float | None, str | None]],
) -> None:
    """Runs the solver in an isolated process to allow forceful termination on timeout.

    The worker self-times the solve so that subprocess startup and module import
    overhead are excluded from the reported elapsed time.
    Sends (bins_used, elapsed_seconds, error_string) on the queue.
    """
    try:
        # Import before starting the clock — module load is not solve time.
        from bin_packing.solver import BinPackingSolver

        start: float = time.perf_counter()
        solver = BinPackingSolver(sizes, bin_capacity)
        solver.solve()
        elapsed: float = time.perf_counter() - start
        solution = solver.get_solution()
        out_queue.put((solution.total_bins_used, elapsed, None))
    except Exception as exc:
        # Pass exception as string to avoid PicklingError across processes
        out_queue.put((None, None, f"{type(exc).__name__}: {str(exc)}"))


class FalkenauerVariant(StrEnum):
    T = "T"
    U = "U"

    @property
    def directory(self) -> Path:
        return {
            FalkenauerVariant.T: FALKENAUER_T_DIR,
            FalkenauerVariant.U: FALKENAUER_U_DIR,
        }[self]


@dataclass(slots=True)
class BenchmarkResult:
    instance_name: str
    variant: FalkenauerVariant
    num_items: int
    bin_capacity: int
    bins_used: int
    lower_bound: int
    total_weight: int
    elapsed_time: float
    method: str
    timed_out: bool = False


@dataclass(slots=True)
class FalkenauerInstance:
    name: str
    variant: FalkenauerVariant
    num_items: int
    bin_capacity: int
    sizes: list[int]

    @classmethod
    def from_file(
        cls, filepath: Path, variant: FalkenauerVariant
    ) -> FalkenauerInstance:
        lines: list[str] = filepath.read_text(encoding="utf-8").splitlines()
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


@dataclass(slots=True)
class TableWidths:
    """Dynamically holds the max column widths for perfectionist table formatting."""

    name: int
    items: int
    capacity: int
    lb: int
    bins: int
    gap: int
    time: int
    method: int
    state: int

    @property
    def total_width(self) -> int:
        # Sum of all column widths plus 8 separators of " │ " (3 chars each)
        return (
            self.name
            + self.items
            + self.capacity
            + self.lb
            + self.bins
            + self.gap
            + self.time
            + self.method
            + self.state
            + (8 * 3)
        )


class FalkenauerBenchmark:
    def __init__(
        self,
        variant: FalkenauerVariant = FalkenauerVariant.T,
        time_limit: float | None = None,
    ) -> None:
        self._variant: FalkenauerVariant = variant
        self._instances_dir: Path = variant.directory
        self._results: list[BenchmarkResult] = []
        self._time_limit: float | None = time_limit

    def run(
        self,
        method: str = "b&b",
        num_items: int | None = None,
        max_items: int | None = None,
        generate_graphs: bool = True,
    ) -> None:
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

        self._results.clear()
        widths = self._calculate_widths(instances=instances)

        print(
            f"\n\033[1;36mStarting Benchmark:\033[0m Falkenauer Variant {self._variant.value}"
        )
        self._print_header(widths)

        for i, instance in enumerate(instances):
            result: BenchmarkResult = self._solve(instance, method)
            self._results.append(result)
            self._print_row(result, widths)

            if i < len(instances) - 1:
                if self._inter_instance_pause():
                    print("\n\033[93m  [Benchmark stopped by user]\033[0m")
                    break

        self._print_footer(widths)

        if generate_graphs:
            self._generate_graphs(widths)

    def run_instance(
        self, filepath: str | Path, method: str = "b&b"
    ) -> BenchmarkResult:
        if method != "b&b":
            raise ValueError("This solver version only supports method='b&b'.")

        instance: FalkenauerInstance = FalkenauerInstance.from_file(
            Path(filepath), self._variant
        )
        result: BenchmarkResult = self._solve(instance, method)
        self._results.append(result)
        return result

    def available_sizes(self) -> list[int]:
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
        return list(self._results)

    def clear_results(self) -> None:
        self._results.clear()

    @property
    def _completed(self) -> list[BenchmarkResult]:
        """Results from non-timed-out runs only. Used for all graph plotting."""
        return [r for r in self._results if not r.timed_out]

    def print_summary(self) -> None:
        if not self._results:
            print("\033[93mNo results available. Run the benchmark first.\033[0m")
            return

        widths = self._calculate_widths(results=self._results)
        self._print_header(widths)
        for result in self._results:
            self._print_row(result, widths)
        self._print_footer(widths)

    def _calculate_widths(
        self,
        instances: list[FalkenauerInstance] | None = None,
        results: list[BenchmarkResult] | None = None,
    ) -> TableWidths:
        """Calculates dynamic column widths based strictly on the max expected elements."""

        def _max_len(title: str, values: list[str]) -> int:
            if not values:
                return len(title)
            return max(len(title), max(len(v) for v in values))

        if instances is not None:
            name_w = _max_len("Instance", [inst.name for inst in instances])
            items_w = _max_len("Items", [str(inst.num_items) for inst in instances])
            cap_w = _max_len("Capacity", [str(inst.bin_capacity) for inst in instances])
            lb_w = _max_len(
                "LB",
                [str(ceil(sum(inst.sizes) / inst.bin_capacity)) for inst in instances],
            )
            bins_w = _max_len("Bins", [str(inst.num_items) for inst in instances])
            gap_w = _max_len("Gap", [str(inst.num_items) for inst in instances])
        elif results is not None:
            name_w = _max_len("Instance", [r.instance_name for r in results])
            items_w = _max_len("Items", [str(r.num_items) for r in results])
            cap_w = _max_len("Capacity", [str(r.bin_capacity) for r in results])
            lb_w = _max_len("LB", [str(r.lower_bound) for r in results])
            bins_w = _max_len("Bins", [str(r.bins_used) for r in results])
            gap_w = _max_len("Gap", [str(r.bins_used - r.lower_bound) for r in results])
        else:
            raise ValueError("Either instances or results must be provided.")

        time_w = max(len("Time (s)"), 10)  # Base 10 covers "9999.9999" beautifully
        method_w = _max_len("Method", ["b&b"])
        state_w = _max_len("State", ["Done", "T.O."])

        return TableWidths(
            name_w, items_w, cap_w, lb_w, bins_w, gap_w, time_w, method_w, state_w
        )

    def _load_instances(
        self, num_items: int | None, max_items: int | None
    ) -> list[FalkenauerInstance]:
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

        instances.sort(key=lambda inst: (inst.num_items, inst.name))
        return instances

    def _solve(self, instance: FalkenauerInstance, method: str) -> BenchmarkResult:
        # Compute once — both lower_bound and total_weight need this value.
        total_weight: int = sum(instance.sizes)
        lower_bound: int = ceil(total_weight / instance.bin_capacity)

        if self._time_limit is None:
            # Synchronous execution if no time limit is enforced.
            # Import before starting the clock — module load is not solve time.
            from bin_packing.solver import BinPackingSolver

            start: float = time.perf_counter()
            solver: BinPackingSolver = BinPackingSolver(
                instance.sizes, instance.bin_capacity
            )
            solver.solve()
            elapsed: float = time.perf_counter() - start
            solution = solver.get_solution()
            return BenchmarkResult(
                instance_name=instance.name,
                variant=instance.variant,
                num_items=instance.num_items,
                bin_capacity=instance.bin_capacity,
                bins_used=solution.total_bins_used,
                lower_bound=lower_bound,
                total_weight=total_weight,
                elapsed_time=elapsed,
                method=method,
                timed_out=False,
            )

        # Asynchronous execution with strict multiprocessing isolation.
        # The worker self-times the solve so that subprocess startup and module
        # import overhead are NOT included in the reported elapsed time.
        out_queue: mp.Queue[tuple[int | None, float | None, str | None]] = mp.Queue()
        process: mp.Process = mp.Process(
            target=_solver_worker,
            args=(instance.sizes, instance.bin_capacity, out_queue),
            daemon=True,
        )
        process.start()
        process.join(timeout=self._time_limit)

        timed_out: bool = process.is_alive()

        if timed_out:
            process.terminate()
            process.join()  # Cleanly reap the OS process to avoid zombies
            # On timeout we have no solve result; use lower_bound as a placeholder
            # so display arithmetic (gap, fill rate) remains defined.
            bins_used: int = lower_bound
            elapsed: float = self._time_limit
        else:
            try:
                res, worker_elapsed, err = out_queue.get_nowait()
            except queue.Empty:
                # Process exited without writing to the queue (e.g. killed by OOM).
                raise RuntimeError(
                    f"Solver worker for '{instance.name}' exited without reporting a result."
                )
            if err is not None:
                raise RuntimeError(f"Solver failed on '{instance.name}': {err}")
            if res is None or worker_elapsed is None:
                raise RuntimeError(
                    f"Solver worker for '{instance.name}' returned an incomplete result."
                )
            bins_used = res
            elapsed = worker_elapsed

        return BenchmarkResult(
            instance_name=instance.name,
            variant=instance.variant,
            num_items=instance.num_items,
            bin_capacity=instance.bin_capacity,
            bins_used=bins_used,
            lower_bound=lower_bound,
            total_weight=total_weight,
            elapsed_time=elapsed,
            method=method,
            timed_out=timed_out,
        )

    @staticmethod
    def _inter_instance_pause(wait: float = 3.0) -> bool:
        """Cross-platform pause allowing user to skip/stop."""
        if not sys.stdin.isatty():
            return False

        # Graceful fallback for Windows/systems without termios
        try:
            import select
            import termios
            import tty
        except ImportError:
            time.sleep(wait)
            return False

        fd: int = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except termios.error:
            # Another safeguard if stdin is manipulated
            time.sleep(wait)
            return False

        blank: str = "\r" + " " * 56 + "\r"
        try:
            tty.setcbreak(fd)
            deadline: float = time.perf_counter() + wait
            remaining: float = wait
            while remaining > 0:
                secs_left: int = int(remaining) + 1
                msg = f"\r\033[96mNext instance in {secs_left}s  (press any key to stop) \033[0m"
                print(msg, end="", flush=True)
                ready, _, _ = select.select([sys.stdin], [], [], min(0.1, remaining))
                if ready:
                    sys.stdin.read(1)
                    print(blank, end="", flush=True)
                    return True
                remaining = deadline - time.perf_counter()
            print(blank, end="", flush=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return False

    def _generate_graphs(self, widths: TableWidths | None = None) -> None:
        if not self._results:
            print("\033[93mNo results to plot.\033[0m")
            return

        if not self._completed:
            print("\033[93mAll instances timed out — no graphs to plot.\033[0m")
            return

        w = widths.total_width if widths else 80
        print(f"\n\033[1;35m{'═' * w}\033[0m")
        print(f"\033[1;35m{'GRAPH GENERATION':^{w}}\033[0m")
        print(f"\033[1;35m{'═' * w}\033[0m")

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

        print(f"\033[1;35m{'═' * w}\033[0m\n")

    def _plot_time_vs_n(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))

        ns: list[int] = [r.num_items for r in self._completed]
        times: list[float] = [r.elapsed_time for r in self._completed]

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

        order: list[int] = list(np.argsort(ns))
        ax.plot(
            [ns[i] for i in order],
            [times[i] for i in order],
            color="steelblue",
            linewidth=1.2,
            alpha=0.45,
            linestyle="--",
        )

        for result in sorted(
            self._completed, key=lambda r: r.elapsed_time, reverse=True
        )[:3]:
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
        print(f"\033[94mFig 1 \033[90m→\033[0m {path}")

    def _plot_bins_vs_lb(self) -> None:
        fig, ax = plt.subplots(figsize=(max(10, len(self._completed) * 0.6), 5))

        x: np.ndarray = np.arange(len(self._completed))
        lbs: np.ndarray = np.array([r.lower_bound for r in self._completed])
        gaps: np.ndarray = np.array(
            [r.bins_used - r.lower_bound for r in self._completed]
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

        for i, result in enumerate(self._completed):
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
            [r.instance_name for r in self._completed],
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
        print(f"\033[94mFig 2 \033[90m→\033[0m {path}")

    def _plot_fill_rate(self) -> None:
        fig, ax = plt.subplots(figsize=(max(10, len(self._completed) * 0.6), 5))

        fill_rates: list[float] = [
            r.total_weight / (r.bins_used * r.bin_capacity) * 100
            for r in self._completed
        ]

        cmap = matplotlib.colormaps["RdYlGn"]
        norm = Normalize(min(fill_rates), 100)
        colors: list[tuple[float, float, float, float]] = [
            cmap(norm(v)) for v in fill_rates
        ]

        ax.bar(
            np.arange(len(self._completed)), fill_rates, color=colors, edgecolor="white"
        )
        ax.axhline(100, color="black", linewidth=1.2, linestyle="--", label="100 %")
        ax.axhline(
            80, color="orange", linewidth=1.0, linestyle=":", label="Threshold 80 %"
        )

        for i, rate in enumerate(fill_rates):
            ax.text(i, rate + 0.4, f"{rate:.1f}%", ha="center", va="bottom", fontsize=7)

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label="Fill rate (%)", pad=0.02)

        ax.set_xticks(np.arange(len(self._completed)))
        ax.set_xticklabels(
            [r.instance_name for r in self._completed],
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
        print(f"\033[94mFig 3 \033[90m→\033[0m {path}")

    def _plot_time_by_size_group(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))

        size_groups: dict[int, list[float]] = {}
        for result in self._completed:
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

        viridis_cmap = matplotlib.colormaps["viridis"]
        palette = [viridis_cmap(v) for v in np.linspace(0.2, 0.85, len(sorted_sizes))]
        for patch, color in zip(box["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

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
        print(f"\033[94mFig 4 \033[90m→\033[0m {path}")

    def _print_header(self, widths: TableWidths) -> None:
        header: str = (
            f"\033[1m{'Instance':<{widths.name}} │ "
            f"{'Items':<{widths.items}} │ "
            f"{'Capacity':<{widths.capacity}} │ "
            f"{'LB':<{widths.lb}} │ "
            f"{'Bins':<{widths.bins}} │ "
            f"{'Gap':<{widths.gap}} │ "
            f"{'Time (s)':<{widths.time}} │ "
            f"{'Method':<{widths.method}} │ "
            f"{'State':<{widths.state}}\033[0m"
        )
        separator = "\033[90m" + "─" * widths.total_width + "\033[0m"

        print(separator)
        print(header)
        print(separator)

    def _print_row(self, result: BenchmarkResult, widths: TableWidths) -> None:
        gap: int = result.bins_used - result.lower_bound
        time_str: str = f"{result.elapsed_time:.4f}"

        # Apply padding to raw state string before wrapping in ANSI to prevent length issues
        raw_state: str = "T.O." if result.timed_out else "Done"
        padded_state: str = f"{raw_state:<{widths.state}}"
        state_colored: str = (
            f"\033[91m{padded_state}\033[0m"
            if result.timed_out
            else f"\033[92m{padded_state}\033[0m"
        )

        print(
            f"{result.instance_name:<{widths.name}} │ "
            f"{str(result.num_items):<{widths.items}} │ "
            f"{str(result.bin_capacity):<{widths.capacity}} │ "
            f"{str(result.lower_bound):<{widths.lb}} │ "
            f"{str(result.bins_used):<{widths.bins}} │ "
            f"{str(gap):<{widths.gap}} │ "
            f"{time_str:<{widths.time}} │ "
            f"{result.method:<{widths.method}} │ "
            f"{state_colored}"
        )

    def _print_footer(self, widths: TableWidths) -> None:
        # Close the table
        separator = "\033[90m" + "─" * widths.total_width + "\033[0m"
        print(separator)

        # Print the professional stats block
        print(f"\n\033[1;36m{'═' * widths.total_width}\033[0m")
        print(f"\033[1;36m{'BENCHMARK STATISTICS':^{widths.total_width}}\033[0m")
        print(f"\033[1;36m{'═' * widths.total_width}\033[0m")

        if not self._results:
            print(" No results to display.")
            return

        total_time: float = sum(r.elapsed_time for r in self._results)
        avg_time: float = total_time / len(self._results)
        avg_bins: float = sum(r.bins_used for r in self._results) / len(self._results)
        timeout_count: int = sum(1 for r in self._results if r.timed_out)

        print(f"\033[1mInstances processed :\033[0m {len(self._results)}")
        print(f"\033[1mTotal elapsed time  :\033[0m {total_time:.4f} s")
        print(f"\033[1mAverage time        :\033[0m {avg_time:.4f} s")
        print(f"\033[1mAverage bins used   :\033[0m {avg_bins:.2f}")
        if timeout_count:
            print(
                f"\033[1mTimeouts            :\033[0m \033[91m{timeout_count} / {len(self._results)}\033[0m"
            )
        print(f"\033[1;36m{'═' * widths.total_width}\033[0m\n")


if __name__ == "__main__":
    # Workaround for macOS/Windows multiprocessing standard spawn behavior
    # to avoid runtime issues if called globally
    mp.freeze_support()

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[v.value for v in FalkenauerVariant],
        default=FalkenauerVariant.T,
    )

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--num-items", type=int, default=None, metavar="N")
    size_group.add_argument("--max-items", type=int, default=None, metavar="N")

    parser.add_argument("--no-graphs", action="store_true", default=False)
    parser.add_argument("--time-limit", type=float, default=None, metavar="SECS")

    args: argparse.Namespace = parser.parse_args()
    variant: FalkenauerVariant = FalkenauerVariant(args.variant)

    bench: FalkenauerBenchmark = FalkenauerBenchmark(
        variant, time_limit=args.time_limit
    )

    try:
        bench.run(
            method="b&b",
            num_items=args.num_items,
            max_items=args.max_items,
            generate_graphs=not args.no_graphs,
        )
    except KeyboardInterrupt:
        print(
            "\n\n\033[91m\033[1m[!] Benchmark abruptly stopped by user (KeyboardInterrupt).\033[0m"
        )
        sys.exit(130)
    except Exception as e:
        print(f"\n\n\033[91m\033[1m[!] An unexpected error occurred: {e}\033[0m")
        sys.exit(1)
