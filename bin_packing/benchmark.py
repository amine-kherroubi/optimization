from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import threading
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


# ── Project paths ──────────────────────────────────────────────────────────────

_PACKAGE_DIR: Path = Path(__file__).parent
_BENCHMARKS_ROOT: Path = _PACKAGE_DIR.parent / "benchmarks"
GRAPHS_DIR: Path = _PACKAGE_DIR / "results"


# ═══════════════════════════════════════════════════════════════════════════════
# Core data types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class BenchmarkInstance:
    """A single problem instance, independent of its source dataset."""

    name: str
    dataset_key: str
    num_items: int
    bin_capacity: int
    sizes: list[int]


@dataclass(slots=True)
class BenchmarkResult:
    """The outcome of solving one instance."""

    instance_name: str
    dataset_key: str
    num_items: int
    bin_capacity: int
    bins_used: int
    lower_bound: int
    total_weight: int
    elapsed_time: float
    method: str
    timed_out: bool = False


@dataclass(slots=True)
class TableWidths:
    """Column widths for the results table, computed dynamically from data."""

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
            + (8 * 3)  # 8 " | " separators, 3 chars each
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Instance parsers
# ═══════════════════════════════════════════════════════════════════════════════


def parse_standard(filepath: Path, dataset_key: str) -> BenchmarkInstance:
    """Parses the standard one-instance-per-file format:
        line 0   — number of items
        line 1   — bin capacity
        lines 2+ — item sizes, one per line
    Used by: Falkenauer T/U, Scholl 1/2/3.
    """
    lines = filepath.read_text(encoding="utf-8").splitlines()
    num_items: int = int(lines[0])
    bin_capacity: int = int(lines[1])
    sizes: list[int] = [int(lines[i]) for i in range(2, 2 + num_items)]
    return BenchmarkInstance(
        name=filepath.stem,
        dataset_key=dataset_key,
        num_items=num_items,
        bin_capacity=bin_capacity,
        sizes=sizes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset registry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable descriptor for a benchmark dataset.

    To register a new dataset, create a DatasetConfig and call
    register_dataset(). No other code in this module needs to change.
    """

    key: str  # CLI identifier, e.g. "scholl-1"
    label: str  # Human-readable name
    directory: Path  # Root directory of instance files
    parser: Callable[[Path, str], BenchmarkInstance]  # File → BenchmarkInstance
    glob: str = "*.txt"  # Filename glob pattern


DATASET_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(config: DatasetConfig) -> None:
    """Register a dataset configuration. Raises ValueError on duplicate key."""
    if config.key in DATASET_REGISTRY:
        raise ValueError(f"Dataset key '{config.key}' is already registered.")
    DATASET_REGISTRY[config.key] = config


# ── Built-in datasets ──────────────────────────────────────────────────────────

register_dataset(
    DatasetConfig(
        key="falkenauer-t",
        label="Falkenauer T",
        directory=_BENCHMARKS_ROOT / "Falkenauer" / "Falkenauer_T",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="falkenauer-u",
        label="Falkenauer U",
        directory=_BENCHMARKS_ROOT / "Falkenauer" / "Falkenauer U",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="scholl-1",
        label="Scholl 1",
        directory=_BENCHMARKS_ROOT / "Scholl" / "Scholl_1",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="scholl-2",
        label="Scholl 2",
        directory=_BENCHMARKS_ROOT / "Scholl" / "Scholl_2",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="scholl-3",
        label="Scholl 3",
        directory=_BENCHMARKS_ROOT / "Scholl" / "Scholl_3",
        parser=parse_standard,
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# Solver worker (multiprocessing isolation)
# ═══════════════════════════════════════════════════════════════════════════════


def _solver_worker(
    sizes: list[int],
    bin_capacity: int,
    method: str,
    out_queue: mp.Queue[tuple[int | None, float | None, str | None]],
) -> None:
    """Run the solver in an isolated subprocess and report (bins, elapsed, error)."""
    import signal
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent  # dossier contenant bin_packing
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    signal.signal(
        signal.SIGINT, signal.SIG_IGN
    )  # parent handles Ctrl+C via process.terminate()
    try:
        from bin_packing.solver import BinPackingSolver

        start: float = time.perf_counter()
        solver = BinPackingSolver(sizes, bin_capacity)
        solver.solve(method)
        elapsed: float = time.perf_counter() - start
        solution = solver.get_solution()
        out_queue.put((solution.total_bins_used, elapsed, None))
    except Exception as exc:
        # Stringify to avoid pickling errors across process boundaries.
        out_queue.put((None, None, f"{type(exc).__name__}: {exc}"))


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════


class _StdinWatcher(threading.Thread):
    """Daemon thread that sets stop_flag when the user presses 'q'.

    The caller is responsible for putting stdin into cbreak mode before
    starting this thread and restoring it afterwards. This thread only reads.
    """

    def __init__(self, stop_flag: threading.Event) -> None:
        super().__init__(daemon=True)
        self._stop_flag = stop_flag
        self._quit = threading.Event()

    def stop(self) -> None:
        self._quit.set()

    def run(self) -> None:
        try:
            import select
        except ImportError:
            return

        while not self._quit.is_set():
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                key = sys.stdin.read(1)
                if key.lower() == "q":
                    self._stop_flag.set()
                    break


class Benchmark:
    """Dataset-agnostic benchmark runner. Accepts any DatasetConfig."""

    def __init__(
        self,
        dataset: DatasetConfig,
        time_limit: float | None = None,
    ) -> None:
        self._dataset: DatasetConfig = dataset
        self._time_limit: float | None = time_limit
        self._results: list[BenchmarkResult] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        method: str = "branch and bound",
        num_items: int | None = None,
        max_items: int | None = None,
        generate_graphs: bool = True,
    ) -> None:
        if num_items is not None and max_items is not None:
            raise ValueError("num_items and max_items are mutually exclusive.")

        instances = self._load_instances(num_items, max_items)
        if not instances:
            qualifier = ""
            if num_items is not None:
                qualifier = f" with exactly {num_items} items"
            elif max_items is not None:
                qualifier = f" with at most {max_items} items"
            raise FileNotFoundError(
                f"No instances{qualifier} found in '{self._dataset.directory}'."
            )

        self._results.clear()
        widths = self._calculate_widths(instances=instances)

        print(f"\n\033[1;36mStarting Benchmark:\033[0m {self._dataset.label}")
        self._print_header(widths)

        # Put stdin into cbreak mode in the main thread so that the main
        # thread's finally block is guaranteed to restore it — even on
        # KeyboardInterrupt. Daemon threads are torn down before their finally
        # blocks run when the process exits, so terminal restore must live here.
        old_terminal_settings = None
        if sys.stdin.isatty():
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old_terminal_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
            except Exception:
                old_terminal_settings = None

        stop_flag = threading.Event()
        stdin_watcher = _StdinWatcher(stop_flag)
        if old_terminal_settings is not None:
            stdin_watcher.start()

        try:
            for instance in instances:
                if stop_flag.is_set():
                    print("\n\033[93m[Benchmark stopped by user]\033[0m")
                    break
                try:
                    result = self._solve(instance, method, stop_flag)
                except Exception as exc:
                    print(f"\033[91m[!] Skipping '{instance.name}': {exc}\033[0m")
                    continue

                self._results.append(result)
                self._print_row(result, widths)
        finally:
            stdin_watcher.stop()
            if stdin_watcher.is_alive():
                stdin_watcher.join(timeout=0.5)
            if old_terminal_settings is not None:
                try:
                    import termios

                    termios.tcsetattr(
                        sys.stdin.fileno(), termios.TCSADRAIN, old_terminal_settings
                    )
                except Exception:
                    pass

        self._print_footer(widths)

        if generate_graphs and self._results:
            self._generate_graphs()

    def run_instance(
        self, filepath: str | Path, method: str = "branch and bound"
    ) -> BenchmarkResult:
        instance = self._dataset.parser(Path(filepath), self._dataset.key)
        result = self._solve(instance, method)
        self._results.append(result)
        return result

    def available_sizes(self) -> list[int]:
        sizes: set[int] = set()
        for filepath in self._dataset.directory.glob(self._dataset.glob):
            try:
                inst = self._dataset.parser(filepath, self._dataset.key)
                sizes.add(inst.num_items)
            except (ValueError, IndexError):
                continue
        return sorted(sizes)

    def get_results(self) -> list[BenchmarkResult]:
        return list(self._results)

    def clear_results(self) -> None:
        self._results.clear()

    def print_summary(self) -> None:
        if not self._results:
            print("\033[93mNo results available. Run the benchmark first.\033[0m")
            return
        widths = self._calculate_widths(results=self._results)
        self._print_header(widths)
        for result in self._results:
            self._print_row(result, widths)
        self._print_footer(widths)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def _completed(self) -> list[BenchmarkResult]:
        """Results that finished within the time limit."""
        return [r for r in self._results if not r.timed_out]

    def _load_instances(
        self,
        num_items: int | None,
        max_items: int | None,
    ) -> list[BenchmarkInstance]:
        instances: list[BenchmarkInstance] = []
        for filepath in self._dataset.directory.glob(self._dataset.glob):
            try:
                inst = self._dataset.parser(filepath, self._dataset.key)
            except (ValueError, IndexError):
                continue
            if num_items is not None and inst.num_items != num_items:
                continue
            if max_items is not None and inst.num_items > max_items:
                continue
            instances.append(inst)
        instances.sort(key=lambda i: (i.num_items, i.name))
        return instances

    def _solve(
        self,
        instance: BenchmarkInstance,
        method: str,
        stop_flag: threading.Event | None = None,
    ) -> BenchmarkResult:
        total_weight = sum(instance.sizes)
        lower_bound = ceil(total_weight / instance.bin_capacity)
        solver_method = "branch and bound" if method == "b&b" else method

        if self._time_limit is None:
            from bin_packing.solver import BinPackingSolver

            start = time.perf_counter()
            solver = BinPackingSolver(instance.sizes, instance.bin_capacity)
            solver.solve(solver_method)
            elapsed = time.perf_counter() - start
            solution = solver.get_solution()
            return BenchmarkResult(
                instance_name=instance.name,
                dataset_key=instance.dataset_key,
                num_items=instance.num_items,
                bin_capacity=instance.bin_capacity,
                bins_used=solution.total_bins_used,
                lower_bound=lower_bound,
                total_weight=total_weight,
                elapsed_time=elapsed,
                method=method,
                timed_out=False,
            )

        out_queue: mp.Queue[tuple[int | None, float | None, str | None]] = mp.Queue()
        process = mp.Process(
            target=_solver_worker,
            args=(instance.sizes, instance.bin_capacity, solver_method, out_queue),
            daemon=True,
        )
        process.start()

        deadline = time.monotonic() + (self._time_limit or float("inf"))
        poll_interval = 0.05
        while process.is_alive() and time.monotonic() < deadline:
            if stop_flag is not None and stop_flag.is_set():
                break
            process.join(timeout=poll_interval)

        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join()
            bins_used = lower_bound  # sentinel; excluded from quality stats
            elapsed = self._time_limit
        else:
            try:
                res, worker_elapsed, err = out_queue.get_nowait()
            except queue.Empty:
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
            dataset_key=instance.dataset_key,
            num_items=instance.num_items,
            bin_capacity=instance.bin_capacity,
            bins_used=bins_used,
            lower_bound=lower_bound,
            total_weight=total_weight,
            elapsed_time=elapsed,
            method=method,
            timed_out=timed_out,
        )

    # ── Table rendering ────────────────────────────────────────────────────────

    def _calculate_widths(
        self,
        instances: list[BenchmarkInstance] | None = None,
        results: list[BenchmarkResult] | None = None,
    ) -> TableWidths:
        def _max_len(title: str, values: list[str]) -> int:
            if not values:
                return len(title)
            return max(len(title), max(len(v) for v in values))

        if instances is not None:
            name_w = _max_len("Instance", [i.name for i in instances])
            items_w = _max_len("Items", [str(i.num_items) for i in instances])
            cap_w = _max_len("Capacity", [str(i.bin_capacity) for i in instances])
            lb_w = _max_len(
                "LB", [str(ceil(sum(i.sizes) / i.bin_capacity)) for i in instances]
            )
            bins_w = _max_len("Bins", [str(i.num_items) for i in instances])
            gap_w = _max_len("Gap", [str(i.num_items) for i in instances])
        elif results is not None:
            name_w = _max_len("Instance", [r.instance_name for r in results])
            items_w = _max_len("Items", [str(r.num_items) for r in results])
            cap_w = _max_len("Capacity", [str(r.bin_capacity) for r in results])
            lb_w = _max_len("LB", [str(r.lower_bound) for r in results])
            bins_w = _max_len("Bins", [str(r.bins_used) for r in results])
            gap_w = _max_len("Gap", [str(r.bins_used - r.lower_bound) for r in results])
        else:
            raise ValueError("Either instances or results must be provided.")

        time_w = max(len("Time (s)"), 10)
        method_w = _max_len(
            "Method", [r.method for r in (results or [])] or ["branch and bound"]
        )
        state_w = _max_len("State", ["Done", "T.O."])

        return TableWidths(
            name_w, items_w, cap_w, lb_w, bins_w, gap_w, time_w, method_w, state_w
        )

    def _print_header(self, widths: TableWidths) -> None:
        header = (
            f"\033[1m{'Instance':<{widths.name}} \u2502 "
            f"{'Items':<{widths.items}} \u2502 "
            f"{'Capacity':<{widths.capacity}} \u2502 "
            f"{'LB':<{widths.lb}} \u2502 "
            f"{'Bins':<{widths.bins}} \u2502 "
            f"{'Gap':<{widths.gap}} \u2502 "
            f"{'Time (s)':<{widths.time}} \u2502 "
            f"{'Method':<{widths.method}} \u2502 "
            f"{'State':<{widths.state}}\033[0m"
        )
        separator = "\033[90m" + "\u2500" * widths.total_width + "\033[0m"
        print(separator)
        print(header)
        print(separator)

    def _print_row(self, result: BenchmarkResult, widths: TableWidths) -> None:
        gap = result.bins_used - result.lower_bound
        time_str = f"{result.elapsed_time:.4f}"

        raw_state = "T.O." if result.timed_out else "Done"
        padded_state = f"{raw_state:<{widths.state}}"
        state_colored = (
            f"\033[91m{padded_state}\033[0m"
            if result.timed_out
            else f"\033[92m{padded_state}\033[0m"
        )

        print(
            f"{result.instance_name:<{widths.name}} \u2502 "
            f"{str(result.num_items):<{widths.items}} \u2502 "
            f"{str(result.bin_capacity):<{widths.capacity}} \u2502 "
            f"{str(result.lower_bound):<{widths.lb}} \u2502 "
            f"{str(result.bins_used):<{widths.bins}} \u2502 "
            f"{str(gap):<{widths.gap}} \u2502 "
            f"{time_str:<{widths.time}} \u2502 "
            f"{result.method:<{widths.method}} \u2502 "
            f"{state_colored}"
        )

    def _print_footer(self, widths: TableWidths) -> None:
        separator = "\033[90m" + "\u2500" * widths.total_width + "\033[0m"
        print(separator)

        print(f"\n\033[1;36m{chr(0x2550) * widths.total_width}\033[0m")
        print(f"\033[1;36m{'BENCHMARK STATISTICS':^{widths.total_width}}\033[0m")
        print(f"\033[1;36m{chr(0x2550) * widths.total_width}\033[0m")

        if not self._results:
            print(" No results to display.")
            return

        completed = self._completed
        timeout_count = len(self._results) - len(completed)
        total_time = sum(r.elapsed_time for r in self._results)

        print(f"\033[1mInstances processed      :\033[0m {len(self._results)}")

        if completed:
            average_time = sum(r.elapsed_time for r in completed) / len(completed)
            average_bins = sum(r.bins_used for r in completed) / len(completed)
            print(f"\033[1mAverage time (completed) :\033[0m {average_time:.4f} s")
            print(f"\033[1mAverage bins (completed) :\033[0m {average_bins:.2f}")

        print(f"\033[1mTotal elapsed time       :\033[0m {total_time:.4f} s")

        if timeout_count:
            print(
                f"\033[1mTimeouts                 :\033[0m "
                f"\033[91m{timeout_count} / {len(self._results)}\033[0m"
            )

        print(f"\033[1;36m{chr(0x2550) * widths.total_width}\033[0m\n")

    # ── Graph generation ───────────────────────────────────────────────────────

    def _generate_graphs(self) -> None:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        key = self._dataset.key
        print(f"\n\033[1;36mGenerating graphs...\033[0m")
        self._plot_solve_times(key)
        self._plot_bins_vs_lb(key)
        self._plot_fill_rate(key)
        self._plot_time_by_size_group(key)

    def _plot_solve_times(self, dataset_key: str) -> None:
        fig, ax = plt.subplots(figsize=(max(10, len(self._completed) * 0.6), 5))

        times = [r.elapsed_time for r in self._completed]
        max_t = max(times)
        colors = ["#e74c3c" if t == max_t else "#3498db" for t in times]

        ax.bar(
            range(len(self._completed)),
            times,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(self._completed)))
        ax.set_xticklabels(
            [r.instance_name for r in self._completed],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Solve time (s)", fontsize=12)
        ax.set_title(
            f"Solve Time per Instance  \u2014  {self._dataset.label}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        fig.tight_layout()

        path = GRAPHS_DIR / f"fig1_solve_times_{dataset_key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(
            f"\033[94mFig 1 \033[90m\u2192\033[0m {path.relative_to(_PACKAGE_DIR.parent)}"
        )

    def _plot_bins_vs_lb(self, dataset_key: str) -> None:
        fig, ax = plt.subplots(figsize=(max(10, len(self._completed) * 0.6), 5))

        x = range(len(self._completed))
        lb_vals = [r.lower_bound for r in self._completed]
        gap_vals = [r.bins_used - r.lower_bound for r in self._completed]

        ax.bar(
            x,
            lb_vals,
            label="Lower Bound",
            color="#2ecc71",
            alpha=0.85,
            edgecolor="white",
        )
        ax.bar(
            x,
            gap_vals,
            bottom=lb_vals,
            label="Gap",
            color="#e74c3c",
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [r.instance_name for r in self._completed],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("Bins", fontsize=12)
        ax.set_title(
            f"Bins Used vs Lower Bound  \u2014  {self._dataset.label}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

        path = GRAPHS_DIR / f"fig2_bins_vs_lb_{dataset_key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(
            f"\033[94mFig 2 \033[90m\u2192\033[0m {path.relative_to(_PACKAGE_DIR.parent)}"
        )

    def _plot_fill_rate(self, dataset_key: str) -> None:
        fig, ax = plt.subplots(figsize=(max(10, len(self._completed) * 0.6), 5))

        fill_rates = [
            r.total_weight / (r.bins_used * r.bin_capacity) * 100
            for r in self._completed
        ]
        cmap = matplotlib.colormaps["RdYlGn"]
        norm = Normalize(min(fill_rates), 100)
        colors = [cmap(norm(v)) for v in fill_rates]

        ax.bar(range(len(self._completed)), fill_rates, color=colors, edgecolor="white")
        ax.axhline(100, color="black", linewidth=1.2, linestyle="--", label="100 %")
        ax.axhline(
            80, color="orange", linewidth=1.0, linestyle=":", label="Threshold 80 %"
        )

        for i, rate in enumerate(fill_rates):
            ax.text(i, rate + 0.4, f"{rate:.1f}%", ha="center", va="bottom", fontsize=7)

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label="Fill rate (%)", pad=0.02)

        ax.set_xticks(range(len(self._completed)))
        ax.set_xticklabels(
            [r.instance_name for r in self._completed],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_ylim(0, 115)
        ax.set_ylabel("Average bin fill rate (%)", fontsize=12)
        ax.set_title(
            f"Solution Quality: Average Bin Fill Rate  \u2014  {self._dataset.label}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

        path = GRAPHS_DIR / f"fig3_fill_rate_{dataset_key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(
            f"\033[94mFig 3 \033[90m\u2192\033[0m {path.relative_to(_PACKAGE_DIR.parent)}"
        )

    def _plot_time_by_size_group(self, dataset_key: str) -> None:
        size_groups: dict[int, list[float]] = {}
        for r in self._completed:
            size_groups.setdefault(r.num_items, []).append(r.elapsed_time)

        if len(size_groups) < 2:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        sorted_sizes = sorted(size_groups.keys())
        group_times = [size_groups[n] for n in sorted_sizes]
        labels = [str(n) for n in sorted_sizes]

        box = ax.boxplot(
            group_times,
            tick_labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        viridis = matplotlib.colormaps["viridis"]
        palette = [viridis(v) for v in np.linspace(0.2, 0.85, len(sorted_sizes))]
        for patch, color in zip(box["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        rng = np.random.default_rng(seed=0)
        for idx, times in enumerate(group_times):
            jitter = rng.uniform(-0.12, 0.12, size=len(times))
            ax.scatter(
                np.full(len(times), idx + 1) + jitter,
                times,
                s=28,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
                color=palette[idx],
            )

        ax.set_yscale("log")
        ax.set_xlabel("Number of items (n)", fontsize=12)
        ax.set_ylabel("Solve time (s)  [log scale]", fontsize=12)
        ax.set_title(
            f"Solve Time Distribution by Instance Size  \u2014  {self._dataset.label}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        fig.tight_layout()

        path = GRAPHS_DIR / f"fig4_time_by_size_{dataset_key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(
            f"\033[94mFig 4 \033[90m\u2192\033[0m {path.relative_to(_PACKAGE_DIR.parent)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.freeze_support()

    _dataset_keys = sorted(DATASET_REGISTRY.keys())
    _dataset_help = "\n".join(
        f"  {k:22s} {DATASET_REGISTRY[k].label}" for k in _dataset_keys
    )

    arg_parser = argparse.ArgumentParser(
        description="Bin-packing benchmark runner.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    arg_parser.add_argument(
        "--dataset",
        choices=_dataset_keys,
        required=True,
        metavar="DATASET",
        help=f"Dataset to benchmark. Available:\n{_dataset_help}",
    )

    size_group = arg_parser.add_mutually_exclusive_group()
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

    arg_parser.add_argument(
        "--no-graphs",
        action="store_true",
        default=False,
        help="Skip graph generation.",
    )
    arg_parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECS",
        help="Per-instance time limit in seconds (enables multiprocessing isolation).",
    )
    arg_parser.add_argument(
        "--method",
        choices=["branch and bound", "backtracking", "dynamic programming"],
        default="branch and bound",
        metavar="METHOD",
        help=(
            "Solving method (default: 'branch and bound').\n"
            "  branch and bound    \u2014 exact, with L1 pruning\n"
            "  backtracking        \u2014 exact, symmetry-breaking\n"
            "  dynamic programming \u2014 exact, bitmask DP (n \u2264 20 only)"
        ),
    )

    args = arg_parser.parse_args()
    dataset_cfg = DATASET_REGISTRY[args.dataset]
    bench = Benchmark(dataset_cfg, time_limit=args.time_limit)

    try:
        bench.run(
            method=args.method,
            num_items=args.num_items,
            max_items=args.max_items,
            generate_graphs=not args.no_graphs,
        )
    except KeyboardInterrupt:
        print(
            "\n\n\033[91m\033[1m[!] Benchmark abruptly stopped by user "
            "(KeyboardInterrupt).\033[0m"
        )
        sys.exit(130)
    except Exception as e:
        print(f"\n\n\033[91m\033[1m[!] An unexpected error occurred: {e}\033[0m")
        sys.exit(1)
