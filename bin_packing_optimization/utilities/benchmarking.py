from __future__ import annotations

import csv
import inspect
import multiprocessing as mp
import queue
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from math import ceil
from pathlib import Path
from types import ModuleType
from typing import Any

from bin_packing_optimization.datasets.types import Instance, DatasetConfig
from bin_packing_optimization.datasets.registry import DATASET_REGISTRY
from bin_packing_optimization.datasets.solutions import get_instance_solution


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
            + (8 * 3)
        )


def _format_float(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _solver_worker(
    sizes: list[int],
    bin_capacity: int,
    method: str | None,
    method_args: dict[str, Any],
    solver_module: ModuleType,
    out_queue: mp.Queue[tuple[int | None, float | None, str | None]],
) -> None:
    """Run the solver in an isolated subprocess and report (bins, elapsed, error)."""
    import signal
    import time

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        BinPackingSolver = solver_module.BinPackingSolver
        start: float = time.perf_counter()
        solver = BinPackingSolver(sizes, bin_capacity)
        _invoke_solver_safely(solver, method, method_args)
        elapsed: float = time.perf_counter() - start
        solution = solver.get_solution()
        out_queue.put((solution.total_bins_used, elapsed, None))
    except Exception as exc:
        out_queue.put((None, None, f"{type(exc).__name__}: {exc}"))


def _invoke_solver_safely(
    solver: Any,
    method: str | None,
    method_args: dict[str, Any] | None = None,
) -> None:
    method_args = dict(method_args or {})
    solve_sig = inspect.signature(solver.solve)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in solve_sig.parameters.values()
    )
    accepts_method_param = "method" in solve_sig.parameters

    if method is not None and not accepts_method_param:
        print(
            "\033[93m[warning]\033[0m Method was provided, but this solver does not "
            "accept a 'method' parameter in solve(); it will be ignored."
        )

    if method_args and not accepts_kwargs:
        unsupported_args = sorted(
            k for k in method_args.keys() if k not in solve_sig.parameters
        )
        if unsupported_args:
            print(
                "\033[93m[warning]\033[0m Ignoring unsupported method_args for "
                f"this solver: {', '.join(unsupported_args)}"
            )
            method_args = {
                k: v for k, v in method_args.items() if k in solve_sig.parameters
            }

    call_kwargs = dict(method_args)
    if accepts_method_param:
        call_kwargs["method"] = method
    solver.solve(**call_kwargs)


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
    """Dataset-agnostic benchmark runner."""

    def __init__(
        self,
        dataset: DatasetConfig,
        solver_module: ModuleType,
        time_limit: float | None = None,
    ) -> None:
        self._dataset: DatasetConfig = dataset
        self._solver_module: ModuleType = solver_module
        self._time_limit: float | None = time_limit
        self._results: list[BenchmarkResult] = []

    @classmethod
    def from_dataset_key(
        cls,
        dataset_key: str,
        solver_module: ModuleType,
        time_limit: float | None = None,
    ) -> "Benchmark":
        """Create a benchmark by resolving a dataset key from the registry."""
        try:
            dataset = DATASET_REGISTRY[dataset_key]
        except KeyError as exc:
            available = ", ".join(sorted(DATASET_REGISTRY.keys()))
            raise ValueError(
                f"Unknown dataset key '{dataset_key}'. Available keys: {available}"
            ) from exc
        return cls(dataset=dataset, solver_module=solver_module, time_limit=time_limit)

    def run(
        self,
        method: str | None,
        method_args: dict[str, Any] | None = None,
        num_items: int | None = None,
        max_items: int | None = None,
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
                    result = self._solve(instance, method, method_args, stop_flag)
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

    def run_instance(
        self,
        instance: Instance,
        method: str | None = None,
        method_args: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        result = self._solve(instance, method, method_args)
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

    def save_results_to_csv(
        self,
        csv_out: str | Path | None = None,
        results_dir: str | Path | None = None,
        timestamp: bool = True,
    ) -> Path:
        """Persist benchmark results to a CSV file.

        Parameters
        ----------
        csv_out:
            Optional explicit file path to write. When provided, its parent
            directories will be created if needed and the path will be used as-is.
        results_dir:
            When ``csv_out`` is not provided, the results file will be created
            under this directory. Defaults to the project-wide ``results/``
            directory.
        timestamp:
            When True, append a timestamp to the filename to avoid accidental
            overwrites across runs.

        Returns
        -------
        Path
            The path to the written CSV file.
        """
        if not self._results:
            raise ValueError("No results to save. Run the benchmark first.")

        if csv_out:
            out_path = Path(csv_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default location: put results under the solver's folder so outputs
            # are grouped by method. Structure:
            # <solver_dir>/results/<dataset_key>/<timestamp>/results.csv
            ts = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
            base_dir = (
                Path(results_dir)
                if results_dir is not None
                else Path("results") / self._dataset.key / ts
            )
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / "results.csv"

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "instance_name",
                    "dataset_key",
                    "num_items",
                    "bin_capacity",
                    "bins_used",
                    "lower_bound",
                    "total_weight",
                    "elapsed_time",
                    "method",
                    "timed_out",
                ],
            )
            writer.writeheader()
            for r in self._results:
                writer.writerow(asdict(r))

        return out_path

    @property
    def _completed(self) -> list[BenchmarkResult]:
        """Results that finished within the time limit."""
        return [r for r in self._results if not r.timed_out]

    def _load_instances(
        self,
        num_items: int | None,
        max_items: int | None,
    ) -> list[Instance]:
        instances: list[Instance] = []
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
        instance: Instance,
        method: str | None,
        method_args: dict[str, Any] | None = None,
        stop_flag: threading.Event | None = None,
    ) -> BenchmarkResult:
        total_weight = sum(instance.sizes)
        fallback_lower_bound = ceil(total_weight / instance.bin_capacity)
        reference_solution = get_instance_solution(instance.dataset_key, instance.name)
        lower_bound = (
            reference_solution.best_lb
            if reference_solution is not None
            else fallback_lower_bound
        )
        solver_method = "branch and bound" if method == "b&b" else method
        method_label = method if method is not None else "<default>"

        if self._time_limit is None:
            BinPackingSolver = self._solver_module.BinPackingSolver

            start = time.perf_counter()
            solver = BinPackingSolver(instance.sizes, instance.bin_capacity)
            _invoke_solver_safely(solver, solver_method, method_args)
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
                method=method_label,
                timed_out=False,
            )

        out_queue: mp.Queue[tuple[int | None, float | None, str | None]] = mp.Queue()
        process = mp.Process(
            target=_solver_worker,
            args=(
                instance.sizes,
                instance.bin_capacity,
                solver_method,
                dict(method_args or {}),
                self._solver_module,
                out_queue,
            ),
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
            bins_used = lower_bound
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
            method=method_label,
            timed_out=timed_out,
        )

    def _calculate_widths(
        self,
        instances: list[Instance] | None = None,
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

        time_candidates = []
        if results is not None:
            time_candidates = [_format_float(r.elapsed_time) for r in results]
        time_w = max(len("Time (s)"), 10, *(len(t) for t in time_candidates))
        method_w = max(
            len("Method"),
            20,  # minimum width
            max((len(r.method) for r in (results or [])), default=0),
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
        time_str = _format_float(result.elapsed_time)

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
        print()


def create_benchmark(
    dataset_key: str,
    solver_module: ModuleType,
    time_limit: float | None = None,
) -> Benchmark:
    """Return a benchmark configured from a dataset registry key."""
    return Benchmark.from_dataset_key(
        dataset_key=dataset_key,
        solver_module=solver_module,
        time_limit=time_limit,
    )
