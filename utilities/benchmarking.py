from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import multiprocessing as mp
import queue
import sys
import threading
import time
from dataclasses import dataclass, asdict
from math import ceil
from pathlib import Path
from typing import Any

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.types import Instance, DatasetConfig
from datasets.registry import DATASET_REGISTRY


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


def _solver_worker(
    sizes: list[int],
    bin_capacity: int,
    method: str | None,
    method_args: dict[str, Any],
    solver_path: str,
    out_queue: mp.Queue[tuple[int | None, float | None, str | None]],
) -> None:
    """Run the solver in an isolated subprocess and report (bins, elapsed, error)."""
    import importlib.util
    import signal
    import time
    from pathlib import Path

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        abs_path = Path(__file__).parent / solver_path
        spec = importlib.util.spec_from_file_location("solver", abs_path)
        assert spec is not None
        solver_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = solver_mod
        assert spec.loader is not None
        spec.loader.exec_module(solver_mod)
        BinPackingSolver = solver_mod.BinPackingSolver

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
                "\033[93m[warning]\033[0m Ignoring unsupported --method-args for "
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
        solver_path: Path,
        time_limit: float | None = None,
    ) -> None:
        self._dataset: DatasetConfig = dataset
        self._solver_path: Path = solver_path
        self._time_limit: float | None = time_limit
        self._results: list[BenchmarkResult] = []

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
        filepath: str | Path,
        method: str | None = None,
        method_args: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        instance = self._dataset.parser(Path(filepath), self._dataset.key)
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
        lower_bound = ceil(total_weight / instance.bin_capacity)
        solver_method = "branch and bound" if method == "b&b" else method
        method_label = method if method is not None else "<default>"

        if self._time_limit is None:
            abs_path = self._solver_path
            spec = importlib.util.spec_from_file_location("solver", abs_path)
            assert spec is not None
            solver_mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = solver_mod
            assert spec.loader is not None
            spec.loader.exec_module(solver_mod)
            BinPackingSolver = solver_mod.BinPackingSolver

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
                str(self._solver_path),
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

        time_w = max(len("Time (s)"), 10)
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
        "--solver",
        required=True,
        metavar="PATH",
        help="Path to the solver.py to benchmark (e.g. 1_exact_methods/solver.py).",
    )
    arg_parser.add_argument(
        "--dataset",
        choices=_dataset_keys,
        required=True,
        metavar="DATASET",
        help=f"Dataset to benchmark. Available:\n{_dataset_help}",
    )
    arg_parser.add_argument(
        "--method",
        default=None,
        metavar="METHOD",
        help="Optional solving method passed to BinPackingSolver.solve(method=...).",
    )
    arg_parser.add_argument(
        "--method-args",
        default=None,
        metavar="K=V,...",
        help=(
            "Optional comma-separated key/value params passed to solve(), "
            "for example: model_path=foo.pkl,max_iterations=5000"
        ),
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
        "--csv-out",
        default=None,
        metavar="FILE",
        help="Path to save benchmark results in CSV format.",
    )
    arg_parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECS",
        help="Per-instance time limit in seconds (enables multiprocessing isolation).",
    )

    args = arg_parser.parse_args()

    def _parse_method_args(raw_args: str | None) -> dict[str, Any]:
        if raw_args is None or not raw_args.strip():
            return {}
        parsed: dict[str, Any] = {}
        for chunk in raw_args.split(","):
            piece = chunk.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise ValueError(
                    f"Invalid --method-args entry '{piece}'. Expected key=value."
                )
            key, value = piece.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("Method-arg keys cannot be empty.")
            value = value.strip()
            lowered = value.lower()
            if lowered == "true":
                parsed[key] = True
            elif lowered == "false":
                parsed[key] = False
            else:
                try:
                    parsed[key] = int(value)
                except ValueError:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
        return parsed

    method_args = _parse_method_args(args.method_args)

    solver_path = (_PROJECT_ROOT / args.solver).resolve()
    if not solver_path.is_file():
        arg_parser.error(f"Solver file not found: {solver_path}")

    dataset_cfg = DATASET_REGISTRY[args.dataset]
    bench = Benchmark(dataset_cfg, solver_path, time_limit=args.time_limit)

    try:
        bench.run(
            method=args.method,
            method_args=method_args,
            num_items=args.num_items,
            max_items=args.max_items,
        )

        if args.csv_out:
            results = bench.get_results()
            with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
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
                for r in results:
                    writer.writerow(asdict(r))

    except KeyboardInterrupt:
        print(
            "\n\n\033[91m\033[1m[!] Benchmark abruptly stopped by user "
            "(KeyboardInterrupt).\033[0m"
        )
        sys.exit(130)
    except Exception as e:
        print(f"\n\n\033[91m\033[1m[!] An unexpected error occurred: {e}\033[0m")
        sys.exit(1)
