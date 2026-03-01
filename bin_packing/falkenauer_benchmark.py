from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# from bin_packing.slow import BinPacking, Solution
from bin_packing.fast import BinPacking, Solution

BENCHMARKS_ROOT: Path = Path(__file__).parent.parent / "benchmarks/Falkenauer"
FALKENAUER_T_DIR: Path = BENCHMARKS_ROOT / "Falkenauer_T"
FALKENAUER_U_DIR: Path = BENCHMARKS_ROOT / "Falkenauer U"


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
    ) -> None:
        """Run the benchmark on a filtered subset of instances.

        num_items: run only instances with exactly this many items.
        max_items: run only instances with at most this many items.
        Omit both to run all instances.
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
        """Load instance files from the variant directory, optionally filtered by item count."""
        instances: list[FalkenauerInstance] = []
        for filepath in sorted(self._instances_dir.glob("*.txt")):
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
        return instances

    def _solve(self, instance: FalkenauerInstance, method: str) -> BenchmarkResult:
        """Solve a parsed instance and return a timed result."""
        solver: BinPacking = BinPacking(instance.sizes, instance.bin_capacity)

        start: float = time.perf_counter()
        solver.solve()
        elapsed: float = time.perf_counter() - start

        solution: Solution = solver.get_solution()
        return BenchmarkResult(
            instance_name=instance.name,
            variant=instance.variant,
            num_items=instance.num_items,
            bin_capacity=instance.bin_capacity,
            bins_used=solution.bins_used,
            elapsed_time=elapsed,
            method=method,
        )

    def _header_and_separator(self, name_width: int) -> tuple[str, str]:
        """Build the table header string and a matching separator line."""
        header: str = (
            f"{'Instance':<{name_width}}  "
            f"{'Items':>5}  "
            f"{'Capacity':>8}  "
            f"{'Bins':>4}  "
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
        print(
            f"{result.instance_name:<{name_width}}  "
            f"{result.num_items:>5}  "
            f"{result.bin_capacity:>8}  "
            f"{result.bins_used:>4}  "
            f"{result.elapsed_time:>10.4f}  "
            f"{result.method}"
        )

    def _print_footer(self, name_width: int) -> None:
        """Print the closing separator and aggregate statistics."""
        _, separator = self._header_and_separator(name_width)
        print(separator)
        total_time: float = sum(r.elapsed_time for r in self._results)
        avg_bins: float = sum(r.bins_used for r in self._results) / len(self._results)
        print(f"Instances : {len(self._results)}")
        print(f"Total time: {total_time:.4f} s")
        print(f"Avg bins  : {avg_bins:.2f}")


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

    args: argparse.Namespace = parser.parse_args()
    variant: FalkenauerVariant = FalkenauerVariant(args.variant)

    bench: FalkenauerBenchmark = FalkenauerBenchmark(variant)
    bench.run(method="b&b", num_items=args.num_items, max_items=args.max_items)
