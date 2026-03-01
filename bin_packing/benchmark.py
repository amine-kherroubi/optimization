import time
from dataclasses import dataclass
from pathlib import Path

# from bin_packing.slow import BinPacking, Solution
from bin_packing.fast import BinPacking, Solution


@dataclass(slots=True)
class BenchmarkResult(object):
    instance_name: str  # Name of the instance file (without extension)
    num_items: int  # Number of items in the instance
    bin_capacity: int  # Capacity of each bin
    bins_used: int  # Number of bins used in the solution
    elapsed_time: float  # Wall-clock solve time in seconds
    method: str  # Solving method used


class Benchmark(object):
    """Benchmarks BinPacking instances loaded from text files.

    Instance file format:
        Line 1: number of items
        Line 2: bin capacity
        Lines 3+: one item size per line
    """

    def __init__(self, instances_dir: str | Path) -> None:
        self._instances_dir: Path = Path(instances_dir)
        self._results: list[BenchmarkResult] = []

    @staticmethod
    def _parse_instance(filepath: Path) -> tuple[int, list[int]]:
        """Parse an instance file and return (bin_capacity, sizes)."""
        lines: list[str] = filepath.read_text().splitlines()
        num_items: int = int(lines[0])
        bin_capacity: int = int(lines[1])
        sizes: list[int] = [int(lines[i]) for i in range(2, 2 + num_items)]
        return bin_capacity, sizes

    def run(self, method: str = "b&b") -> None:
        """Run the benchmark on every *.txt instance in the instances directory."""
        if method != "b&b":
            raise ValueError("This solver version only supports method='b&b'.")

        instance_files: list[Path] = sorted(self._instances_dir.glob("*.txt"))
        if not instance_files:
            raise FileNotFoundError(
                f"No instance files found in '{self._instances_dir}'."
            )

        self._results.clear()
        for filepath in instance_files:
            self._results.append(self._run_single(filepath, method))

    def run_instance(
        self, filepath: str | Path, method: str = "b&b"
    ) -> BenchmarkResult:
        """Run the benchmark on a single instance file and append the result."""
        if method != "b&b":
            raise ValueError("This solver version only supports method='b&b'.")

        result: BenchmarkResult = self._run_single(Path(filepath), method)
        self._results.append(result)
        return result

    def _run_single(self, filepath: Path, method: str) -> BenchmarkResult:
        """Parse, solve, and time a single instance."""
        bin_capacity: int
        sizes: list[int]
        bin_capacity, sizes = self._parse_instance(filepath)

        solver: BinPacking = BinPacking(sizes, bin_capacity)

        start: float = time.perf_counter()
        solver.solve()
        elapsed: float = time.perf_counter() - start

        solution: Solution = solver.get_solution()
        return BenchmarkResult(
            instance_name=filepath.stem,
            num_items=len(sizes),
            bin_capacity=bin_capacity,
            bins_used=solution.bins_used,
            elapsed_time=elapsed,
            method=method,
        )

    def get_results(self) -> list[BenchmarkResult]:
        """Return a copy of all collected benchmark results."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()

    def print_summary(self) -> None:
        """Print a formatted summary table of all benchmark results."""
        if not self._results:
            print("No results available. Run the benchmark first.")
            return

        # Column widths
        name_width: int = max(len(r.instance_name) for r in self._results)
        name_width = max(name_width, len("Instance"))

        header: str = (
            f"{'Instance':<{name_width}}  "
            f"{'Items':>5}  "
            f"{'Capacity':>8}  "
            f"{'Bins':>4}  "
            f"{'Time (s)':>10}  "
            f"{'Method'}"
        )
        separator: str = "-" * len(header)

        print(separator)
        print(header)
        print(separator)

        for result in self._results:
            print(
                f"{result.instance_name:<{name_width}}  "
                f"{result.num_items:>5}  "
                f"{result.bin_capacity:>8}  "
                f"{result.bins_used:>4}  "
                f"{result.elapsed_time:>10.4f}  "
                f"{result.method}"
            )

        print(separator)

        # Aggregate statistics
        total_time: float = sum(r.elapsed_time for r in self._results)
        avg_bins: float = sum(r.bins_used for r in self._results) / len(self._results)
        print(f"Instances : {len(self._results)}")
        print(f"Total time: {total_time:.4f} s")
        print(f"Avg bins  : {avg_bins:.2f}")


if __name__ == "__main__":
    print("Running benchmark on all instances in 'benchmarks'...")
    instances_dir: Path = Path(__file__).parent.parent / "benchmarks"
    bench: Benchmark = Benchmark(instances_dir)
    bench.run(method="b&b")
    bench.print_summary()
