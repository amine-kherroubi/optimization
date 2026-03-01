import time
from dataclasses import dataclass
from pathlib import Path

# from bin_packing.slow import BinPacking, Solution
from bin_packing.fast import BinPacking, Solution


@dataclass(slots=True)
class BenchmarkResult:
    instance_name: str
    num_items: int
    bin_capacity: int
    bins_used: int
    elapsed_time: float  # seconds
    method: str


class Benchmark:
    """Benchmarks BinPacking instances from text files.

    Instance file format:
        Line 1: number of items
        Line 2: bin capacity
        Lines 3+: one item size per line
    """

    def __init__(self, instances_dir: str | Path) -> None:
        self._instances_dir: Path = Path(instances_dir)
        self._results: list[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_instance(filepath: Path) -> tuple[int, list[int]]:
        """Return (bin_capacity, sizes) parsed from an instance file."""
        lines = filepath.read_text().splitlines()
        num_items = int(lines[0])
        bin_capacity = int(lines[1])
        sizes = [int(lines[i]) for i in range(2, 2 + num_items)]
        return bin_capacity, sizes

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(self, method: str = "b&b") -> None:
        """Run the benchmark on every *.txt instance in the instances directory."""
        if method != "b&b":
            raise ValueError("This solver version only supports method='b&b'.")

        instance_files = sorted(self._instances_dir.glob("*.txt"))
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

        result = self._run_single(Path(filepath), method)
        self._results.append(result)
        return result

    def _run_single(self, filepath: Path, method: str) -> BenchmarkResult:
        """Parse, solve, and time one instance."""
        bin_capacity, sizes = self._parse_instance(filepath)
        solver = BinPacking(sizes, bin_capacity)

        start = time.perf_counter()
        solver.solve()  # updated: new solver has no method parameter
        elapsed = time.perf_counter() - start

        solution: Solution = solver.get_solution()
        return BenchmarkResult(
            instance_name=filepath.stem,
            num_items=len(sizes),
            bin_capacity=bin_capacity,
            bins_used=solution.bins_used,
            elapsed_time=elapsed,
            method=method,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> list[BenchmarkResult]:
        """Return a copy of all collected benchmark results."""
        return list(self._results)

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted summary table of all benchmark results."""
        if not self._results:
            print("No results available. Run the benchmark first.")
            return

        name_w = max(len(r.instance_name) for r in self._results)
        name_w = max(name_w, len("Instance"))

        header = (
            f"{'Instance':<{name_w}}  "
            f"{'Items':>5}  "
            f"{'Capacity':>8}  "
            f"{'Bins':>4}  "
            f"{'Time (s)':>10}  "
            f"{'Method'}"
        )
        sep = "-" * len(header)

        print(sep)
        print(header)
        print(sep)

        for r in self._results:
            print(
                f"{r.instance_name:<{name_w}}  "
                f"{r.num_items:>5}  "
                f"{r.bin_capacity:>8}  "
                f"{r.bins_used:>4}  "
                f"{r.elapsed_time:>10.4f}  "
                f"{r.method}"
            )

        print(sep)
        total_time = sum(r.elapsed_time for r in self._results)
        avg_bins = sum(r.bins_used for r in self._results) / len(self._results)
        print(f"Instances : {len(self._results)}")
        print(f"Total time: {total_time:.4f} s")
        print(f"Avg bins  : {avg_bins:.2f}")


if __name__ == "__main__":
    print("Running benchmark on all instances in 'Benchmarks'...")
    instances_dir = Path(__file__).parent.parent / "benchmarks"
    bench = Benchmark(instances_dir)
    bench.run(method="b&b")
    bench.print_summary()
