from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass(slots=True)
class ResultRow:
    instance_name: str
    dataset_key: str
    num_items: int
    bin_capacity: int
    bins_used: int
    lower_bound: int
    total_weight: int
    elapsed_time: float
    method: str
    timed_out: bool

    @property
    def gap(self) -> int:
        return self.bins_used - self.lower_bound

    @property
    def fill_rate(self) -> float:
        return self.total_weight / (self.bins_used * self.bin_capacity) * 100


def _read_dict_rows(csv_path: str | Path):
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _validate_columns(fieldnames: list[str] | None) -> None:
    required = {
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
    }
    missing = required.difference(fieldnames or [])
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")


def _parse_result_row(row: dict[str, str]) -> ResultRow:
    return ResultRow(
        instance_name=row["instance_name"],
        dataset_key=row["dataset_key"],
        num_items=int(row["num_items"]),
        bin_capacity=int(row["bin_capacity"]),
        bins_used=int(row["bins_used"]),
        lower_bound=int(row["lower_bound"]),
        total_weight=int(row["total_weight"]),
        elapsed_time=float(row["elapsed_time"]),
        method=row["method"],
        timed_out=str(row["timed_out"]).lower() == "true",
    )


def _load_results_granular(csv_path: str | Path) -> list[ResultRow]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _validate_columns(reader.fieldnames)
        return [_parse_result_row(row) for row in reader]


def _load_results_monolithic(csv_path: str | Path) -> list[ResultRow]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
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
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")

        rows: list[ResultRow] = []
        for row in reader:
            rows.append(
                ResultRow(
                    instance_name=row["instance_name"],
                    dataset_key=row["dataset_key"],
                    num_items=int(row["num_items"]),
                    bin_capacity=int(row["bin_capacity"]),
                    bins_used=int(row["bins_used"]),
                    lower_bound=int(row["lower_bound"]),
                    total_weight=int(row["total_weight"]),
                    elapsed_time=float(row["elapsed_time"]),
                    method=row["method"],
                    timed_out=str(row["timed_out"]).lower() == "true",
                )
            )
    return rows


def load_results(csv_path: str | Path) -> list[ResultRow]:
    return _load_results_granular(csv_path)


def _split_completed(rows: list[ResultRow]) -> tuple[list[ResultRow], list[ResultRow]]:
    completed = [r for r in rows if not r.timed_out]
    timed_out = [r for r in rows if r.timed_out]
    return completed, timed_out


def _base_summary(rows: list[ResultRow], completed: list[ResultRow], timed_out: list[ResultRow]) -> dict[str, float | int]:
    return {
        "instances": len(rows),
        "completed": len(completed),
        "timeouts": len(timed_out),
        "total_time_s": sum(r.elapsed_time for r in rows),
    }


def _completed_summary(completed: list[ResultRow]) -> dict[str, float | int]:
    gaps = [r.gap for r in completed]
    return {
        "avg_time_completed_s": mean(r.elapsed_time for r in completed),
        "median_time_completed_s": median(r.elapsed_time for r in completed),
        "avg_bins_completed": mean(r.bins_used for r in completed),
        "avg_gap_completed": mean(gaps),
        "max_gap_completed": max(gaps),
        "avg_fill_rate_completed_pct": mean(r.fill_rate for r in completed),
    }


def _summarize_granular(rows: list[ResultRow]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("No benchmark rows loaded.")
    completed, timed_out = _split_completed(rows)
    out = _base_summary(rows, completed, timed_out)
    if completed:
        out.update(_completed_summary(completed))
    return out


def _summarize_monolithic(rows: list[ResultRow]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("No benchmark rows loaded.")

    completed = [r for r in rows if not r.timed_out]
    elapsed = [r.elapsed_time for r in rows]
    gaps = [r.gap for r in completed] if completed else []

    out: dict[str, float | int] = {
        "instances": len(rows),
        "completed": len(completed),
        "timeouts": len(rows) - len(completed),
        "total_time_s": sum(elapsed),
    }
    if completed:
        out.update(
            {
                "avg_time_completed_s": mean(r.elapsed_time for r in completed),
                "median_time_completed_s": median(r.elapsed_time for r in completed),
                "avg_bins_completed": mean(r.bins_used for r in completed),
                "avg_gap_completed": mean(gaps),
                "max_gap_completed": max(gaps),
                "avg_fill_rate_completed_pct": mean(r.fill_rate for r in completed),
            }
        )
    return out


def summarize(rows: list[ResultRow]) -> dict[str, float | int]:
    return _summarize_granular(rows)


def summarize_by_size(rows: list[ResultRow]) -> list[dict[str, float | int]]:
    groups: dict[int, list[ResultRow]] = {}
    for r in rows:
        groups.setdefault(r.num_items, []).append(r)

    results: list[dict[str, float | int]] = []
    for n in sorted(groups):
        chunk = groups[n]
        completed = [r for r in chunk if not r.timed_out]
        results.append(
            {
                "num_items": n,
                "instances": len(chunk),
                "completed": len(completed),
                "timeouts": len(chunk) - len(completed),
                "avg_time_s": (
                    mean(r.elapsed_time for r in completed) if completed else 0.0
                ),
                "avg_gap": mean(r.gap for r in completed) if completed else 0.0,
            }
        )
    return results


def _print_summary(
    summary: dict[str, float | int], by_size: list[dict[str, float | int]]
) -> None:
    print("\n=== Benchmark Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value:.4f}")
        else:
            print(f"{key:28s}: {value}")

    print("\n=== By num_items ===")
    for row in by_size:
        print(
            f"n={row['num_items']:4d} | instances={row['instances']:3d} | "
            f"completed={row['completed']:3d} | timeouts={row['timeouts']:3d} | "
            f"avg_time={row['avg_time_s']:.4f}s | avg_gap={row['avg_gap']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute statistics from benchmark CSV output."
    )
    parser.add_argument("csv", help="Path to benchmark CSV file.")
    args = parser.parse_args()

    rows = load_results(args.csv)
    _print_summary(summarize(rows), summarize_by_size(rows))
