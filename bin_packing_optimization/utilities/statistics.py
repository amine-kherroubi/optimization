from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence, Dict


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


@dataclass(slots=True)
class SizeSummary:
    """Per-size-bucket summary returned by summarize_by_size."""

    num_items: int
    instances: int
    completed: int
    timeouts: int
    avg_time_s: float
    avg_gap: float


def _validate_columns(fieldnames: Iterable[str] | None) -> None:
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


def load_results(csv_path: str | Path) -> list[ResultRow]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _validate_columns(reader.fieldnames)
        return [_parse_result_row(row) for row in reader]


def _split_completed(rows: list[ResultRow]) -> tuple[list[ResultRow], list[ResultRow]]:
    completed = [r for r in rows if not r.timed_out]
    timed_out = [r for r in rows if r.timed_out]
    return completed, timed_out


def _base_summary(
    rows: list[ResultRow], completed: list[ResultRow], timed_out: list[ResultRow]
) -> dict[str, float | int]:
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


def summarize(rows: list[ResultRow]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("No benchmark rows loaded.")
    completed, timed_out = _split_completed(rows)
    out = _base_summary(rows, completed, timed_out)
    if completed:
        out.update(_completed_summary(completed))
    return out


def summarize_by_size(rows: list[ResultRow]) -> list[SizeSummary]:
    groups: dict[int, list[ResultRow]] = {}
    for r in rows:
        groups.setdefault(r.num_items, []).append(r)

    results: list[SizeSummary] = []
    for n in sorted(groups):
        chunk = groups[n]
        completed = [r for r in chunk if not r.timed_out]
        results.append(
            SizeSummary(
                num_items=n,
                instances=len(chunk),
                completed=len(completed),
                timeouts=len(chunk) - len(completed),
                avg_time_s=(
                    mean(r.elapsed_time for r in completed) if completed else 0.0
                ),
                avg_gap=mean(r.gap for r in completed) if completed else 0.0,
            )
        )
    return results


def _print_summary(summary: dict[str, float | int], by_size: list[SizeSummary]) -> None:
    print("\n=== Benchmark Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value:.4f}")
        else:
            print(f"{key:28s}: {value}")

    print("\n=== By num_items ===")
    for row in by_size:
        print(
            f"n={row.num_items:4d} | instances={row.instances:3d} | "
            f"completed={row.completed:3d} | timeouts={row.timeouts:3d} | "
            f"avg_time={row.avg_time_s:.4f}s | avg_gap={row.avg_gap:.3f}"
        )


# Project root (used for sensible default output locations)
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def load_results_grouped(csv_paths: Sequence[str | Path]) -> Dict[str, list[ResultRow]]:
    """Load multiple CSV files and return a mapping from file-stem to rows.

    This is convenient for comparative analyses across many CSV outputs.
    """
    out: Dict[str, list[ResultRow]] = {}
    for p in csv_paths:
        key = Path(p).stem
        out[key] = load_results(p)
    return out


def summarize_by_method(rows: list[ResultRow]) -> dict[str, dict[str, float | int]]:
    """Compute the same summary statistics as :func:`summarize`, but grouped by `method`.

    Returns a mapping method -> summary-dict.
    """
    groups: dict[str, list[ResultRow]] = {}
    for r in rows:
        groups.setdefault(r.method, []).append(r)

    out: dict[str, dict[str, float | int]] = {}
    for method, grp in groups.items():
        completed, timed_out = _split_completed(grp)
        s = _base_summary(grp, completed, timed_out)
        if completed:
            s.update(_completed_summary(completed))
        out[method] = s
    return out


def summarize_multiple(csv_paths: Sequence[str | Path]) -> Dict[str, dict]:
    """Summarize multiple CSV files; returns a mapping file-stem -> summary."""
    out: Dict[str, dict] = {}
    for p in csv_paths:
        try:
            rows = load_results(p)
            out[Path(p).stem] = summarize(rows)
        except Exception as exc:  # keep failures local to that file
            out[Path(p).stem] = {"error": str(exc)}
    return out


def export_summary_csv(
    summary: dict[str, dict],
    out_dir: str | Path | None = None,
    filename: str | None = None,
    timestamp: bool = True,
) -> Path:
    """Write a dictionary-of-dicts summary to a CSV file.

    The first column will be the group key (e.g. method or filename) and the
    remaining columns are the union of keys found in the inner dictionaries.
    """
    out_dir = Path(out_dir) if out_dir else _PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
        filename = f"summary{('_' + ts) if ts else ''}.csv"
    path = out_dir / filename

    # union of inner keys
    extra_fields: set[str] = set()
    for v in summary.values():
        if isinstance(v, dict):
            extra_fields.update(v.keys())

    fieldnames = ["group"] + sorted(extra_fields)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, metrics in summary.items():
            row: dict[str, object] = {"group": group}
            if isinstance(metrics, dict):
                for k in extra_fields:
                    v = metrics.get(k)
                    row[k] = "" if v is None else v
            else:
                row["value"] = metrics
            writer.writerow(row)
    return path


def print_benchmark_report(csv_path: str | Path) -> tuple[dict[str, float | int], list[SizeSummary]]:
    """Load a benchmark CSV and print overall + per-size summaries."""
    rows = load_results(csv_path)
    summary = summarize(rows)
    by_size = summarize_by_size(rows)

    print("Overall")
    for key, value in summary.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:<30s}: {formatted}")

    print("\nBy instance size")
    for row in by_size:
        print(
            f"  n={row.num_items:4d} | completed={row.completed}/{row.instances}"
            f" | avg_time={row.avg_time_s:.4f}s | avg_gap={row.avg_gap:.3f}"
        )

    return summary, by_size
