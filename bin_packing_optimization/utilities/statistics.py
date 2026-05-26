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


def _fmt_float(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _print_section(title: str) -> None:
    line = "═" * 90
    print(f"\n\033[1;36m{line}\033[0m")
    print(f"\033[1;36m{title:^90}\033[0m")
    print(f"\033[1;36m{line}\033[0m")


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


def load_results_many(csv_paths: Sequence[str | Path]) -> list[ResultRow]:
    """Load and concatenate rows from many benchmark result CSV files."""
    rows: list[ResultRow] = []
    for p in csv_paths:
        rows.extend(load_results(p))
    return rows


def filter_rows_by_items(
    rows: Sequence[ResultRow],
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> list[ResultRow]:
    """Filter rows by instance size.

    `num_items` is exact-match and mutually exclusive with (`min_items`, `max_items`).
    """
    if num_items is not None and (min_items is not None or max_items is not None):
        raise ValueError("num_items is mutually exclusive with min_items/max_items.")
    if min_items is not None and max_items is not None and min_items > max_items:
        raise ValueError("min_items cannot be greater than max_items.")

    out: list[ResultRow] = []
    for row in rows:
        if num_items is not None and row.num_items != num_items:
            continue
        if min_items is not None and row.num_items < min_items:
            continue
        if max_items is not None and row.num_items > max_items:
            continue
        out.append(row)
    return out


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
    times = [r.elapsed_time for r in completed]
    fill_rates = [r.fill_rate for r in completed]
    return {
        "avg_time_completed_s": mean(r.elapsed_time for r in completed),
        "median_time_completed_s": median(r.elapsed_time for r in completed),
        "min_time_completed_s": min(times),
        "max_time_completed_s": max(times),
        "avg_bins_completed": mean(r.bins_used for r in completed),
        "avg_gap_completed": mean(gaps),
        "median_gap_completed": median(gaps),
        "min_gap_completed": min(gaps),
        "max_gap_completed": max(gaps),
        "avg_fill_rate_completed_pct": mean(fill_rates),
        "median_fill_rate_completed_pct": median(fill_rates),
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
    _print_section("BENCHMARK SUMMARY")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:34s}: {_fmt_float(value)}")
        else:
            print(f"{key:34s}: {value}")

    _print_section("SUMMARY BY NUM_ITEMS")
    print(
        f"{'n':>6} │ {'inst':>6} │ {'done':>6} │ {'tout':>6} │ {'avg_time(s)':>14} │ {'avg_gap':>10}"
    )
    print("─" * 90)
    for row in by_size:
        print(
            f"{row.num_items:>6d} │ {row.instances:>6d} │ {row.completed:>6d} │ {row.timeouts:>6d} │ "
            f"{_fmt_float(row.avg_time_s):>14} │ {row.avg_gap:>10.3f}"
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


def summarize_multiple_by_dataset(
    csv_paths: Sequence[str | Path],
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """Summarize many CSV files and aggregate rows by dataset key.

    Unlike :func:`summarize_multiple` (grouped by file), this function merges
    rows from all inputs and returns one summary per ``dataset_key``.
    """
    groups: dict[str, list[ResultRow]] = {}
    for p in csv_paths:
        rows = filter_rows_by_items(
            load_results(p),
            num_items=num_items,
            min_items=min_items,
            max_items=max_items,
        )
        for row in rows:
            groups.setdefault(row.dataset_key, []).append(row)

    return {dataset_key: summarize(rows) for dataset_key, rows in groups.items()}


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


def print_benchmark_report(
    csv_path: str | Path,
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> tuple[dict[str, float | int], list[SizeSummary]]:
    """Load a benchmark CSV and print overall + per-size summaries."""
    rows = filter_rows_by_items(
        load_results(csv_path),
        num_items=num_items,
        min_items=min_items,
        max_items=max_items,
    )
    summary = summarize(rows)
    by_size = summarize_by_size(rows)

    _print_section("OVERALL")
    for key, value in summary.items():
        formatted = _fmt_float(value) if isinstance(value, float) else str(value)
        print(f"  {key:<34s}: {formatted}")

    _print_section("BY INSTANCE SIZE")
    print(f"{'n':>6} │ {'done/total':>12} │ {'avg_time(s)':>14} │ {'avg_gap':>10}")
    print("─" * 60)
    for row in by_size:
        print(
            f"{row.num_items:>6d} │ {f'{row.completed}/{row.instances}':>12} │ "
            f"{_fmt_float(row.avg_time_s):>14} │ {row.avg_gap:>10.3f}"
        )

    return summary, by_size


def print_method_comparison_report(
    csv_paths: Sequence[str | Path],
) -> dict[str, dict[str, float | int]]:
    """Print one summary row per method from multiple benchmark CSV files."""
    if not csv_paths:
        raise ValueError("At least one CSV path must be provided.")

    summary = summarize_by_method(load_results_many(csv_paths))
    _print_section("METHOD COMPARISON")
    print(
        f"{'method':<18} │ {'done/total':>12} │ {'avg_bins':>10} │ "
        f"{'avg_gap':>10} │ {'avg_fill(%)':>12} │ {'avg_time(s)':>12}"
    )
    print("─" * 86)
    for method, metrics in summary.items():
        completed = int(metrics.get("completed", 0))
        instances = int(metrics.get("instances", 0))
        avg_bins = metrics.get("avg_bins_completed", 0.0)
        avg_gap = metrics.get("avg_gap_completed", 0.0)
        avg_fill = metrics.get("avg_fill_rate_completed_pct", 0.0)
        avg_time = metrics.get("avg_time_completed_s", 0.0)
        print(
            f"{method:<18} │ {f'{completed}/{instances}':>12} │ "
            f"{_fmt_float(float(avg_bins)):>10} │ "
            f"{_fmt_float(float(avg_gap)):>10} │ "
            f"{_fmt_float(float(avg_fill)):>12} │ "
            f"{_fmt_float(float(avg_time)):>12}"
        )

    return summary


def print_multi_benchmark_report(
    csv_paths: Sequence[str | Path],
    *,
    num_items: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
) -> tuple[dict[str, dict[str, float | int]], dict[str, list[SizeSummary]]]:
    """Print per-dataset summaries aggregated from many benchmark CSV files."""
    if not csv_paths:
        raise ValueError("At least one CSV path must be provided.")

    rows = filter_rows_by_items(
        load_results_many(csv_paths),
        num_items=num_items,
        min_items=min_items,
        max_items=max_items,
    )
    if not rows:
        raise ValueError("No benchmark rows loaded.")

    grouped: dict[str, list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault(row.dataset_key, []).append(row)

    summaries: dict[str, dict[str, float | int]] = {}
    by_size_map: dict[str, list[SizeSummary]] = {}
    for dataset_key in sorted(grouped):
        dataset_rows = grouped[dataset_key]
        summary = summarize(dataset_rows)
        by_size = summarize_by_size(dataset_rows)
        summaries[dataset_key] = summary
        by_size_map[dataset_key] = by_size

        _print_section(f"OVERALL — {dataset_key}")
        for key, value in summary.items():
            formatted = _fmt_float(value) if isinstance(value, float) else str(value)
            print(f"  {key:<34s}: {formatted}")

        _print_section(f"BY INSTANCE SIZE — {dataset_key}")
        print(f"{'n':>6} │ {'done/total':>12} │ {'avg_time(s)':>14} │ {'avg_gap':>10}")
        print("─" * 60)
        for item in by_size:
            print(
                f"{item.num_items:>6d} │ {f'{item.completed}/{item.instances}':>12} │ "
                f"{_fmt_float(item.avg_time_s):>14} │ {item.avg_gap:>10.3f}"
            )

    return summaries, by_size_map
