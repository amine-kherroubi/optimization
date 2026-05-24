from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from bin_packing_optimization.population_based_metaheuristics.solver import BinPackingSolver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = PROJECT_ROOT / "datasets"
RESULTS_DIR = (
    PROJECT_ROOT / "results" / "4_population_based_metaheuristics" / "aco_tuning"
)

ACO_ROUNDS = 1

ACO_GRID = {
    "num_ants": [10, 16, 24, 32],
    "generations": [40, 80, 120, 200],
    "alpha": [0.5, 1.0, 1.5, 2.0],
    "beta": [2.0, 3.0, 4.0, 5.0],
    "evaporation_rate": [0.10, 0.20, 0.35, 0.50],
    "elite_ants": [1, 2, 3, 5],
    "local_search_steps": [0, 20, 40, 80],
    "deposit_strength": [100.0, 500.0, 1000.0, 5000.0],
}

ACO_DEFAULTS = {
    "num_ants": 16,
    "generations": 80,
    "alpha": 1.0,
    "beta": 3.0,
    "evaporation_rate": 0.20,
    "pheromone_init": 1.0,
    "pheromone_min": 0.01,
    "pheromone_max": 6.0,
    "elite_ants": 3,
    "local_search_steps": 40,
    "deposit_strength": 1000.0,
}

DATASET_PATHS = {
    "falkenauer-t": BENCHMARKS_ROOT / "Falkenauer" / "Falkenauer_T",
    "falkenauer-u": BENCHMARKS_ROOT / "Falkenauer" / "Falkenauer U",
    "scholl-1": BENCHMARKS_ROOT / "Scholl" / "Scholl_1",
    "scholl-2": BENCHMARKS_ROOT / "Scholl" / "Scholl_2",
    "scholl-3": BENCHMARKS_ROOT / "Scholl" / "Scholl_3",
}

PARAM_NAMES = list(ACO_DEFAULTS)
TUNED_PARAM_NAMES = list(ACO_GRID)


@dataclass(frozen=True, slots=True)
class Instance:
    name: str
    dataset: str
    path: Path
    num_items: int
    bin_capacity: int
    sizes: list[int]

    @property
    def lower_bound(self) -> int:
        return ceil(sum(self.sizes) / self.bin_capacity)


def parse_instance(path: Path, dataset: str) -> Instance:
    lines = path.read_text(encoding="utf-8").splitlines()
    num_items = int(lines[0])
    bin_capacity = int(lines[1])
    sizes = [int(lines[i]) for i in range(2, 2 + num_items)]
    return Instance(path.stem, dataset, path, num_items, bin_capacity, sizes)


def load_instances(
    dataset_keys: list[str],
    *,
    max_items: int | None,
    max_per_size: int,
) -> list[Instance]:
    selected: list[Instance] = []

    for dataset in dataset_keys:
        instances_by_size: dict[int, list[Instance]] = {}
        for path in sorted(DATASET_PATHS[dataset].glob("*.txt")):
            instance = parse_instance(path, dataset)
            if max_items is not None and instance.num_items > max_items:
                continue
            instances_by_size.setdefault(instance.num_items, []).append(instance)

        for size in sorted(instances_by_size):
            selected.extend(instances_by_size[size][:max_per_size])

    return selected


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_aco(
    instance: Instance,
    seed: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    random.seed(seed)

    start = time.perf_counter()
    solver = BinPackingSolver(instance.sizes, instance.bin_capacity)
    solver.solve("aco", **params)
    elapsed = time.perf_counter() - start

    solution = solver.get_solution()
    gap = solution.total_bins_used - instance.lower_bound

    return {
        "dataset": instance.dataset,
        "instance": instance.name,
        "items": instance.num_items,
        "seed": seed,
        "lower_bound": instance.lower_bound,
        "bins": solution.total_bins_used,
        "gap": gap,
        "elapsed_time": elapsed,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty run list.")

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    dataset_avg_gaps = [
        mean([float(row["gap"]) for row in dataset_rows])
        for dataset_rows in by_dataset.values()
    ]
    dataset_avg_times = [
        mean([float(row["elapsed_time"]) for row in dataset_rows])
        for dataset_rows in by_dataset.values()
    ]
    gaps = [int(row["gap"]) for row in rows]

    return {
        "runs": len(rows),
        "avg_gap": mean(dataset_avg_gaps),
        "worst_dataset_avg_gap": max(dataset_avg_gaps),
        "best_gap": min(gaps),
        "worst_gap": max(gaps),
        "exact_hits": sum(1 for gap in gaps if gap == 0),
        "avg_time": mean(dataset_avg_times),
    }


def summary_key(summary: dict[str, Any]) -> tuple[float, float, int, float]:
    return (
        float(summary["avg_gap"]),
        float(summary["worst_dataset_avg_gap"]),
        -int(summary["exact_hits"]),
        float(summary["avg_time"]),
    )


def evaluate_params(
    params: dict[str, Any],
    instances: list[Instance],
    seeds: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for seed in seeds:
            rows.append(run_aco(instance, seed, params))
    return summarize_runs(rows), rows


def format_summary(summary: dict[str, Any]) -> str:
    return (
        f"avg_gap={float(summary['avg_gap']):.3f}, "
        f"worst_dataset_gap={float(summary['worst_dataset_avg_gap']):.3f}, "
        f"exact={int(summary['exact_hits'])}/{int(summary['runs'])}, "
        f"avg_time={float(summary['avg_time']):.3f}s"
    )


def log_row(
    writer: csv.DictWriter,
    *,
    label: str,
    round_index: int,
    parameter: str,
    value: Any,
    params: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    row = {
        "label": label,
        "round": round_index,
        "parameter": parameter,
        "value": value,
        **summary,
        **params,
    }
    writer.writerow(row)


def tune_one_by_one(
    *,
    label: str,
    base_params: dict[str, Any],
    grid: dict[str, list[Any]],
    instances: list[Instance],
    seeds: list[int],
    rounds: int,
    writer: csv.DictWriter,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    best_params = dict(base_params)
    best_summary, best_rows = evaluate_params(best_params, instances, seeds)

    print(f"\n{label}")
    print("-" * len(label))
    print(f"start: {format_summary(best_summary)}")
    log_row(
        writer,
        label=label,
        round_index=0,
        parameter="initial",
        value="",
        params=best_params,
        summary=best_summary,
    )

    for round_index in range(1, rounds + 1):
        print(f"\nround {round_index}/{rounds}")
        for parameter, values in grid.items():
            candidates: list[
                tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]
            ] = []

            print(f"  tuning {parameter}")
            for value in values:
                candidate_params = dict(best_params)
                candidate_params[parameter] = value
                summary, rows = evaluate_params(candidate_params, instances, seeds)
                candidates.append((summary, candidate_params, rows))
                log_row(
                    writer,
                    label=label,
                    round_index=round_index,
                    parameter=parameter,
                    value=value,
                    params=candidate_params,
                    summary=summary,
                )
                print(f"    {parameter}={value}: {format_summary(summary)}")

            best_summary, best_params, best_rows = min(
                candidates,
                key=lambda item: summary_key(item[0]),
            )
            print(f"  selected {parameter}={best_params[parameter]}")

    return best_params, best_summary, best_rows


def group_by_dataset(instances: list[Instance]) -> dict[str, list[Instance]]:
    grouped: dict[str, list[Instance]] = {}
    for instance in instances:
        grouped.setdefault(instance.dataset, []).append(instance)
    return grouped


def print_dataset_summaries(rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    print("\nFinal parameters by dataset:")
    for dataset in sorted(by_dataset):
        summary = summarize_runs(by_dataset[dataset])
        print(f"- {dataset}: {format_summary(summary)}")


def selected_grid(parameters: list[str] | None) -> dict[str, list[Any]]:
    if parameters is None:
        return ACO_GRID
    return {parameter: ACO_GRID[parameter] for parameter in parameters}


@dataclass(frozen=True, slots=True)
class TuningOptions:
    datasets: list[str]
    max_items: int | None
    max_per_size: int
    seeds: list[int]
    rounds: int
    grid: dict[str, list[Any]]


@dataclass(frozen=True, slots=True)
class TuningJob:
    label: str
    datasets: list[str]
    output: Path


class AcoTuner:
    def __init__(self, options: TuningOptions):
        self._options = options

    def run_job(self, job: TuningJob) -> float:
        start = time.perf_counter()
        instances = load_instances(
            job.datasets,
            max_items=self._options.max_items,
            max_per_size=self._options.max_per_size,
        )
        if not instances:
            raise RuntimeError(
                f"No benchmark instances matched the filters for {job.label}."
            )

        job.output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label",
            "round",
            "parameter",
            "value",
            "runs",
            "avg_gap",
            "worst_dataset_avg_gap",
            "best_gap",
            "worst_gap",
            "exact_hits",
            "avg_time",
            *PARAM_NAMES,
        ]

        print("\n" + "=" * 80)
        print(f"Job: {job.label}")
        print(f"Datasets: {', '.join(job.datasets)}")
        print(f"Instances: {len(instances)}")
        print(f"Seeds: {', '.join(str(seed) for seed in self._options.seeds)}")
        print(f"Tuned parameters: {len(self._options.grid)}")
        print(f"Output: {display_path(job.output)}")
        print("=" * 80, flush=True)

        with job.output.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            best_params, best_summary, best_rows = tune_one_by_one(
                label=job.label,
                base_params=ACO_DEFAULTS,
                grid=self._options.grid,
                instances=instances,
                seeds=self._options.seeds,
                rounds=self._options.rounds,
                writer=writer,
            )

        print(f"\nBest params for {job.label}:")
        print(best_params)
        print(format_summary(best_summary))
        print_dataset_summaries(best_rows)

        elapsed = time.perf_counter() - start
        print(f"Finished {job.label} in {elapsed / 60:.2f} minutes.", flush=True)
        return elapsed


class AcoTuningSuite:
    def __init__(self, options: TuningOptions, output_dir: Path):
        self._options = options
        self._output_dir = output_dir

    def _jobs(self) -> list[TuningJob]:
        jobs = [
            TuningJob(
                label="ACO/global",
                datasets=self._options.datasets,
                output=self._output_dir / "aco_tuning_global.csv",
            )
        ]

        for dataset in self._options.datasets:
            jobs.append(
                TuningJob(
                    label=f"ACO/{dataset}",
                    datasets=[dataset],
                    output=self._output_dir
                    / f"aco_tuning_{dataset.replace('-', '_')}.csv",
                )
            )

        return jobs

    def run(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        jobs = self._jobs()
        total_start = time.perf_counter()

        print("ACO tuning suite")
        print(f"Jobs: {len(jobs)}")
        print(f"Datasets: {', '.join(self._options.datasets)}")
        print(f"Output directory: {display_path(self._output_dir)}")

        tuner = AcoTuner(self._options)
        for job in jobs:
            tuner.run_job(job)

        total_elapsed = time.perf_counter() - total_start
        print("\nAll ACO tuning jobs finished.")
        print(f"Total time: {total_elapsed / 60:.2f} minutes.")
        print("Generated files:")
        for job in jobs:
            print(f"- {display_path(job.output)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tune ACO parameters. By default, runs a full suite: global tuning "
            "plus one tuning job per selected dataset."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_PATHS),
        default=sorted(DATASET_PATHS),
        help="Datasets to sample.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=250,
        help="Skip instances above this item count. Use 0 for no limit.",
    )
    parser.add_argument(
        "--max-per-size",
        type=int,
        default=1,
        help="Instances sampled per dataset and item-count group.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=ACO_ROUNDS,
        help="One-by-one tuning rounds.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=TUNED_PARAM_NAMES,
        default=None,
        help="Optional subset of parameters to tune.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run one tuning job only, using --datasets, --output, and --label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "aco_tuning.csv",
        help="CSV output path for --single mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for the default full suite.",
    )
    parser.add_argument(
        "--label",
        default="ACO/global",
        help="Label written to the tuning log in --single mode.",
    )
    args = parser.parse_args()

    if args.max_per_size <= 0:
        raise ValueError("max-per-size must be positive.")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive.")

    max_items = None if args.max_items == 0 else args.max_items
    options = TuningOptions(
        args.datasets,
        max_items,
        args.max_per_size,
        parse_seeds(args.seeds),
        args.rounds,
        selected_grid(args.parameters),
    )

    if args.single:
        AcoTuner(options).run_job(TuningJob(args.label, args.datasets, args.output))
    else:
        AcoTuningSuite(options, args.output_dir).run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
