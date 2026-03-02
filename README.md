# Optimization Labs Repository
## Bin Packing Problem
### Commands to run the benchmark
All commands must be run from the **project root**.

The benchmark runner supports multiple datasets selected via `--dataset`. The Falkenauer and Scholl families are built in.

```shell
# Falkenauer T — all instances
python3 -m bin_packing.benchmark --dataset falkenauer-t

# Falkenauer U — only 120-item instances
python3 -m bin_packing.benchmark --dataset falkenauer-u --num-items 120

# Falkenauer T — all instances up to 249 items
python3 -m bin_packing.benchmark --dataset falkenauer-t --max-items 249

# Scholl 1 — all instances, with a 10-second per-instance time limit
python3 -m bin_packing.benchmark --dataset scholl-1 --time-limit 10

# Scholl 2 — backtracking method, skip graph generation
python3 -m bin_packing.benchmark --dataset scholl-2 --method backtracking --no-graphs

# Scholl 3 — branch and bound, no time limit
python3 -m bin_packing.benchmark --dataset scholl-3
```

#### Options
| Flag           | Values                                                                           | Default            | Description                                                        |
| -------------- | -------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------ |
| `--dataset`    | `falkenauer-t`, `falkenauer-u`, `scholl-1`, `scholl-2`, `scholl-3`              | *(required)*       | Dataset to benchmark                                               |
| `--method`     | `branch and bound`, `backtracking`, `dynamic programming`                        | `branch and bound` | Solving algorithm (`dynamic programming` limited to n ≤ 20 items)  |
| `--num-items`  | integer                                                                          | —                  | Run only instances with **exactly** N items                        |
| `--max-items`  | integer                                                                          | —                  | Run only instances with **at most** N items                        |
| `--time-limit` | float                                                                            | —                  | Maximum solve time (in seconds) per instance                       |
| `--no-graphs`  | flag                                                                             | off                | Skip graph generation after the run                                |

`--num-items` and `--max-items` are mutually exclusive.

#### Datasets
| Key             | Label        | Source directory                         | Format          |
| --------------- | ------------ | ---------------------------------------- | --------------- |
| `falkenauer-t`  | Falkenauer T | `benchmarks/Falkenauer/Falkenauer_T/`    | Standard (see below) |
| `falkenauer-u`  | Falkenauer U | `benchmarks/Falkenauer/Falkenauer U/`    | Standard        |
| `scholl-1`      | Scholl 1     | `benchmarks/Scholl/Scholl_1/`            | Standard        |
| `scholl-2`      | Scholl 2     | `benchmarks/Scholl/Scholl_2/`            | Standard        |
| `scholl-3`      | Scholl 3     | `benchmarks/Scholl/Scholl_3/`            | Standard        |

**Standard file format** (one instance per `.txt` file):
```
<number of items>
<bin capacity>
<item size 1>
<item size 2>
...
```

#### Extending to new datasets
Register a new dataset by adding a `DatasetConfig` and calling `register_dataset()` anywhere before `Benchmark.run()` is invoked — typically at the top of `benchmark.py` alongside the built-in entries:

```python
from bin_packing.benchmark import DatasetConfig, parse_standard, register_dataset

register_dataset(DatasetConfig(
    key="my-dataset",
    label="My Custom Dataset",
    directory=Path("benchmarks/MyDataset"),
    parser=parse_standard,   # or a custom Callable[[Path, str], BenchmarkInstance]
    glob="*.txt",
))
```

If the file format differs from the standard, supply a custom `parser` function with the signature `(filepath: Path, dataset_key: str) -> BenchmarkInstance`.

#### Output
Results are printed as a table with columns: `Instance`, `Items`, `Capacity`, `LB` (theoretical lower bound), `Bins`, `Gap` (bins − LB), `Time (s)`, `Method`, and `State` (`Done` or `T.O.` for timeouts).

A summary block reports the number of instances processed, provably optimal solutions (gap = 0), total and average solve time, average bins used, and the number of timeouts (if any).

Instances that raise an error (e.g. `dynamic programming` requested for n > 20) are skipped with a warning; the remainder of the run continues normally.

Unless `--no-graphs` is passed, four PNG graphs are saved to `bin_packing/results/`, each suffixed with the dataset key:

| File                                   | Description                                                    |
| -------------------------------------- | -------------------------------------------------------------- |
| `fig1_solve_times_<dataset-key>.png`   | Solve time per instance, slowest highlighted in red            |
| `fig2_bins_vs_lb_<dataset-key>.png`    | Bins used vs lower bound, with gap highlighted in red          |
| `fig3_fill_rate_<dataset-key>.png`     | Average bin fill rate per instance on a red-yellow-green scale |
| `fig4_time_by_size_<dataset-key>.png`  | Box plot of solve times grouped by instance size (omitted when all instances share the same size) |
