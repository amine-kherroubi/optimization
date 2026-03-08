# Optimization Labs — Bin Packing Problem

## Project structure

```
optimization/
├── benchmark.py
├── 1_exact/
│   └── solver.py          # Backtracking, Branch & Bound, Dynamic Programming
├── 2_heuristics/
│   └── solver.py          # Next Fit, First Fit, Best Fit
├── results/
│   ├── 1_exact/           # Graphs from exact runs
│   └── 2_heuristics/      # Graphs from heuristic runs
└── benchmarks/
    ├── Falkenauer/
    │   ├── Falkenauer_T/
    │   └── Falkenauer U/
    └── Scholl/
        ├── Scholl_1/
        ├── Scholl_2/
        └── Scholl_3/
```

All commands must be run from the **project root**.

---

## Running the benchmark

```shell
python benchmark.py --solver <path/to/solver.py> --dataset <dataset> --method <method>
```

### Examples

```shell
# Exact — Branch & Bound on all Falkenauer T instances
python benchmark.py --solver 1_exact/solver.py --dataset falkenauer-t --method "branch and bound"

# Exact — Backtracking, at most 60 items, 10-second time limit
python benchmark.py --solver 1_exact/solver.py --dataset scholl-1 --method backtracking --max-items 60 --time-limit 10

# Exact — Bitmask DP, exactly 20 items, skip graphs
python benchmark.py --solver 1_exact/solver.py --dataset falkenauer-u --method "dynamic programming" --num-items 20 --no-graphs

# Heuristics — Best Fit on all Scholl 2 instances
python benchmark.py --solver 2_heuristics/solver.py --dataset scholl-2 --method "best fit"

# Heuristics — First Fit, at most 120 items
python benchmark.py --solver 2_heuristics/solver.py --dataset falkenauer-u --method "first fit" --max-items 120
```

### Options

| Flag           | Description                                         | Default      |
| -------------- | --------------------------------------------------- | ------------ |
| `--solver`     | Path to the `solver.py` to use                      | *(required)* |
| `--dataset`    | Dataset key (see table below)                       | *(required)* |
| `--method`     | Solving method passed to `BinPackingSolver.solve()` | *(required)* |
| `--num-items`  | Run only instances with **exactly** N items         | —            |
| `--max-items`  | Run only instances with **at most** N items         | —            |
| `--time-limit` | Per-instance time limit in seconds                  | —            |
| `--no-graphs`  | Skip graph generation                               | off          |

`--num-items` and `--max-items` are mutually exclusive.

### Available methods

| Solver         | Methods                                                   |
| -------------- | --------------------------------------------------------- |
| `1_exact`      | `branch and bound`, `backtracking`, `dynamic programming` |
| `2_heuristics` | `next fit`, `first fit`, `best fit`                       |

> `dynamic programming` is limited to instances with n ≤ 20 items.

---

## Datasets

| Key            | Label        | Directory                             |
| -------------- | ------------ | ------------------------------------- |
| `falkenauer-t` | Falkenauer T | `benchmarks/Falkenauer/Falkenauer_T/` |
| `falkenauer-u` | Falkenauer U | `benchmarks/Falkenauer/Falkenauer U/` |
| `scholl-1`     | Scholl 1     | `benchmarks/Scholl/Scholl_1/`         |
| `scholl-2`     | Scholl 2     | `benchmarks/Scholl/Scholl_2/`         |
| `scholl-3`     | Scholl 3     | `benchmarks/Scholl/Scholl_3/`         |

**Standard file format** (one instance per `.txt` file):
```
<number of items>
<bin capacity>
<item size 1>
<item size 2>
...
```

### Adding a new dataset

Add a `DatasetConfig` entry near the top of `benchmark.py`:

```python
register_dataset(DatasetConfig(
    key="my-dataset",
    label="My Dataset",
    directory=Path("benchmarks/MyDataset"),
    parser=parse_standard,  # or a custom Callable[[Path, str], BenchmarkInstance]
))
```

Supply a custom `parser` if the file format differs from standard. Its signature must be `(filepath: Path, dataset_key: str) -> BenchmarkInstance`.

---

## Output

Results are printed as a table: `Instance`, `Items`, `Capacity`, `LB`, `Bins`, `Gap` (bins − LB), `Time (s)`, `Method`, `State` (`Done` / `T.O.`).

A summary block follows with aggregate statistics. Instances that raise an error are skipped with a warning; the rest of the run continues normally.

Unless `--no-graphs` is set, four PNG graphs are saved to `results/<solver-folder>/`:

| File                              | Description                                                         |
| --------------------------------- | ------------------------------------------------------------------- |
| `fig1_solve_times_<dataset>.png`  | Solve time per instance, slowest in red                             |
| `fig2_bins_vs_lb_<dataset>.png`   | Bins used vs lower bound, gap in red                                |
| `fig3_fill_rate_<dataset>.png`    | Average bin fill rate per instance (red → yellow → green)           |
| `fig4_time_by_size_<dataset>.png` | Solve time distribution by instance size (omitted if all same size) |
