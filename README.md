# Bin Packing Optimization

A research-oriented project for comparing algorithm families on the **1D Bin Packing Problem (BPP)**.

## Repository organization

```text
.
├── benchmark.py
├── 1_exact_methods/
├── 2_specific_heuristics/
├── 3_trajectory_based_metaheuristics/
├── 4_population_based_metaheuristics/
├── datasets/
└── results/
```

### Solver taxonomy

1. **Exact methods** (`1_exact_methods/`): optimal but expensive.
2. **Specific heuristics** (`2_specific_heuristics/`): fast constructive/improvement rules.
3. **Trajectory-based metaheuristics** (`3_trajectory_based_metaheuristics/`): single-solution neighborhood search.
4. **Population-based metaheuristics** (`4_population_based_metaheuristics/`): GA/ACO-style population search.

---

## Run benchmarks

All commands are expected to be run from the project root.

```bash
python benchmark.py --solver <solver_path> --dataset <dataset_key> --method <method_name>
```

### Example commands

```bash
python benchmark.py --solver 1_exact_methods/solver.py --dataset falkenauer-t --method "branch and bound"
python benchmark.py --solver 2_specific_heuristics/solver.py --dataset scholl-2 --method "best fit"
python benchmark.py --solver 3_trajectory_based_metaheuristics/solver.py --dataset scholl-2 --method "tabu search lns"
python benchmark.py --solver 4_population_based_metaheuristics/solver.py --dataset falkenauer-u --method "genetic algorithm memetic"
```

### CLI options

- `--solver` (required): path to a `solver.py`.
- `--dataset` (required): one of the registered datasets.
- `--method` (required): method string accepted by the selected solver.
- `--num-items`: keep only instances with exactly `N` items.
- `--max-items`: keep only instances with at most `N` items.
- `--time-limit`: per-instance timeout in seconds.
- `--no-graphs`: disable plot generation.

> `--num-items` and `--max-items` are mutually exclusive.

---

## Datasets

- `falkenauer-t` → `datasets/Falkenauer/Falkenauer_T/`
- `falkenauer-u` → `datasets/Falkenauer/Falkenauer U/`
- `scholl-1` → `datasets/Scholl/Scholl_1/`
- `scholl-2` → `datasets/Scholl/Scholl_2/`
- `scholl-3` → `datasets/Scholl/Scholl_3/`

Standard instance format:

```text
<number_of_items>
<bin_capacity>
<item_size_1>
<item_size_2>
...
```

---

## Results

Benchmark outputs are written under:

```text
results/<solver_folder>/
```

Typical outputs include runtime plots, bins-vs-lower-bound plots, and fill-rate summaries.

For population-based tuning workflows:

- `results/4_population_based_metaheuristics/aco_tuning/`
- `results/4_population_based_metaheuristics/ga_tuning/`
- `results/4_population_based_metaheuristics/population_comparison.csv`
