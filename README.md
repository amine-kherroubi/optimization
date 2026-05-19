# Bin Packing Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?style=flat&logo=linux&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A research-oriented project for evaluating algorithm families on the **1D Bin Packing Problem (BPP)**.

## Repository organization

```text
.
├── benchmark.py
├── 1_exact_methods/
├── 2_specific_heuristics/
├── 3_trajectory_based_metaheuristics/
├── 4_population_based_metaheuristics/
├── 5_hybrid_ml_metaheuristics/
├── datasets/
└── results/
```

### Solver taxonomy

1. **Exact methods** (`1_exact_methods/`): optimal approaches with high computational cost.
2. **Specific heuristics** (`2_specific_heuristics/`): constructive and local improvement strategies.
3. **Trajectory-based metaheuristics** (`3_trajectory_based_metaheuristics/`): single-solution neighborhood search methods.
4. **Population-based metaheuristics** (`4_population_based_metaheuristics/`): population-driven search methods such as GA and ACO.
5. **Hybrid ML + metaheuristics** (`5_hybrid_ml_metaheuristics/`): adaptive frameworks combining machine learning and metaheuristic search.

---

## Run benchmarks

All commands are intended to be executed from the project root.

```bash
python benchmark.py --solver <solver_path> --dataset <dataset_key> [--method <method_name>] [--method-args "k=v,..."]
```

### Example commands

```bash
python benchmark.py --solver 1_exact_methods/solver.py --dataset falkenauer-t --method "branch and bound"
python benchmark.py --solver 2_specific_heuristics/solver.py --dataset scholl-2 --method "best fit"
python benchmark.py --solver 3_trajectory_based_metaheuristics/solver.py --dataset scholl-2 --method "tabu search lns"
python benchmark.py --solver 4_population_based_metaheuristics/solver.py --dataset falkenauer-u --method "genetic algorithm memetic"
python benchmark.py --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py --dataset falkenauer-u --method-args "model_path=5_hybrid_ml_metaheuristics/hybrid_alns/repair_model.pkl,max_iterations=5000"
```

### CLI options

- `--solver` (required): path to a `solver.py`.
- `--dataset` (required): one of the registered datasets.
- `--method` (optional): method identifier accepted by the selected solver.
- `--method-args` (optional): comma-separated `key=value` arguments forwarded to `solve(...)`.
- `--num-items`: restrict evaluation to instances with exactly `N` items.
- `--max-items`: restrict evaluation to instances with at most `N` items.
- `--time-limit`: per-instance timeout in seconds.
- `--no-graphs`: disable plot generation.

> `--num-items` and `--max-items` are mutually exclusive.

### Method routing behavior

- Benchmark execution introspects each solver’s `solve(...)` signature.
- If `--method` is provided but unsupported by the solver, the benchmark emits a warning and ignores it.
- `--method-args` values are type-coerced (`bool`, `int`, `float`, then `str`) before dispatch.
- Unsupported keyword arguments are ignored when the solver does not accept `**kwargs`.

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

Benchmark outputs are written to:

```text
results/<solver_folder>/
```

Typical outputs include runtime plots, solution-quality comparisons against lower bounds, and fill-rate statistics.

Population-based tuning outputs are stored under:

- `results/4_population_based_metaheuristics/aco_tuning/`
- `results/4_population_based_metaheuristics/ga_tuning/`
- `results/4_population_based_metaheuristics/population_comparison.csv`
