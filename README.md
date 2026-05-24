# Bin Packing Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Project-0A66C2?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A professional research repository for benchmarking algorithm families on the **1D Bin Packing Problem (BPP)**.

## Repository Structure

```text
.
├── 1_exact_methods/
├── 2_specific_heuristics/
├── 3_trajectory_based_metaheuristics/
├── 4_population_based_metaheuristics/
├── hybrid_ml_metaheuristics/
├── datasets/
├── utilities/
│   ├── benchmarking.py
│   ├── graphing.py
│   └── statistics.py
└── results/
```

## Solver Families

1. **Exact methods** (`1_exact_methods/`) for optimal solutions.
2. **Specific heuristics** (`2_specific_heuristics/`) for fast constructive/improvement baselines.
3. **Trajectory-based metaheuristics** (`3_trajectory_based_metaheuristics/`) for single-solution neighborhood search.
4. **Population-based metaheuristics** (`4_population_based_metaheuristics/`) for population-driven search (GA, ACO).
5. **Hybrid ML + metaheuristics** (`hybrid_ml_metaheuristics/`) for adaptive learning-enhanced pipelines.

## Benchmarking

Run all experiments from the project root using:

```bash
python utilities/benchmarking.py --solver <solver_path> --dataset <dataset_key> [--method <method_name>] [--method-args "k=v,..."]
```

### Example Commands

```bash
python utilities/benchmarking.py --solver 1_exact_methods/solver.py --dataset falkenauer-t --method "branch and bound"
python utilities/benchmarking.py --solver 2_specific_heuristics/solver.py --dataset scholl-2 --method "best fit"
python utilities/benchmarking.py --solver 3_trajectory_based_metaheuristics/solver.py --dataset scholl-2 --method "tabu search lns"
python utilities/benchmarking.py --solver 4_population_based_metaheuristics/solver.py --dataset falkenauer-u --method "genetic algorithm memetic"
python utilities/benchmarking.py --solver hybrid_ml_metaheuristics/hybrid_alns/solver.py --dataset falkenauer-u --method-args "model_path=hybrid_ml_metaheuristics/hybrid_alns/models/repair_model.pkl,max_iterations=5000"
```

### Core CLI Options

- `--solver` (required): path to a solver module.
- `--dataset` (required): dataset key from the registry.
- `--method` (optional): method name/alias accepted by the solver.
- `--method-args` (optional): comma-separated keyword arguments for `solve(...)`.
- `--num-items` / `--max-items` (optional): instance-size filtering.
- `--time-limit` (optional): per-instance timeout in seconds.
- `--time-limit` (optional): per-instance timeout in seconds.

## Datasets

Registered dataset keys include:

- `falkenauer-t`
- `falkenauer-u`
- `scholl-1`
- `scholl-2`
- `scholl-3`

## Results

Outputs are written automatically under the solver's folder using the
following hierarchy:

```
<solver_folder>/results/<dataset_key>/<YYYYMMDD_HHMMSS>/
	results.csv           # benchmark CSV
	graphs/               # generated PNG figures
```

No CLI flags are required to set output paths — the runner prints the
final path after each run. All `results/` directories are ignored by
default (see `.gitignore`).

Models for hybrid approaches are kept in their module folders. In
particular, trained model files (`.pkl`) for the hybrid ALNS are stored
in `hybrid_ml_metaheuristics/hybrid_alns/models/` and are NOT
gitignored. Manage those artifacts intentionally.
