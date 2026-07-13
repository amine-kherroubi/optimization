# Trajectory-Based Metaheuristics

## Overview

This module contains single-solution trajectory-based search strategies for the 1D Bin Packing Problem.

## Available methods

- `simulated annealing`
- `simulated annealing reheating`
- `simulated annealing adaptive`
- `tabu search`
- `reactive tabu search`
- `tabu search lns`
- `tabu search diversified`

## Quick start

```python
import importlib

from bin_packing_optimization.utilities.benchmarking import create_benchmark

solver_module = importlib.import_module(
    "bin_packing_optimization.trajectory_based_metaheuristics.solver"
)

benchmark = create_benchmark(
    dataset_key="scholl-2",
    solver_module=solver_module,
    time_limit=None,
)
benchmark.run(method="tabu search lns")
benchmark.save_results_to_csv()
```

## Output

Benchmark results are written under `results/<dataset_key>/<timestamp>/results.csv`. Graph files are produced in the same timestamped directory under `graphs/` when requested.
