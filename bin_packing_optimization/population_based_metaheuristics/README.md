# Population-Based Metaheuristics

## Overview

This module contains population-driven search approaches for the 1D Bin Packing Problem.

## Available methods

- `genetic algorithm`
- `genetic algorithm memetic`
- `genetic algorithm island`
- `ant colony optimization`

## Quick start

```python
import importlib

from bin_packing_optimization.utilities.benchmarking import create_benchmark

solver_module = importlib.import_module(
    "bin_packing_optimization.population_based_metaheuristics.solver"
)

benchmark = create_benchmark(
    dataset_key="falkenauer-u",
    solver_module=solver_module,
    time_limit=None,
)
benchmark.run(method="genetic algorithm")
benchmark.save_results_to_csv()
```

## Output

Benchmark results are written under `results/<dataset_key>/<timestamp>/results.csv`. Graph generation writes files into the same timestamped directory under `graphs/`.
