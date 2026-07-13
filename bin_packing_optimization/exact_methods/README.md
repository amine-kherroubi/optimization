# Exact Methods

## Overview

This module contains exact 1D Bin Packing solvers that prioritize optimality over runtime.

## Available methods

- `backtracking`
- `branch and bound`
- `dynamic programming`

## Quick start

```python
import importlib

from bin_packing_optimization.utilities.benchmarking import create_benchmark

solver_module = importlib.import_module("bin_packing_optimization.exact_methods.solver")

benchmark = create_benchmark(
    dataset_key="falkenauer-t",
    solver_module=solver_module,
    time_limit=None,
)
benchmark.run(method="branch and bound")
benchmark.save_results_to_csv()
```

## Output

Benchmark results are written under `results/<dataset_key>/<timestamp>/results.csv`. Graphs are saved in the sibling `graphs/` directory whenever graph generation is enabled.
