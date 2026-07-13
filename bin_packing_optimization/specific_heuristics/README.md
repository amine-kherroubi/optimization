# Specific Heuristics

## Overview

This module contains fast constructive heuristics and lightweight local-improvement strategies for the 1D Bin Packing Problem.

## Available methods

- `next fit`, `first fit`, `best fit`, `worst fit`
- `next fit decreasing`, `first fit decreasing`, `best fit decreasing`, `worst fit decreasing`
- `relocation`, `swap`

## Quick start

```python
import importlib

from bin_packing_optimization.utilities.benchmarking import create_benchmark

solver_module = importlib.import_module(
    "bin_packing_optimization.specific_heuristics.solver"
)

benchmark = create_benchmark(
    dataset_key="scholl-2",
    solver_module=solver_module,
    time_limit=None,
)
benchmark.run(method="best fit")
benchmark.save_results_to_csv()
```

## Output

Benchmark results are written under `results/<dataset_key>/<timestamp>/results.csv`. Related graph files are written to the same timestamped output directory when enabled.
