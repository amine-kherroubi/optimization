# Exact Methods

## Overview

This module contains exact 1D Bin Packing solvers that prioritize optimality.

## Implemented Methods

- `branch and bound`
- `backtracking`
- `dynamic programming`

## Benchmark Usage

Run from repository root:

```bash
python -m bin_packing.utilities.benchmarking --solver bin_packing/exact_methods/solver.py --dataset falkenauer-t --method "branch and bound"
```

## Results & Models

Benchmark outputs are written automatically under the solver folder as:

```
bin_packing/exact_methods/results/<dataset_key>/<YYYYMMDD_HHMMSS>/
	results.csv
	graphs/
```

No output path needs to be provided; the runner creates the directory
and prints the full path. Result folders are gitignored by default.
