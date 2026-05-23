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
python utilities/benchmarking.py --solver 1_exact_methods/solver.py --dataset falkenauer-t --method "branch and bound"
```
