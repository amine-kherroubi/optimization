# Trajectory-Based Metaheuristics

## Overview

This module contains single-solution trajectory-based search strategies.

## Implemented Methods

- `simulated annealing`
- `simulated annealing reheating`
- `simulated annealing adaptive`
- `tabu search`
- `reactive tabu search`
- `tabu search lns`
- `tabu search diversified`

## Benchmark Usage

Run from repository root:

```bash
python utilities/benchmarking.py --solver 3_trajectory_based_metaheuristics/solver.py --dataset scholl-2 --method "tabu search lns"
```
