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

## Results & Models

Benchmark outputs are written automatically under the solver folder as:

```
3_trajectory_based_metaheuristics/results/<dataset_key>/<YYYYMMDD_HHMMSS>/
	results.csv
	graphs/
```

No output path needs to be provided; the runner creates the directory
and prints the full path. Result folders are gitignored by default.
