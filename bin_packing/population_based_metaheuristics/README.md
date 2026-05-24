# Population-Based Metaheuristics

## Overview

This module contains population-driven search approaches.

## Implemented Methods

- `genetic algorithm`
- `genetic algorithm memetic`
- `genetic algorithm island`
- `ant colony optimization`

## Benchmark Usage

Run from repository root:

```bash
python -m bin_packing.utilities.benchmarking --solver bin_packing/population_based_metaheuristics/solver.py --dataset falkenauer-u --method "genetic algorithm"
```

## Results & Models

Benchmark outputs are written automatically under the solver folder as:

```
bin_packing/population_based_metaheuristics/results/<dataset_key>/<YYYYMMDD_HHMMSS>/
	results.csv
	graphs/
```

No output path needs to be provided; the runner creates the directory
and prints the full path. Result folders are gitignored by default.
