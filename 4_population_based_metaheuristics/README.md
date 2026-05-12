# Population-Based Metaheuristics

This module contains population-level search methods for the 1D Bin Packing Problem.

## Implemented methods
- `genetic algorithm` (`ga`)
- `genetic algorithm memetic` (`ga memetic`)
- `genetic algorithm island` (`ga island`)
- `ant colony optimization` (`aco`)

## Benchmark usage
```bash
python benchmark.py --solver 4_population_based_metaheuristics/solver.py --dataset falkenauer-u --method "genetic algorithm"
```

## Tuning scripts
- ACO tuning:
  ```bash
  python 4_population_based_metaheuristics/tune_aco.py
  ```
- GA tuning:
  ```bash
  python 4_population_based_metaheuristics/tune_ga.py
  ```

Generated files are written under `results/4_population_based_metaheuristics/`.
