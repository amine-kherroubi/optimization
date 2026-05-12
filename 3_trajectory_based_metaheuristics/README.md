# Trajectory-Based Metaheuristics

This module contains single-solution trajectory-based metaheuristics for the 1D Bin Packing Problem.

## Implemented methods
- `simulated annealing` (`sa`)
- `simulated annealing reheating` (`sa reheating`)
- `simulated annealing adaptive` (`sa adaptive`)
- `tabu search` (`ts`)
- `reactive tabu search` (`rts`)
- `tabu search lns` (`ts lns`, `lnts`)
- `tabu search diversified` (`ts diversified`, `hybrid tabu`)

## Benchmark usage
```bash
python benchmark.py --solver 3_trajectory_based_metaheuristics/solver.py --dataset scholl-2 --method "tabu search lns"
```

## Notes
- These methods explore neighborhoods around a current solution.
- Population-based methods (GA/ACO) are implemented in `4_population_based_metaheuristics/`.
