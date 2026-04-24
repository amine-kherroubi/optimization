# 3_metaheuristics

Metaheuristic solvers for the 1D bin packing problem. These methods search large neighborhoods and can improve solution quality compared to simple heuristics, without optimality guarantees.

## Methods
Supported methods:
- `simulated annealing` (`sa`)
- `simulated annealing reheating` (`sa reheating`)
- `simulated annealing adaptive` (`sa adaptive`)
- `tabu search` (`ts`)
- `reactive tabu search` (`rts`)
- `tabu search lns` (`ts lns`, `lnts`)
- `tabu search diversified` (`ts diversified`, `hybrid tabu`)

## Usage (benchmark runner)
```shell
python benchmark.py --solver 3_metaheuristics/solver.py --dataset scholl-2 --method "tabu search lns"
```

## Output
`BinPackingSolver.get_solution()` returns a `BinPackingSolution` with:
- `total_bins_used`: number of bins in the solution.
- `bin_assignments`: mapping of bin index to a list of item indices.
- `final_bin_loads`: load of each bin in the same order as the mapping.

## Notes
- Simulated annealing variants start from a descending first-fit style construction and apply relocation/swap neighborhoods.
- `simulated annealing reheating` adds restart pressure when search stagnates.
- `simulated annealing adaptive` adjusts temperature based on observed acceptance ratio.
- Tabu variants support aspiration, and advanced variants include reactive tenure updates and optional LNS/diversification.
- Genetic algorithm variants were moved to `4_population_based/solver.py`.
