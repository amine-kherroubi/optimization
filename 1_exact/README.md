# 1_exact

Exact (optimal) solvers for the 1D bin packing problem.

## Methods
Supported methods:
- `backtracking`
- `branch and bound`
- `dynamic programming` (bitmask DP; n <= 20)

## Usage (benchmark runner)
```shell
python benchmark.py --solver 1_exact/solver.py --dataset falkenauer-t --method "branch and bound"
```

## Output
`BinPackingSolver.get_solution()` returns a `BinPackingSolution` with:
- `total_bins_used`: number of bins in the solution.
- `bin_assignments`: mapping of bin index to a list of item indices.
- `final_bin_loads`: load of each bin in the same order as the mapping.

## Notes
- Items are internally sorted in descending size order for pruning and faster search.
  The indices in `bin_assignments` refer to that sorted order.
- `dynamic programming` uses O(2^n) memory and is limited to n <= 20.
