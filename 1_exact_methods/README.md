# Exact Methods

This module contains **optimal** algorithms for the 1D Bin Packing Problem.

## Implemented methods
- `branch and bound`
- `backtracking`
- `dynamic programming` (bitmask DP, only for `n <= 20`)

## Benchmark usage
```bash
python benchmark.py --solver 1_exact_methods/solver.py --dataset falkenauer-t --method "branch and bound"
```

## Notes
- Exact methods prioritize correctness and optimality over runtime.
- `dynamic programming` has exponential memory usage (`O(2^n)`).
