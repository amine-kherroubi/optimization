# 2_heuristics

Heuristic solvers for the 1D bin packing problem. These methods are fast but not guaranteed to be optimal.

## Methods
Supported methods:
- `next fit` (`nf`)
- `first fit` (`ff`)
- `best fit` (`bf`)
- `worst fit` (`wf`)
- `next fit decreasing` (`nfd`)
- `first fit decreasing` (`ffd`)
- `best fit decreasing` (`bfd`)
- `worst fit decreasing` (`wfd`)
- `relocation`
- `swap`

## Usage (benchmark runner)
```shell
python benchmark.py --solver 2_heuristics/solver.py --dataset scholl-2 --method "best fit"
```

## Output
`BinPackingSolver.get_solution()` returns a `BinPackingSolution` with:
- `total_bins_used`: number of bins in the solution.
- `bin_assignments`: mapping of bin index to a list of item indices.
- `final_bin_loads`: load of each bin in the same order as the mapping.

## Notes
- The indices in `bin_assignments` refer to the original input order of `item_sizes`.
- `relocation` starts from a First Fit Decreasing solution and attempts to empty light bins.
- `swap` runs relocation and then attempts pairwise swaps followed by additional relocations.
