# Specific Heuristics

This module contains constructive and local-improvement heuristics for the 1D Bin Packing Problem.

## Implemented methods
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

## Benchmark usage
```bash
python benchmark.py --solver 2_specific_heuristics/solver.py --dataset scholl-2 --method "best fit"
```

## Notes
- These methods are fast, but not guaranteed to be optimal.
- `relocation` and `swap` are improvement heuristics starting from constructive solutions.
