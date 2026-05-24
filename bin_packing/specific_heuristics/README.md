# Specific Heuristics

## Overview

This module contains fast constructive and lightweight local-improvement heuristics.

## Implemented Methods

- `next fit`, `first fit`, `best fit`, `worst fit`
- `next fit decreasing`, `first fit decreasing`, `best fit decreasing`, `worst fit decreasing`
- `relocation`, `swap`

## Benchmark Usage

Run from repository root:

```bash
python -m bin_packing.utilities.benchmarking --solver bin_packing/specific_heuristics/solver.py --dataset scholl-2 --method "best fit"
```

## Results & Models

Benchmark outputs are written automatically under the solver folder as:

```
bin_packing/specific_heuristics/results/<dataset_key>/<YYYYMMDD_HHMMSS>/
	results.csv
	graphs/
```

No output path needs to be provided; the runner creates the directory
and prints the full path. Result folders are gitignored by default.
