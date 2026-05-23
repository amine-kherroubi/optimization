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
python utilities/benchmarking.py --solver 2_specific_heuristics/solver.py --dataset scholl-2 --method "best fit"
```
