# Hybrid ALNS (ML + Metaheuristics)

## Overview

This module contains the hybrid ALNS pipeline with learned repair.

## Main Components

- `solver.py`: runtime ALNS solver
- `training/train_repair_model.py`: repair-model training entry point
- `models/repair_model.pkl`: expected model artifact

## Benchmark Usage

Run from repository root:

```bash
python utilities/benchmarking.py --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py --dataset falkenauer-u --method-args "model_path=5_hybrid_ml_metaheuristics/hybrid_alns/models/repair_model.pkl,max_iterations=5000"
```
