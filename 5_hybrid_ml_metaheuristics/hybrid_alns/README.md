# Hybrid ALNS (ML + Metaheuristics)

## Overview

This module contains the hybrid ALNS pipeline with learned repair.

## Main Components

- `solver.py`: runtime ALNS solver
 - `solver.py`: runtime ALNS solver
 - `repair_model_training/train_repair_model.py`: repair-model training entry point
 - `models/repair_model_v1.pkl`, `models/repair_model_v2.pkl`, `models/alns_states_v1.pkl`: example artifacts

## Benchmark Usage

Run from repository root:

```bash
python utilities/benchmarking.py --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py --dataset falkenauer-u --method-args "model_path=5_hybrid_ml_metaheuristics/hybrid_alns/models/repair_model_v2.pkl,max_iterations=5000"
```
