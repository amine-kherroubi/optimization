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

## Results & Models

Benchmark outputs are written automatically under the solver folder as:

```
5_hybrid_ml_metaheuristics/hybrid_alns/results/<dataset_key>/<YYYYMMDD_HHMMSS>/
	results.csv
	graphs/
```

Trained model artifacts (`.pkl`) produced by the training pipeline are
stored in `5_hybrid_ml_metaheuristics/hybrid_alns/models/` and are NOT
gitignored so they can be tracked or shared explicitly. Result folders
are gitignored by default.
