# ALNS + ML

This subfolder contains one **complete** implementation path for 1D bin packing:

- ALNS destroy/repair search
- Thompson Sampling for destroy-operator selection
- Learned repair model (required at runtime)

No optional branches are used in this approach; every run follows the same model-guided repair path.

## Files

- `solver.py`: inference-time hybrid solver (requires `model_path`)
- `train_repair_model.py`: offline imitation-learning trainer

## Train once

```bash
python 5_hybrid_ml_metaheuristics/alns_ml_full/train_repair_model.py \
  --instances 5000 \
  --output 5_hybrid_ml_metaheuristics/alns_ml_full/repair_model.pkl
```

## Run benchmark

```bash
python benchmark.py \
  --solver 5_hybrid_ml_metaheuristics/alns_ml_full/solver.py \
  --dataset falkenauer-u \
  --method "hybrid alns" \
  --method-args "model_path=5_hybrid_ml_metaheuristics/alns_ml_full/repair_model.pkl,max_iterations=5000"
```

## Why this is a sound approach

- **ALNS + Simulated Annealing**: established mechanism to escape local minima via large neighborhood perturbations.
- **Thompson Sampling**: Bayesian multi-armed bandit for adaptive operator selection with balanced exploration/exploitation.
- **Behavioral cloning of BFD**: stable offline supervision objective for learning a repair scoring policy over feasible bin placements.

## Practical notes

- Default SA settings are tuned for longer runs (`max_iterations=5000`, `alpha_cool=0.9995`).
- Learned repair enforces feasibility checks and opens a new bin when no existing bin can fit the current item.
- Feature schema is explicitly documented in both training and inference code and kept aligned intentionally.
