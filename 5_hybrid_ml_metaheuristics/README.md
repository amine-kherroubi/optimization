# 5 — Hybrid ML + Metaheuristics (ALNS)

This module implements a **hybrid ALNS solver** for 1D Bin Packing that combines:

1. **Metaheuristics**: Adaptive Large Neighborhood Search (destroy/repair + simulated annealing acceptance).
2. **ML-guided operator adaptation**: Thompson Sampling bandit for choosing destroy operators online.
3. **Optional ML-guided repair**: load a pre-trained classifier (`model_path`) to score feasible bin insertions.

## Implemented design

- **Warm start**: First-Fit Decreasing (FFD).
- **Destroy operators**:
  - Random bin removal.
  - Worst-fill bin removal.
  - Related-item removal (size-similarity around a seed).
- **Repair operators**:
  - Best-Fit Decreasing (default, robust fallback).
  - Learned repair when a model is provided.
- **Acceptance**: Simulated Annealing (Metropolis criterion).
- **Adaptation**: Thompson Sampling with Beta-Bernoulli posteriors.

## Usage

Use this solver with `benchmark.py`:

```bash
python benchmark.py \
  --solver 5_hybrid_ml_metaheuristics/solver.py \
  --dataset falkenauer-u \
  --method "hybrid alns"
```

Optional parameters are passed through benchmark method args (if supported by your runner):

- `max_iterations` (default: `5000`)
- `initial_temperature` (default: `1/log(2)`)
- `alpha_cool` (default: `0.9995`)
- `model_path` (optional pickle model with `predict_proba`)

### Train and use the learned repair model

Train once offline:

```bash
python 5_hybrid_ml_metaheuristics/train_repair_model.py \
  --instances 5000 \
  --output 5_hybrid_ml_metaheuristics/repair_model.pkl
```

Then run the solver with that model:

```bash
python benchmark.py \
  --solver 5_hybrid_ml_metaheuristics/solver.py \
  --dataset falkenauer-u \
  --method "hybrid alns" \
  --method-args "model_path=5_hybrid_ml_metaheuristics/repair_model.pkl,max_iterations=5000"
```

If `model_path` is omitted, the solver automatically falls back to BFD repair.

### Why this approach is theoretically sound

- **ALNS + Simulated Annealing** gives a standard large-neighborhood exploration/exploitation loop.
- **Thompson Sampling (Beta-Bernoulli)** is a principled online method for destroy-operator selection under uncertain rewards.
- **Offline imitation learning from BFD** keeps repair behavior stable and interpretable while allowing richer contextual scoring than pure hand-coded best-fit tie-breaking.

## Notes on performance and stability

- The implementation is optimized for medium-scale classroom benchmarks (up to a few hundred items).
- Invariant rebuilds are done in linear time after destructive moves for safety and clarity.
- If no ML model is provided, the solver remains fully operational using pure ALNS + BFD repair.

## Recommended model tooling

For training the optional repair model, use:

- `scikit-learn` (e.g., Logistic Regression)
- `numpy`
- `pandas`
- `scipy` (diagnostics/plots)

Model contract: binary classifier exposing `predict_proba(X)[:, 1]`.
