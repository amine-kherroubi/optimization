# ALNS + ML (1D Bin Packing)

This folder contains a **single production path** for 1D bin packing:

- Adaptive Large Neighborhood Search (ALNS)
- Thompson Sampling for destroy-operator selection
- A trained classification model for learned repair (**required** at solve time)

## Files

- `solver.py`: runtime hybrid ALNS solver.
- `train_repair_model.py`: offline data generation + model training script.

## Runtime requirements

- Python 3.10+
- `numpy`
- `scikit-learn`

Install dependencies (example):

```bash
pip install numpy scikit-learn
```

## Train the repair model

```bash
python 5_hybrid_ml_metaheuristics/hybrid_alns/train_repair_model.py \
  --instances 5000 \
  --n-min 50 \
  --n-max 200 \
  --max-negatives 5 \
  --seed 0 \
  --output 5_hybrid_ml_metaheuristics/hybrid_alns/repair_model.pkl
```

What this does exactly:

1. Generates synthetic instances with item sizes sampled uniformly in `[0.1, 0.9]`.
2. Replays Best-Fit Decreasing (BFD) decisions to create supervised labels over feasible bins.
3. Trains `sklearn.linear_model.LogisticRegression`.
4. Saves a pickle model file.

## Use the solver in benchmark.py

```bash
python benchmark.py \
  --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py \
  --dataset falkenauer-u \
  --method "hybrid alns" \
  --method-args "model_path=5_hybrid_ml_metaheuristics/hybrid_alns/repair_model.pkl,max_iterations=5000"
```

## Important behavior details

- `model_path` is mandatory in `solve(...)`; missing model path raises `ValueError`.
- If no existing bin is feasible for an item during repair, the solver opens a new bin.
- Feature schema is intentionally duplicated between training and inference **by contract**; if one side changes, the other must change identically.
- Solve defaults are:
  - `max_iterations=5000`
  - `initial_temperature=1 / ln(2)`
  - `alpha_cool=0.9995`

## Validation tips

- Ensure the model file exists before running benchmark.
- If you change feature engineering, retrain the model immediately.
- Keep seeds fixed when comparing algorithm variants (`seed` in training, fixed RNG in solver).
