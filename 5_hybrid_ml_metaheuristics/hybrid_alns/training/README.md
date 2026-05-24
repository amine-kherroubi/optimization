# Training Folder

This folder contains the offline training workflow for the hybrid ALNS repair model.

## Contents

```text
training/
├── README.md
├── repair_model_training.ipynb
├── collect_alns_states.py
├── features.py
├── train_repair_model.py
└── __init__.py
```

Generated artifacts are written next to these files unless you pass a different output path:

- `repair_model_v1.pkl` - baseline model trained on synthetic data only
- `alns_states_v1.pkl` - DAgger-lite collection of real ALNS states
- `repair_model_v2.pkl` - model trained with synthetic + ALNS-augmented

## Workflow

### Step 1: Train the baseline model

```bash
python train_repair_model.py --instances 5000 --workers 4 --output repair_model_v1.pkl
```

### Step 2: Collect real ALNS states

```bash
python collect_alns_states.py --model-path repair_model_v1.pkl --instances 500 --output alns_states_v1.pkl
```

### Step 3: Fine-tune with augmented data

```bash
python train_repair_model.py \
  --instances 2000 \
  --n-min 50 --n-max 200 \
  --max-negatives 3 \
  --seed 0 \
  --workers 2 \
  --augment-with alns_states_v1.pkl \
  --output repair_model_v2.pkl \
  --cv-folds 3 \
  --no-plots \
  --no-learning-curves
```

### Step 4: Validate in the benchmark

Run the benchmark from the project root so the model path resolves correctly:

```bash
cd ../../..
python benchmark.py \
  --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py \
  --dataset falkenauer-u \
  --method-args "model_path=5_hybrid_ml_metaheuristics/hybrid_alns/models/repair_model_v2.pkl"
```

## Notebook

Use `repair_model_training.ipynb` for an executable, step-by-step version of the same workflow.

## Notes

- `features.py` is the single source of truth for the 11-feature contract.
- `tqdm` is optional, but it improves the collection and training feedback.
- `numba` is optional; the pure-Python fallback remains valid.
- If you want the fastest run during development, keep `--no-plots` and `--no-learning-curves` enabled.
