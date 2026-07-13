# Repair Model Training

This folder contains the offline training workflow for the learning-guided ALNS repair model. The scripts here define the feature contract, generate synthetic training data, collect ALNS states, and train the repair model.

## Contents

```text
repair_model_training/
├── README.md
├── repair_model_training.ipynb
├── features.py
├── generate_dataset.py
├── collect_alns_states.py
├── train_repair_model.py
└── __init__.py
```

## Generated artifacts

Artifacts are written next to the training scripts unless an explicit output path is provided:

- `training_data/synthetic_v1.pkl` — baseline synthetic dataset
- `training_data/synthetic_v2.pkl` — supplementary synthetic dataset
- `training_data/alns_states_v1.pkl` — ALNS state collection for covariate-shift mitigation
- `repair_model_v1.pkl` — baseline repair model
- `repair_model_v2.pkl` — retrained repair model

## Workflow overview

1. Generate baseline synthetic data.
2. Train the initial repair model.
3. Collect ALNS states from the solver.
4. Retrain the model with the augmented dataset.
5. Validate the resulting model in the benchmark harness.

## Example usage

```python
from bin_packing_optimization.learning_guided_metaheuristics.hybrid_alns.repair_model_training import (
    collect_alns_states,
    generate_dataset,
    train_repair_model,
)
```

The training functions are exposed through the package initializer so that notebooks and scripts can reuse the same entry points.

## Related notebooks

- `repair_model_training.ipynb`: interactive walkthrough of the training pipeline
- `../execution.ipynb` and `../performance_evaluation_v0.ipynb`: related experiment notebooks in the parent folder

## Notes

- `features.py` is the single source of truth for the feature contract. Update `FEATURE_VERSION` whenever the contract changes.
- `tqdm` is optional but recommended for progress feedback.
- `numba` is optional; the pure-Python fallback remains valid.
- Set `no_plots=True` and `no_learning_curves=True` for faster development runs.
