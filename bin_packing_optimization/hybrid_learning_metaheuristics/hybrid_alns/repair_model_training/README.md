# Training Folder

This folder contains the offline training workflow for the hybrid ALNS repair model.

## Contents

```text
repair_model_training/
├── README.md
├── repair_model_training.ipynb      ← original notebook
├── features.py                      ← single source of truth for the 11-feature contract
├── generate_dataset.py              ← synthetic data generation
├── collect_alns_states.py           ← DAgger-lite ALNS state collection
├── train_repair_model.py            ← model training with integrity checks & quality gates
└── __init__.py
```

Improved notebooks live in the adjacent `../offline_training/` folder.

Generated artifacts are written next to these files unless you pass a different output path:

- `training_data/synthetic_v1.pkl`  — baseline synthetic dataset
- `training_data/synthetic_v2.pkl`  — supplementary synthetic dataset (v2 pipeline)
- `training_data/alns_states_v1.pkl` — DAgger-lite ALNS state collection
- `repair_model_v1.pkl`             — baseline model (synthetic data only)
- `repair_model_v2.pkl`             — improved model (synthetic + ALNS-augmented)

## Results & Models

Production models are saved to:

```
hybrid_learning_metaheuristics/hybrid_alns/models/
```

## Workflow

### Step 0: Seeds — define once, use everywhere

```python
SEED              = 42   # train/test split + model random_state
SYNTHETIC_V1_SEED = 0    # generate_dataset v1
SYNTHETIC_V2_SEED = 2    # generate_dataset v2
ALNS_SEED         = 1    # collect_alns_states
```

### Step 1: Generate baseline synthetic data

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns \
    .repair_model_training.generate_dataset import GenerateDatasetConfig, generate_dataset

generate_dataset(GenerateDatasetConfig(
    instances=4000, n_min=50, n_max=200, max_negatives=5,
    seed=SYNTHETIC_V1_SEED, workers=1,
    output="training_data/synthetic_v1.pkl",
))
```

### Step 2: Train the baseline model (v1)

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns \
    .repair_model_training.train_repair_model import TrainRepairModelConfig, train_repair_model

train_repair_model(TrainRepairModelConfig(
    data=["training_data/synthetic_v1.pkl"],
    output="repair_model_v1.pkl",
    seed=SEED,
    min_roc_auc=0.80,
    min_average_precision=0.60,
    require_alns_states=False,   # v1 baseline: no ALNS data required
    cv_folds=5,
))
```

### Step 3: Collect real ALNS states (covariate-shift mitigation)

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns \
    .repair_model_training.collect_alns_states import CollectAlnsStatesConfig, collect_alns_states

collect_alns_states(CollectAlnsStatesConfig(
    model_path="repair_model_v1.pkl",
    instances=500, n_min=50, n_max=200,
    max_negatives=5, iterations=200,
    seed=ALNS_SEED,
    output="training_data/alns_states_v1.pkl",
))
```

### Step 4: Retrain with synthetic + ALNS data (v2)

```python
generate_dataset(GenerateDatasetConfig(
    instances=2000, n_min=50, n_max=200, max_negatives=3,
    seed=SYNTHETIC_V2_SEED, workers=1,
    output="training_data/synthetic_v2.pkl",
))

train_repair_model(TrainRepairModelConfig(
    data=["training_data/synthetic_v2.pkl", "training_data/alns_states_v1.pkl"],
    output="repair_model_v2.pkl",
    seed=SEED,
    min_roc_auc=0.80,
    min_average_precision=0.60,
    require_alns_states=True,    # v2: ALNS data required; raises if absent
    cv_folds=3,
))
```

### Step 5: Validate in the benchmark (optional)

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns import hybrid_alns_solver
from bin_packing_optimization.utilities.benchmarking import create_benchmark

benchmark = create_benchmark("falkenauer-u", hybrid_alns_solver)
benchmark.run(method=None, method_args={"max_iterations": 5000})
benchmark.save_results_to_csv()
```

## Key `TrainRepairModelConfig` fields

| Field | Default | Purpose |
|---|---|---|
| `seed` | `42` | Single source of randomness: train/test split + GradientBoosting `random_state` |
| `min_roc_auc` | `0.70` | Quality gate — warns if holdout ROC-AUC falls below this |
| `min_average_precision` | `0.50` | Quality gate — warns if holdout AP falls below this |
| `require_alns_states` | `True` | Raises if no ALNS-tagged dataset is provided (set `False` for v1 baseline) |
| `cv_folds` | `5` | Cross-validation folds |
| `grid_search` | `False` | Enable GridSearchCV hyperparameter sweep (slow, ~1–2 h) |

## Notebooks

repair_model_training.ipynb — step-by-step notebook for local execution, including the documented walkthrough and rationale.
../offline_training/offline_training_improve.ipynb — Colab-ready pipeline.
## Notes

- `features.py` is the single source of truth for the 11-feature contract.
  Increment `FEATURE_VERSION` whenever the contract changes.
- `tqdm` is optional but recommended for progress feedback.
- `numba` is optional; the pure-Python fallback is always valid.
- Set `no_plots=True` and `no_learning_curves=True` for faster development runs.