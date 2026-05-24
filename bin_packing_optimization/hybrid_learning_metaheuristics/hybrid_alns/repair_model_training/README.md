# Training Folder

This folder contains the offline training workflow for the hybrid ALNS repair model.

## Contents

```text
repair_model_training/
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

## Results & Models

Trained repair models are saved to:

```
hybrid_ml_metaheuristics/hybrid_alns/models/
```

These `.pkl` files are intentionally not gitignored so they can be
managed and distributed as needed. Other runtime results are written to
the usual `results/<dataset>/<timestamp>/` folders.

## Workflow

### Step 1: Train the baseline model

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.train_repair_model import (
    TrainRepairModelConfig,
    train_repair_model,
)

train_repair_model(
    TrainRepairModelConfig(
        data=["training_data/synthetic.pkl"],
        output="repair_model_v1.pkl",
        cv_folds=5,
        no_learning_curves=False,
        no_plots=False,
        grid_search=False,
        verbose=False,
    )
)
```

### Step 2: Collect real ALNS states

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.collect_alns_states import (
    CollectAlnsStatesConfig,
    collect_alns_states,
)

collect_alns_states(
    CollectAlnsStatesConfig(
        model_path="repair_model_v1.pkl",
        instances=500,
        n_min=50,
        n_max=200,
        max_negatives=5,
        iterations=200,
        seed=1,
        output="alns_states_v1.pkl",
    )
)
```

### Step 3: Fine-tune with augmented data

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.train_repair_model import (
    TrainRepairModelConfig,
    train_repair_model,
)

train_repair_model(
    TrainRepairModelConfig(
        data=["training_data/synthetic.pkl", "alns_states_v1.pkl"],
        output="repair_model_v2.pkl",
        cv_folds=3,
        no_learning_curves=True,
        no_plots=True,
        grid_search=False,
        verbose=False,
    )
)
```

### Step 4: Validate in the benchmark

Use direct function calls:

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns import hybrid_alns_solver
from bin_packing_optimization.utilities.benchmarking import create_benchmark

benchmark = create_benchmark("falkenauer-u", hybrid_alns_solver)
benchmark.run(method=None, method_args={"max_iterations": 5000})
benchmark.save_results_to_csv()
```

## Notebook

Use `repair_model_training.ipynb` for an executable, step-by-step version of the same workflow.

## Notes

- `features.py` is the single source of truth for the 11-feature contract.
- `tqdm` is optional, but it improves the collection and training feedback.
- `numba` is optional; the pure-Python fallback remains valid.
- If you want the fastest run during development, set `no_plots=True` and `no_learning_curves=True` in the config.
