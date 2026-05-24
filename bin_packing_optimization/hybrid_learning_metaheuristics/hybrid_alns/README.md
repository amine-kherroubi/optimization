# Hybrid ALNS (ML + Metaheuristics)

## Overview

This module contains the hybrid ALNS pipeline with learned repair.

## Main Components

- `solver.py`: runtime ALNS solver
- `repair_model_training/train_repair_model.py`: repair-model training entry point
- `models/repair_model_v1.pkl`, `models/repair_model_v2.pkl`, `models/alns_states_v1.pkl`: example artifacts

## Benchmark Usage

Use object instantiation + function calls (no CLI):

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns import hybrid_alns_solver
from bin_packing_optimization.utilities.benchmarking import create_benchmark

benchmark = create_benchmark(
    dataset_key="falkenauer-u",
    solver_module=hybrid_alns_solver,
    time_limit=None,
)
benchmark.run(method=None, method_args={"max_iterations": 5000})
csv_path = benchmark.save_results_to_csv()
print(csv_path)
```

## Results & Models

Benchmark outputs are written automatically under the solver folder as:

```
bin_packing/hybrid_ml_metaheuristics/hybrid_alns/results/<dataset_key>/<YYYYMMDD_HHMMSS>/
	results.csv
	graphs/
```

Trained model artifacts (`.pkl`) produced by the training pipeline are
stored in `bin_packing/hybrid_ml_metaheuristics/hybrid_alns/models/` and are NOT
gitignored so they can be tracked or shared explicitly. Result folders
are gitignored by default.


### Notebook Model Loading

`performance_evaluation.ipynb` loads the selected model artifact explicitly from
`bin_packing.hybrid_ml_metaheuristics.hybrid_alns.models` using
`importlib.resources` + `pickle`, so model-version choice stays in the notebook.
