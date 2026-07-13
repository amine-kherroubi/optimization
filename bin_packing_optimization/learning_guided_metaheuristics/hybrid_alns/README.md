# Hybrid ALNS (Machine Learning + Metaheuristics)

## Overview

This package contains a learning-guided ALNS pipeline that combines search operators with a learned repair model.

## Main components

- `hybrid_alns_solver.py`: runtime ALNS solver entry point
- `repair_model_training/`: training pipeline for the repair model
- `models/`: example model artifacts and checkpoints

## Quick start

```python
import importlib

from bin_packing_optimization.utilities.benchmarking import create_benchmark

solver_module = importlib.import_module(
    "bin_packing_optimization.learning_guided_metaheuristics.hybrid_alns.hybrid_alns_solver"
)

benchmark = create_benchmark(
    dataset_key="falkenauer-u",
    solver_module=solver_module,
    time_limit=None,
)
benchmark.run(method=None, method_args={"max_iterations": 5000})
benchmark.save_results_to_csv()
```

## Output

Benchmark outputs are written under `results/<dataset_key>/<timestamp>/results.csv`. Trained model artifacts (`.pkl`) are stored under `bin_packing_optimization/learning_guided_metaheuristics/hybrid_alns/models/`.

## Notebook model loading

The notebook workflow selects model artifacts explicitly from the package’s model directory, so the model version is tracked alongside the experiment.
