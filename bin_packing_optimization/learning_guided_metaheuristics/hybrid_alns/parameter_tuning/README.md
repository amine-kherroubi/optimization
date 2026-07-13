# Parameter Tuning

This folder contains notebooks and helper scripts for tuning learning-guided ALNS-related parameters.

## Output convention

Parameter tuning outputs follow the same result convention as the main benchmark workflow and are written under `results/<dataset_key>/<timestamp>/results.csv`. Related graph files are produced in the sibling `graphs/` directory when generated.

Trained model artifacts (`.pkl`) are stored under `bin_packing_optimization/learning_guided_metaheuristics/hybrid_alns/models/` and are not automatically ignored by Git.
