# Offline Training (Improved)

This folder contains Colab-ready notebooks for training the Hybrid ALNS repair
model with explicit reproducibility, dataset integrity checks, quality gates,
and covariate-shift mitigation.

## What changed
- Reproducibility: seeds are defined once and passed to dataset generation, ALNS
  state collection, and model training. Train/test split and model RNG use the
  same seed.
- Dataset integrity checks: feature version, feature count, label values, and
  NaNs are validated at load time.
- Dataset summaries: merged datasets print total rows, positive rate, and
  per-source counts; the ALNS ratio is reported.
- Quality gates: holdout ROC-AUC and Average Precision are computed; warnings
  are emitted if scores fall below configured thresholds.
- Covariate shift: v2+ models require ALNS state data; the ALNS proportion is
  reported explicitly.

## Notebooks
- offline_training_improve.ipynb: Colab-first executable pipeline.
- repair_model_trainig_improvement.ipynb: documented walkthrough with the same
  safeguards.

## Import guidance
Use the configuration-driven entry points to keep reproducibility and checks
intact:

```python
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.generate_dataset import (
  GenerateDatasetConfig,
  generate_dataset,
)
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.collect_alns_states import (
  CollectAlnsStatesConfig,
  collect_alns_states,
)
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.train_repair_model import (
  TrainRepairModelConfig,
  train_repair_model,
)
```

Key TrainRepairModelConfig fields:
- seed: single source of randomness for split and model training.
- min_roc_auc / min_average_precision: quality gate thresholds.
- require_alns_states: enforce covariate-shift mitigation explicitly.
