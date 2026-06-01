from __future__ import annotations

from typing import Any

from .generate_dataset import (
    GenerateDatasetConfig,
    generate_dataset as _generate_dataset,
)
from .train_repair_model import (
    TrainRepairModelConfig,
    train_repair_model as _train_repair_model,
)
from .collect_alns_states import (
    CollectAlnsStatesConfig,
    collect_alns_states as _collect_alns_states,
)


def generate_dataset(
    *,
    instances: int = 5000,
    n_min: int = 50,
    n_max: int = 200,
    max_negatives: int = 5,
    seed: int = 0,
    workers: int = 1,
    output: str = "training_data/synthetic.pkl",
) -> dict[str, Any]:
    return _generate_dataset(
        GenerateDatasetConfig(
            instances=instances,
            n_min=n_min,
            n_max=n_max,
            max_negatives=max_negatives,
            seed=seed,
            workers=workers,
            output=output,
        )
    )


def train_repair_model(
    *,
    data: list[str],
    output: str = "repair_model.pkl",
    seed: int = 42,
    min_roc_auc: float = 0.80,
    min_average_precision: float = 0.60,
    min_top1_accuracy: float = 0.70,
    require_alns_states: bool | None = None,
    no_learning_curves: bool = False,
    no_plots: bool = False,
    cv_folds: int = 5,
    grid_search: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    return _train_repair_model(
        TrainRepairModelConfig(
            data=data,
            output=output,
            seed=seed,
            min_roc_auc=min_roc_auc,
            min_average_precision=min_average_precision,
            min_top1_accuracy=min_top1_accuracy,
            require_alns_states=require_alns_states,
            no_learning_curves=no_learning_curves,
            no_plots=no_plots,
            cv_folds=cv_folds,
            grid_search=grid_search,
            verbose=verbose,
        )
    )


def collect_alns_states(
    *,
    model_path: str,
    instances: int = 500,
    n_min: int = 50,
    n_max: int = 200,
    max_negatives: int = 5,
    iterations: int = 200,
    seed: int = 1,
    output: str = "alns_states.pkl",
) -> dict[str, Any]:
    return _collect_alns_states(
        CollectAlnsStatesConfig(
            model_path=model_path,
            instances=instances,
            n_min=n_min,
            n_max=n_max,
            max_negatives=max_negatives,
            iterations=iterations,
            seed=seed,
            output=output,
        )
    )


__all__ = [
    "generate_dataset",
    "train_repair_model",
    "collect_alns_states",
    "GenerateDatasetConfig",
    "TrainRepairModelConfig",
    "CollectAlnsStatesConfig",
]
