from __future__ import annotations

from typing import Any

from .train_repair_model import (
    TrainRepairModelConfig,
    train_repair_model as _train_repair_model,
)
from .collect_alns_states import (
    CollectAlnsStatesConfig,
    collect_alns_states as _collect_alns_states,
)


def train_repair_model(
    *,
    instances: int = 5000,
    n_min: int = 50,
    n_max: int = 200,
    max_negatives: int = 5,
    seed: int = 0,
    workers: int = 1,
    output: str = "repair_model.pkl",
    augment_with: str | None = None,
    no_learning_curves: bool = False,
    no_plots: bool = False,
    cv_folds: int = 5,
    grid_search: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    return _train_repair_model(
        TrainRepairModelConfig(
            instances=instances,
            n_min=n_min,
            n_max=n_max,
            max_negatives=max_negatives,
            seed=seed,
            workers=workers,
            output=output,
            augment_with=augment_with,
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
    "train_repair_model",
    "collect_alns_states",
    "TrainRepairModelConfig",
    "CollectAlnsStatesConfig",
]
