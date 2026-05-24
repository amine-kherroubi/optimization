from __future__ import annotations

import pickle
from importlib.resources import files
from typing import Any


def load_model_bundle(version: str = "v2") -> Any:
    """Load a packaged repair-model bundle.

    Args:
        version: Model version suffix (e.g., "v1", "v2").
    """
    model_resource = files("bin_packing.hybrid_ml_metaheuristics.hybrid_alns.models") / f"repair_model_{version}.pkl"
    with model_resource.open("rb") as f:
        return pickle.load(f)
