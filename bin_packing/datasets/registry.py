from __future__ import annotations

from bin_packing.datasets.types import DatasetConfig

DATASET_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(config: DatasetConfig) -> None:
    """Register a dataset configuration. Raises ValueError on duplicate key."""
    if config.key in DATASET_REGISTRY:
        raise ValueError(f"Dataset key '{config.key}' is already registered.")
    DATASET_REGISTRY[config.key] = config


from bin_packing.datasets.Falkenauer_T import DATASET_CONFIG as FALKENAUER_T_CONFIG
from bin_packing.datasets.Falkenauer_U import DATASET_CONFIG as FALKENAUER_U_CONFIG
from bin_packing.datasets.Scholl_1 import DATASET_CONFIG as SCHOLL_1_CONFIG
from bin_packing.datasets.Scholl_2 import DATASET_CONFIG as SCHOLL_2_CONFIG
from bin_packing.datasets.Scholl_3 import DATASET_CONFIG as SCHOLL_3_CONFIG

for config in (
    FALKENAUER_T_CONFIG,
    FALKENAUER_U_CONFIG,
    SCHOLL_1_CONFIG,
    SCHOLL_2_CONFIG,
    SCHOLL_3_CONFIG,
):
    register_dataset(config)
