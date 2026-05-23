from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_BENCHMARKS_ROOT: Path = Path(__file__).resolve().parent


@dataclass(slots=True)
class BenchmarkInstance:
    """A single problem instance, independent of its source dataset."""

    name: str
    dataset_key: str
    num_items: int
    bin_capacity: int
    sizes: list[int]


def parse_standard(filepath: Path, dataset_key: str) -> BenchmarkInstance:
    """Parses the standard one-instance-per-file format:
        line 0   — number of items
        line 1   — bin capacity
        lines 2+ — item sizes, one per line
    Used by: Falkenauer T/U, Scholl 1/2/3.
    """
    lines = filepath.read_text(encoding="utf-8").splitlines()
    num_items: int = int(lines[0])
    bin_capacity: int = int(lines[1])
    sizes: list[int] = [int(lines[i]) for i in range(2, 2 + num_items)]
    return BenchmarkInstance(
        name=filepath.stem,
        dataset_key=dataset_key,
        num_items=num_items,
        bin_capacity=bin_capacity,
        sizes=sizes,
    )


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable descriptor for a benchmark dataset.

    To register a new dataset, create a DatasetConfig and call
    register_dataset(). No other code in this module needs to change.
    """

    key: str
    label: str
    directory: Path
    parser: Callable[[Path, str], BenchmarkInstance]
    glob: str = "*.txt"


DATASET_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(config: DatasetConfig) -> None:
    """Register a dataset configuration. Raises ValueError on duplicate key."""
    if config.key in DATASET_REGISTRY:
        raise ValueError(f"Dataset key '{config.key}' is already registered.")
    DATASET_REGISTRY[config.key] = config


register_dataset(
    DatasetConfig(
        key="falkenauer-t",
        label="Falkenauer T",
        directory=_BENCHMARKS_ROOT / "Falkenauer_T",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="falkenauer-u",
        label="Falkenauer U",
        directory=_BENCHMARKS_ROOT / "Falkenauer_U",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="scholl-1",
        label="Scholl 1",
        directory=_BENCHMARKS_ROOT / "Scholl_1",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="scholl-2",
        label="Scholl 2",
        directory=_BENCHMARKS_ROOT / "Scholl_2",
        parser=parse_standard,
    )
)
register_dataset(
    DatasetConfig(
        key="scholl-3",
        label="Scholl 3",
        directory=_BENCHMARKS_ROOT / "Scholl_3",
        parser=parse_standard,
    )
)
