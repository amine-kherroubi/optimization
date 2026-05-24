from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class Instance:
    """A single problem instance, independent of its source dataset."""

    name: str
    dataset_key: str
    num_items: int
    bin_capacity: int
    sizes: list[int]


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable descriptor for a benchmark dataset."""

    key: str
    label: str
    directory: Path
    parser: Callable[[Path, str], Instance]
    glob: str = "*.txt"
