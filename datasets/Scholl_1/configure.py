from __future__ import annotations

from pathlib import Path

from datasets.types import Instance, DatasetConfig


def parse_instance(filepath: Path, dataset_key: str) -> Instance:
    """Parse this dataset's one-instance-per-file benchmark format."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    num_items: int = int(lines[0])
    bin_capacity: int = int(lines[1])
    sizes: list[int] = [int(lines[i]) for i in range(2, 2 + num_items)]
    return Instance(
        name=filepath.stem,
        dataset_key=dataset_key,
        num_items=num_items,
        bin_capacity=bin_capacity,
        sizes=sizes,
    )


DATASET_CONFIG = DatasetConfig(
    key="scholl-1",
    label="Scholl 1",
    directory=Path(__file__).resolve().parent,
    parser=parse_instance,
)
