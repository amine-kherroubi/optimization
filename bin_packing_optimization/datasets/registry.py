from __future__ import annotations

from pathlib import Path

from bin_packing_optimization.datasets.types import DatasetConfig, Instance

DATASET_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(config: DatasetConfig) -> None:
    """Register a dataset configuration. Raises ValueError on duplicate key."""
    if config.key in DATASET_REGISTRY:
        raise ValueError(f"Dataset key '{config.key}' is already registered.")
    DATASET_REGISTRY[config.key] = config


def parse_instance(filepath: Path, dataset_key: str) -> Instance:
    """Parse a one-instance-per-file benchmark format."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    num_items = int(lines[0])
    bin_capacity = int(lines[1])
    sizes = [int(lines[i]) for i in range(2, 2 + num_items)]
    return Instance(
        name=filepath.stem,
        dataset_key=dataset_key,
        num_items=num_items,
        bin_capacity=bin_capacity,
        sizes=sizes,
    )


def _normalize_part(part: str) -> str:
    return part.lower().replace("_", "-")


def _dataset_key(relative_directory: Path) -> str:
    """Create a stable key from the dataset directory path under datasets/."""
    parts = [_normalize_part(part) for part in relative_directory.parts]
    if len(parts) == 1:
        return parts[0]

    parent = parts[-2]
    last = parts[-1]
    if last.startswith(f"{parent}-"):
        return last
    return "-".join(parts)


def _dataset_label(relative_directory: Path) -> str:
    """Create a human-readable label from the dataset directory path under datasets/."""
    return " / ".join(part.replace("_", " ") for part in relative_directory.parts)


def discover_and_register_datasets() -> None:
    """Discover all dataset folders containing .txt instances and register them."""
    datasets_root = Path(__file__).resolve().parent
    for directory in sorted(path for path in datasets_root.rglob("*") if path.is_dir()):
        if not any(directory.glob("*.txt")):
            continue
        relative_directory = directory.relative_to(datasets_root)
        register_dataset(
            DatasetConfig(
                key=_dataset_key(relative_directory),
                label=_dataset_label(relative_directory),
                directory=directory,
                parser=parse_instance,
            )
        )


discover_and_register_datasets()
