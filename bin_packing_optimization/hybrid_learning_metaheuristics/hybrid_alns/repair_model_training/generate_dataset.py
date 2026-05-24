"""Generate a synthetic training dataset for the repair model.

Extracted from train_repair_model.py so that dataset generation and model
training are independent steps. This lets you:
  - Regenerate data without retraining (and vice versa)
  - Version and inspect raw datasets as standalone .pkl files
  - Reuse the same dataset across multiple training runs / hyperparameter sweeps

Output .pkl files are compatible with train_repair_model.py (data field) and
collect_alns_states.py (augmentation data input).

Labelling strategy
------------------
For each synthetic instance we build an FFD start solution then label every
feasible bin placement using the BFD oracle: the bin that minimises
post-placement slack is the positive; other feasible bins are sampled
negatives (up to max_negatives).

Usage
-----
    from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.generate_dataset import GenerateDatasetConfig, generate_dataset

    generate_dataset(GenerateDatasetConfig(instances=4000, seed=0, output="training_data/synthetic_v1.pkl"))

Then pass the output to train_repair_model.py:

    train_repair_model(TrainRepairModelConfig(data=["training_data/synthetic_v1.pkl"]))
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


try:
    from . import features
except ImportError:  # pragma: no cover - fallback for direct script execution
    import features  # type: ignore[no-redef]

FEATURE_VERSION = features.FEATURE_VERSION
N_FEATURES = features.N_FEATURES

DEFAULT_OUTPUT_DIR = "training_data"


# ============================================================================
# Config
# ============================================================================


@dataclass
class GenerateDatasetConfig:
    instances: int = 5000
    n_min: int = 50
    n_max: int = 200
    max_negatives: int = 5
    seed: int = 0
    output: str = f"{DEFAULT_OUTPUT_DIR}/synthetic.pkl"


@dataclass
class DatasetSummary:
    rows: int
    cols: int
    positive_rate: float


# ============================================================================
# Instance sampling
# ============================================================================


def _generate_instance(rng: np.random.Generator, n_min: int, n_max: int) -> np.ndarray:
    """Sample a random bin-packing instance (item sizes in [0.05, 0.95])."""
    n = int(rng.integers(n_min, n_max + 1))
    dist = rng.integers(0, 5)
    if dist < 2:
        sizes = rng.uniform(0.1, 0.9, size=n)
    elif dist < 4:
        half = n // 2
        sizes = np.concatenate(
            [rng.uniform(0.05, 0.35, half), rng.uniform(0.60, 0.95, n - half)]
        )
        rng.shuffle(sizes)
    else:
        sizes = rng.normal(0.5, 0.2, size=n)
    return np.clip(sizes, 0.05, 0.95)


# ============================================================================
# Dataset builder
# ============================================================================


def build_dataset(
    *,
    instances: int,
    n_min: int,
    n_max: int,
    max_negatives: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, DatasetSummary]:
    """Generate a synthetic (X, y) training dataset.

    For each instance:
      1. Build an FFD start solution.
      2. For every item, find all feasible bins.
      3. Label the minimum-slack bin as positive (BFD oracle).
      4. Sample up to max_negatives other feasible bins as negatives.

    Returns
    -------
    X : float32 array of shape (n_rows, N_FEATURES)
    y : int32 array of shape (n_rows,)
    summary : DatasetSummary
    """
    rng = np.random.default_rng(seed)
    mf = features.make_features
    all_X: list[list[float]] = []
    all_y: list[int] = []

    for _ in tqdm(range(instances), desc="Generating dataset", total=instances):
        sizes = _generate_instance(rng, n_min, n_max)
        sizes_list = [float(v) for v in sizes]
        n = len(sizes_list)
        capacity = 1.0

        # Descending size order used for both FFD and size-rank feature
        order = sorted(range(n), key=lambda i: -sizes_list[i])
        size_rank: dict[int, int] = {item: rank for rank, item in enumerate(order)}

        # FFD initial solution
        bins: list[list[int]] = []
        bin_loads: list[float] = []
        for item in order:
            s = sizes_list[item]
            placed = False
            for j, load in enumerate(bin_loads):
                if load + s <= capacity:
                    bins[j].append(item)
                    bin_loads[j] += s
                    placed = True
                    break
            if not placed:
                bins.append([item])
                bin_loads.append(s)

        # BFD oracle labelling
        remaining = len(order)
        denom = max(1, n)
        for item in order:
            item_size = sizes_list[item]
            feasible: list[int] = []
            best_bin = -1
            best_slack = float("inf")
            for j, load in enumerate(bin_loads):
                rem = capacity - load
                if rem + 1e-9 < item_size:
                    continue
                feasible.append(j)
                slack = rem - item_size
                if slack < best_slack:
                    best_slack = slack
                    best_bin = j

            if best_bin == -1:
                remaining -= 1
                continue

            def _feat(bin_idx: int) -> list[float]:
                return mf(
                    item=item,
                    item_size=item_size,
                    bin_items=bins[bin_idx],
                    bin_load=bin_loads[bin_idx],
                    capacity=capacity,
                    sizes=sizes_list,
                    n_total=n,
                    size_rank=size_rank,
                    remaining_ratio=remaining / denom,
                )

            all_X.append(_feat(best_bin))
            all_y.append(1)

            negatives = [j for j in feasible if j != best_bin]
            if len(negatives) > max_negatives:
                negatives = list(
                    rng.choice(negatives, size=max_negatives, replace=False)
                )
            for j in negatives:
                all_X.append(_feat(j))
                all_y.append(0)

            remaining -= 1

    if len(all_X) == 0:
        X_arr = np.zeros((0, N_FEATURES), dtype=np.float32)
        y_arr = np.zeros((0,), dtype=np.int32)
    else:
        X_arr = np.asarray(all_X, dtype=np.float32)
        y_arr = np.asarray(all_y, dtype=np.int32)

    pos_rate = float(y_arr.mean()) if y_arr.size > 0 else 0.0
    summary = DatasetSummary(
        rows=int(X_arr.shape[0]),
        cols=int(N_FEATURES),
        positive_rate=pos_rate,
    )
    return X_arr, y_arr, summary


# ============================================================================
# Public entry point
# ============================================================================


def generate_dataset(config: GenerateDatasetConfig) -> dict:
    """Generate and save a synthetic training dataset.

    Parameters
    ----------
    config : GenerateDatasetConfig

    Returns
    -------
    dict with keys: X, y, feature_version, summary
    """
    print("=" * 70)
    print("DATASET GENERATION")
    print("=" * 70)
    print(f"  Instances          : {config.instances}")
    print(f"  Instance size range: [{config.n_min}, {config.n_max}] items")
    print(f"  Max negatives      : {config.max_negatives}")
    print(f"  Seed               : {config.seed}")
    print(f"  Output             : {config.output}")

    X, y, summary = build_dataset(
        instances=config.instances,
        n_min=config.n_min,
        n_max=config.n_max,
        max_negatives=config.max_negatives,
        seed=config.seed,
    )

    print(f"\nDataset: {summary.rows:,} rows x {summary.cols} features")
    print(
        f"  Positive rate : {summary.positive_rate:.4f}"
        f"  (expected min: {1.0 / (1 + config.max_negatives):.4f})"
    )
    print(f"  Class 0 (neg) : {(y == 0).sum():,}")
    print(f"  Class 1 (pos) : {(y == 1).sum():,}")

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "X": X,
        "y": y,
        "feature_version": FEATURE_VERSION,
        "summary": {
            "instances": config.instances,
            "n_min": config.n_min,
            "n_max": config.n_max,
            "max_negatives": config.max_negatives,
            "seed": config.seed,
            "rows": summary.rows,
            "cols": summary.cols,
            "positive_rate": summary.positive_rate,
        },
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {output_path}  ({size_mb:.1f} MB)")
    print(
        "Next step: pass this file to train_repair_model.py via the data config field"
    )
    return payload


# ============================================================================
