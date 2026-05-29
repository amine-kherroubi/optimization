"""Generate a synthetic training dataset for the repair model.

Extracted from train_repair_model.py so that dataset generation and model
training are independent steps. This lets you:
  - Regenerate data without retraining (and vice versa).
  - Version and inspect raw datasets as standalone .pkl files.
  - Reuse the same dataset across multiple training runs / hyperparameter sweeps.

Output .pkl files are compatible with train_repair_model.py (data field) and
collect_alns_states.py (augmentation data input).

Labelling strategy
------------------
For each synthetic instance we build an FFD (First Fit Decreasing) start
solution, then label every feasible bin placement using the BFD oracle: the
bin that minimises post-placement slack is the positive; other feasible bins
are sampled as negatives (up to max_negatives).

Usage
-----
    from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns \
        .repair_model_training.generate_dataset import GenerateDatasetConfig, generate_dataset

    generate_dataset(GenerateDatasetConfig(
        instances=4000, seed=0, output="training_data/synthetic_v1.pkl"
    ))

Then pass the output to train_repair_model.py:

    train_repair_model(TrainRepairModelConfig(data=["training_data/synthetic_v1.pkl"]))
"""

from __future__ import annotations

import pickle
from multiprocessing import Pool
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
except ImportError:  # pragma: no cover – fallback for direct script execution
    import features  # type: ignore[no-redef]

FEATURE_VERSION = features.FEATURE_VERSION
N_FEATURES = features.N_FEATURES

DEFAULT_OUTPUT_DIR = "training_data"


def _validate_dataset_arrays(X: np.ndarray, y: np.ndarray, *, context: str) -> None:
    """Validate dataset integrity before saving."""
    if X.ndim != 2:
        raise ValueError(f"{context}: X must be 2D, got shape={X.shape}")
    if X.shape[1] != N_FEATURES:
        raise ValueError(
            f"{context}: X has {X.shape[1]} features; expected {N_FEATURES}"
        )
    if y.ndim != 1:
        raise ValueError(f"{context}: y must be 1D, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{context}: X rows ({X.shape[0]}) != y rows ({y.shape[0]})"
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{context}: X contains NaN or inf values")
    if not np.isfinite(y).all():
        raise ValueError(f"{context}: y contains NaN or inf values")
    labels = np.unique(y)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(f"{context}: y has invalid labels: {labels}")


# ============================================================================
# Config
# ============================================================================


@dataclass
class GenerateDatasetConfig:
    """Configuration for dataset generation.

    Attributes
    ----------
    instances:
        Number of synthetic bin-packing instances to generate.
    n_min:
        Minimum number of items per instance.
    n_max:
        Maximum number of items per instance (inclusive).
    max_negatives:
        Maximum number of negative (non-BFD) bin placements to sample per item.
        Controls the positive-to-negative ratio: at most 1 positive per
        (1 + max_negatives) rows.
    seed:
        Root RNG seed.  Each instance gets a deterministically derived
        child seed, so results are fully reproducible regardless of
        worker count.
    workers:
        Number of worker processes.  Use 1 for single-process execution
        (easier to debug); use more for speed on large datasets.
    output:
        Path to the output .pkl file.  Parent directories are created
        automatically.
    """

    instances: int = 5000
    n_min: int = 50
    n_max: int = 200
    max_negatives: int = 5
    seed: int = 0
    workers: int = 1
    output: str = f"{DEFAULT_OUTPUT_DIR}/synthetic.pkl"


@dataclass
class DatasetSummary:
    """Lightweight summary of the generated dataset."""

    rows: int
    cols: int
    positive_rate: float


# ============================================================================
# Instance sampling
# ============================================================================


def _generate_instance(rng: np.random.Generator, n_min: int, n_max: int) -> np.ndarray:
    """Sample a random bin-packing instance.

    Item sizes are drawn from one of three distributions chosen at random:
      - Uniform [0.1, 0.9]           (dist 0–1): general case.
      - Bimodal small/large items    (dist 2–3): stresses packing decisions.
      - Normal around 0.5, σ=0.2    (dist 4):   moderate variability.

    All sizes are clipped to [0.05, 0.95] so no item is trivially tiny or
    impossible to place.

    Parameters
    ----------
    rng:
        Per-instance random generator (already seeded).
    n_min, n_max:
        Inclusive range for the number of items.

    Returns
    -------
    np.ndarray of float64, shape (n,)
        Item sizes in [0.05, 0.95].
    """
    n = int(rng.integers(n_min, n_max + 1))
    dist = rng.integers(0, 5)

    if dist < 2:
        # Uniform distribution over a wide range.
        sizes = rng.uniform(0.1, 0.9, size=n)
    elif dist < 4:
        # Bimodal: equal halves of small and large items, shuffled.
        half = n // 2
        sizes = np.concatenate(
            [
                rng.uniform(0.05, 0.35, half),
                rng.uniform(0.60, 0.95, n - half),
            ]
        )
        rng.shuffle(sizes)
    else:
        # Gaussian centred at 0.5.
        sizes = rng.normal(0.5, 0.2, size=n)

    return np.clip(sizes, 0.05, 0.95)


def _build_instance_rows(
    args: tuple[int, int, int, int],
) -> tuple[list[list[float]], list[int]]:
    """Build feature/label rows for a single synthetic instance.

    This function is the per-worker unit of work.  It must be a module-level
    function (not a lambda or nested function) so that it is picklable for
    multiprocessing.

    Algorithm
    ---------
    1. Generate a random instance using a child RNG derived from ``inst_seed``.
    2. Build an FFD start solution (items sorted descending, each placed in
       the first bin with enough remaining capacity).
    3. For every item in FFD order:
       a. Find all feasible bins (remaining capacity ≥ item size).
       b. Identify the BFD-optimal bin (smallest post-placement slack) as
          the positive label.
       c. Sample up to ``max_negatives`` other feasible bins as negatives.
       d. Compute the feature vector for each selected bin and record the row.
    4. If no feasible bin exists for an item (should not happen in a valid
       FFD solution, but guarded defensively), the item is skipped.

    Parameters
    ----------
    args:
        Tuple of (inst_seed, n_min, n_max, max_negatives).
        Packed as a tuple so imap can pass a single argument.

    Returns
    -------
    rows_X : list of feature vectors (one per positive/negative placement)
    rows_y : list of labels (1 = positive / BFD-optimal, 0 = negative)
    """
    inst_seed, n_min, n_max, max_negatives = args

    rng = np.random.default_rng(inst_seed)
    mf = features.make_features

    sizes = _generate_instance(rng, n_min, n_max)
    sizes_list = [float(v) for v in sizes]
    n = len(sizes_list)
    capacity = 1.0

    # --- FFD start solution ---------------------------------------------------
    # Sort items by descending size; assign each to the first bin that fits.
    order = sorted(range(n), key=lambda i: -sizes_list[i])
    # size_rank[item] = position in the FFD ordering (0 = largest item).
    size_rank: dict[int, int] = {item: rank for rank, item in enumerate(order)}

    bins: list[list[int]] = []  # bins[j] = list of item indices in bin j
    bin_loads: list[float] = []  # bin_loads[j] = total size of items in bin j

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

    # --- Label each placement decision ----------------------------------------
    rows_X: list[list[float]] = []
    rows_y: list[int] = []

    remaining = len(order)  # items yet to be "decided" (counts down in loop)
    denom = max(1, n)  # denominator for remaining_ratio feature

    for item in order:
        item_size = sizes_list[item]

        # Collect all bins that can accept this item.
        feasible: list[int] = []
        best_bin = -1
        best_slack = float("inf")

        for j, load in enumerate(bin_loads):
            remaining_capacity = capacity - load
            if remaining_capacity + 1e-9 < item_size:
                continue
            feasible.append(j)
            slack = remaining_capacity - item_size
            if slack < best_slack:
                best_slack = slack
                best_bin = j

        if best_bin == -1:
            # Defensive guard: no feasible bin found (should not occur in a
            # valid FFD solution because the item was placed during construction).
            remaining -= 1
            continue

        # Helper: compute the feature vector for placing `item` into `bin_idx`.
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

        # Positive: the BFD-optimal bin.
        rows_X.append(_feat(best_bin))
        rows_y.append(1)

        # Negatives: a random subset of the other feasible bins.
        negatives = [j for j in feasible if j != best_bin]
        if len(negatives) > max_negatives:
            negatives = list(rng.choice(negatives, size=max_negatives, replace=False))
        for j in negatives:
            rows_X.append(_feat(j))
            rows_y.append(0)

        remaining -= 1

    return rows_X, rows_y


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
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, DatasetSummary]:
    """Generate a synthetic (X, y) training dataset.

    For each instance:
      1. Build an FFD start solution.
      2. For every item, find all feasible bins.
      3. Label the minimum-slack bin as positive (BFD oracle).
      4. Sample up to ``max_negatives`` other feasible bins as negatives.

    Each instance is seeded deterministically from ``seed`` via
    ``np.random.SeedSequence``, so the dataset is fully reproducible
    regardless of ``workers``.

    Parameters
    ----------
    instances:
        Number of synthetic instances.
    n_min, n_max:
        Item-count range per instance.
    max_negatives:
        Maximum negative samples per item placement decision.
    seed:
        Root RNG seed.
    workers:
        Number of parallel worker processes.

    Returns
    -------
    X : np.ndarray, shape (n_rows, N_FEATURES), dtype float32
    y : np.ndarray, shape (n_rows,), dtype int32
    summary : DatasetSummary
    """
    worker_count = max(1, int(workers))

    # Derive one child seed per instance so results are reproducible
    # regardless of execution order or worker count.
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = [
        int(cs.generate_state(1, dtype=np.uint64)[0])
        for cs in seed_seq.spawn(instances)
    ]
    job_args = [(s, n_min, n_max, max_negatives) for s in child_seeds]

    all_X: list[list[float]] = []
    all_y: list[int] = []

    pool = None
    try:
        if worker_count == 1:
            # Single-process path: simpler to debug and profile.
            iterator = (
                _build_instance_rows(args)
                for args in tqdm(job_args, desc="Generating dataset", total=instances)
            )
        else:
            # Multi-process path: distribute instances across workers.
            # chunksize=16 amortises IPC overhead without starving workers.
            pool = Pool(processes=worker_count)
            iterator = tqdm(
                pool.imap(_build_instance_rows, job_args, chunksize=16),
                desc="Generating dataset",
                total=instances,
            )

        for rows_X, rows_y in iterator:
            all_X.extend(rows_X)
            all_y.extend(rows_y)

    finally:
        # Always clean up the pool, even if an exception occurred mid-iteration.
        # terminate() is safe here: on the normal path the imap iterator is
        # fully consumed so all workers are already idle; on the exception path
        # it cancels pending tasks immediately without hanging.
        if pool is not None:
            pool.terminate()
            pool.join()

    # Build arrays
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
        The same payload that is serialised to disk.
    """
    print("=" * 70)
    print("DATASET GENERATION")
    print("=" * 70)
    print(f"  Instances          : {config.instances}")
    print(f"  Instance size range: [{config.n_min}, {config.n_max}] items")
    print(f"  Max negatives      : {config.max_negatives}")
    print(f"  Seed               : {config.seed}")
    print(f"  Workers            : {config.workers}")
    print(f"  Output             : {config.output}")

    X, y, summary = build_dataset(
        instances=config.instances,
        n_min=config.n_min,
        n_max=config.n_max,
        max_negatives=config.max_negatives,
        seed=config.seed,
        workers=config.workers,
    )

    _validate_dataset_arrays(X, y, context=str(config.output))

    print(f"\nDataset: {summary.rows:,} rows x {summary.cols} features")
    print(
        f"  Positive rate : {summary.positive_rate:.4f}"
        f"  (expected min: {1.0 / (1 + config.max_negatives):.4f})"
    )
    print(f"  Class 0 (neg) : {(y == 0).sum():,}")
    print(f"  Class 1 (pos) : {(y == 1).sum():,}")

    # Write dataset to disk, creating parent directories as needed.
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "X": X,
        "y": y,
        "feature_version": FEATURE_VERSION,
        "source": "synthetic",
        "summary": {
            "instances": config.instances,
            "n_min": config.n_min,
            "n_max": config.n_max,
            "max_negatives": config.max_negatives,
            "seed": config.seed,
            "workers": config.workers,
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
