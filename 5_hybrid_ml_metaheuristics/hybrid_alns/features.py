"""Shared feature engineering for the hybrid ALNS bin packing solver.

Imported by both solver.py (inference) and train_repair_model.py (training)
so the 11-feature contract is defined once and never silently diverges.

Increment FEATURE_VERSION whenever the feature vector changes; the solver
validates this at model-load time to reject stale bundles.
"""

from __future__ import annotations

FEATURE_VERSION = 2
N_FEATURES = 11

# Feature index constants — used for documentation and coefficient analysis.
_F_ITEM_SIZE = 0
_F_ITEM_SIZE_SQ = 1
_F_SIZE_RANK = 2
_F_REMAINING = 3
_F_BIN_LOAD = 4
_F_BIN_REM = 5
_F_SLACK_AFTER = 6
_F_BIN_COUNT = 7
_F_BIN_LARGEST = 8
_F_BIN_SMALLEST = 9
_F_FILL_RATIO = 10


def make_features(
    *,
    item: int,
    item_size: float,
    bin_items: list[int],
    bin_load: float,
    capacity: float,
    sizes: list[float],
    n_total: int,
    size_rank: dict[int, int],
    remaining_ratio: float,
) -> list[float]:
    """Return the 11-dimensional feature vector for a (item, bin) candidate pair.

    All values are normalized by *capacity* so that training (implicit
    capacity = 1.0) and inference (integer capacity) produce identical
    feature distributions.

    Index  Feature
    -----  -------
      0    item_size / C               — normalized item size
      1    (item_size / C)^2           — squared size (non-linear fill effect)
      2    rank(item) / n              — size rank as fraction of all items
      3    remaining_ratio             — scheduling-progress signal
      4    bin_load / C                — current bin utilization
      5    remaining_capacity / C      — residual bin capacity
      6    slack_after / C             — post-placement residual
      7    |B_j| / n                  — bin occupancy count, normalized
      8    max(s_k, k in B_j) / C     — largest item in bin
      9    min(s_k, k in B_j) / C     — smallest item in bin
     10    item_size / remaining_cap   — fill ratio (tightness of fit)
    """
    remaining_capacity = capacity - bin_load
    slack_after = remaining_capacity - item_size
    largest = max((sizes[k] for k in bin_items), default=0.0)
    smallest = min((sizes[k] for k in bin_items), default=0.0)

    feat = [0.0] * N_FEATURES
    feat[_F_ITEM_SIZE] = item_size / capacity
    feat[_F_ITEM_SIZE_SQ] = (item_size / capacity) ** 2
    feat[_F_SIZE_RANK] = size_rank.get(item, 0) / max(1, n_total)
    feat[_F_REMAINING] = remaining_ratio
    feat[_F_BIN_LOAD] = bin_load / capacity
    feat[_F_BIN_REM] = remaining_capacity / capacity
    feat[_F_SLACK_AFTER] = slack_after / capacity
    feat[_F_BIN_COUNT] = len(bin_items) / max(1, n_total)
    feat[_F_BIN_LARGEST] = largest / capacity
    feat[_F_BIN_SMALLEST] = smallest / capacity
    feat[_F_FILL_RATIO] = (
        (item_size / remaining_capacity) if remaining_capacity > 1e-9 else 1.0
    )
    return feat
