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

from typing import Sequence, Mapping
import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - numba optional
    njit = None


def make_features(
    *,
    item: int,
    item_size: float,
    bin_items: Sequence[int],
    bin_load: float,
    capacity: float,
    sizes: Sequence[float],
    n_total: int,
    size_rank: Mapping[int, int],
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

    # Compute largest/smallest in a short local loop to avoid generator overhead
    if bin_items:
        it = iter(bin_items)
        first = next(it)
        max_v = sizes[first]
        min_v = max_v
        for k in it:
            v = sizes[k]
            if v > max_v:
                max_v = v
            if v < min_v:
                min_v = v
        largest = max_v
        smallest = min_v
    else:
        largest = 0.0
        smallest = 0.0

    feat = [0.0] * N_FEATURES
    inv_capacity = 1.0 / capacity
    feat[_F_ITEM_SIZE] = item_size * inv_capacity
    feat[_F_ITEM_SIZE_SQ] = (item_size * inv_capacity) ** 2
    feat[_F_SIZE_RANK] = size_rank.get(item, 0) / max(1, n_total)
    feat[_F_REMAINING] = remaining_ratio
    feat[_F_BIN_LOAD] = bin_load * inv_capacity
    feat[_F_BIN_REM] = remaining_capacity * inv_capacity
    feat[_F_SLACK_AFTER] = slack_after * inv_capacity
    feat[_F_BIN_COUNT] = len(bin_items) / max(1, n_total)
    feat[_F_BIN_LARGEST] = largest * inv_capacity
    feat[_F_BIN_SMALLEST] = smallest * inv_capacity
    feat[_F_FILL_RATIO] = (
        (item_size / remaining_capacity) if remaining_capacity > 1e-9 else 1.0
    )
    return feat


if njit is not None:
    # numba-friendly implementation operating on raw arrays/ints. We keep the
    # Python version as the default for readability and for environments
    # without numba. The JIT expects primitive types and simple loops.
    @njit(cache=True)
    def make_features_jit(
        item: int,
        item_size: float,
        bin_items_arr,
        bin_items_len: int,
        bin_load: float,
        capacity: float,
        sizes_arr,
        n_total: int,
        size_ranks_arr,
        remaining_ratio: float,
    ):
        feat = [0.0] * N_FEATURES
        remaining_capacity = capacity - bin_load
        slack_after = remaining_capacity - item_size
        if bin_items_len > 0:
            first = bin_items_arr[0]
            max_v = sizes_arr[first]
            min_v = max_v
            for ii in range(1, bin_items_len):
                k = bin_items_arr[ii]
                v = sizes_arr[k]
                if v > max_v:
                    max_v = v
                if v < min_v:
                    min_v = v
            largest = max_v
            smallest = min_v
        else:
            largest = 0.0
            smallest = 0.0

        inv_capacity = 1.0 / capacity
        feat[_F_ITEM_SIZE] = item_size * inv_capacity
        feat[_F_ITEM_SIZE_SQ] = (item_size * inv_capacity) ** 2
        feat[_F_SIZE_RANK] = size_ranks_arr[item] / max(1, n_total)
        feat[_F_REMAINING] = remaining_ratio
        feat[_F_BIN_LOAD] = bin_load * inv_capacity
        feat[_F_BIN_REM] = remaining_capacity * inv_capacity
        feat[_F_SLACK_AFTER] = slack_after * inv_capacity
        feat[_F_BIN_COUNT] = bin_items_len / max(1, n_total)
        feat[_F_BIN_LARGEST] = largest * inv_capacity
        feat[_F_BIN_SMALLEST] = smallest * inv_capacity
        feat[_F_FILL_RATIO] = (
            (item_size / remaining_capacity) if remaining_capacity > 1e-9 else 1.0
        )
        return feat

    @njit(cache=True)
    def make_features_batch_jit(
        items_arr,
        item_sizes_arr,
        bin_items_flat,
        bin_offsets,
        bin_loads_arr,
        capacity,
        sizes_arr,
        n_total,
        size_ranks_arr,
        remaining_ratio_arr,
    ):
        n = items_arr.shape[0]
        out = np.empty((n, N_FEATURES), dtype=np.float64)
        for i in range(n):
            item = int(items_arr[i])
            item_size = float(item_sizes_arr[i])
            start = int(bin_offsets[i])
            end = int(bin_offsets[i + 1])
            if end > start:
                first = int(bin_items_flat[start])
                max_v = sizes_arr[first]
                min_v = max_v
                for ii in range(start + 1, end):
                    k = int(bin_items_flat[ii])
                    v = sizes_arr[k]
                    if v > max_v:
                        max_v = v
                    if v < min_v:
                        min_v = v
                largest = max_v
                smallest = min_v
            else:
                largest = 0.0
                smallest = 0.0

            remaining_capacity = capacity - float(bin_loads_arr[i])
            slack_after = remaining_capacity - item_size
            inv_capacity = 1.0 / capacity
            out[i, _F_ITEM_SIZE] = item_size * inv_capacity
            out[i, _F_ITEM_SIZE_SQ] = (item_size * inv_capacity) ** 2
            out[i, _F_SIZE_RANK] = size_ranks_arr[item] / max(1, n_total)
            out[i, _F_REMAINING] = float(remaining_ratio_arr[i])
            out[i, _F_BIN_LOAD] = float(bin_loads_arr[i]) * inv_capacity
            out[i, _F_BIN_REM] = remaining_capacity * inv_capacity
            out[i, _F_SLACK_AFTER] = slack_after * inv_capacity
            out[i, _F_BIN_COUNT] = (end - start) / max(1, n_total)
            out[i, _F_BIN_LARGEST] = largest * inv_capacity
            out[i, _F_BIN_SMALLEST] = smallest * inv_capacity
            out[i, _F_FILL_RATIO] = (
                (item_size / remaining_capacity) if remaining_capacity > 1e-9 else 1.0
            )
        return out

    def make_features_batch_py(
        items_arr,
        item_sizes_arr,
        bin_items_flat,
        bin_offsets,
        bin_loads_arr,
        capacity,
        sizes_arr,
        n_total,
        size_ranks_arr,
        remaining_ratio_arr,
    ):
        n = int(items_arr.shape[0])
        out = np.empty((n, N_FEATURES), dtype=np.float64)
        for i in range(n):
            item = int(items_arr[i])
            item_size = float(item_sizes_arr[i])
            start = int(bin_offsets[i])
            end = int(bin_offsets[i + 1])
            if end > start:
                first = int(bin_items_flat[start])
                max_v = sizes_arr[first]
                min_v = max_v
                for ii in range(start + 1, end):
                    k = int(bin_items_flat[ii])
                    v = sizes_arr[k]
                    if v > max_v:
                        max_v = v
                    if v < min_v:
                        min_v = v
                largest = max_v
                smallest = min_v
            else:
                largest = 0.0
                smallest = 0.0

            remaining_capacity = capacity - float(bin_loads_arr[i])
            slack_after = remaining_capacity - item_size
            inv_capacity = 1.0 / capacity
            out[i, _F_ITEM_SIZE] = item_size * inv_capacity
            out[i, _F_ITEM_SIZE_SQ] = (item_size * inv_capacity) ** 2
            out[i, _F_SIZE_RANK] = size_ranks_arr[item] / max(1, n_total)
            out[i, _F_REMAINING] = float(remaining_ratio_arr[i])
            out[i, _F_BIN_LOAD] = float(bin_loads_arr[i]) * inv_capacity
            out[i, _F_BIN_REM] = remaining_capacity * inv_capacity
            out[i, _F_SLACK_AFTER] = slack_after * inv_capacity
            out[i, _F_BIN_COUNT] = (end - start) / max(1, n_total)
            out[i, _F_BIN_LARGEST] = largest * inv_capacity
            out[i, _F_BIN_SMALLEST] = smallest * inv_capacity
            out[i, _F_FILL_RATIO] = (
                (item_size / remaining_capacity) if remaining_capacity > 1e-9 else 1.0
            )
        return out
