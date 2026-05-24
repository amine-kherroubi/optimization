"""Collect real ALNS repair states for training data augmentation (DAgger-lite).

The repair model is trained on BFD decisions (imitation learning). But
during ALNS, the solver visits states that BFD never produces — partially
destroyed solutions with arbitrary residual bin loads. This covariate shift
degrades repair quality on non-BFD states.

This script runs the ALNS solver on synthetic instances and captures the
actual (bins, displaced_items) states encountered during _repair_learned.
Those states are then labelled with the BFD oracle (minimum-slack bin)
and saved as an augmentation dataset.

Usage
-----
    python collect_alns_states.py \\
        --model-path repair_model.pkl \\
        --instances 500 \\
        --output alns_states.pkl

Then pass --augment-with alns_states.pkl to train_repair_model.py.

Why "DAgger-lite"
-----------------
True DAgger interleaves data collection and retraining in a loop.
Here we do a single offline collection pass: run the existing model,
capture the states it visits, label them with the oracle, and add them
to the next training run. This is cheaper and already substantially
reduces covariate shift.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Sequence, Mapping, Tuple

import math
import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training import features

# Local aliases for API compatibility with older script usage
make_features = features.make_features
N_FEATURES = features.N_FEATURES
FEATURE_VERSION = features.FEATURE_VERSION

# ---------------------------------------------------------------------------
# BFD oracle — labels a repair state with the best-fit decision
# ---------------------------------------------------------------------------


def _bfd_label(
    item: int,
    sizes: Sequence[float],
    bins: list[list[int]],
    bin_loads: Sequence[float],
    capacity: float,
    size_rank: Mapping[int, int],
    remaining_ratio: float,
    max_negatives: int,
    rng: np.random.Generator,
) -> Tuple[list[list[float]], list[int]]:
    """Label one (item, existing_bins) repair state with BFD.

    Returns (X_rows, y_rows) — empty if no feasible bin exists for this item.
    The positive is the bin with minimum post-placement slack (BFD choice).
    Negatives are other feasible bins, subsampled to max_negatives.
    """
    item_size = sizes[item]
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
        return [], []

    mf = make_features
    n_total = len(sizes)

    def _feat(bin_idx: int) -> list[float]:
        return mf(
            item=item,
            item_size=item_size,
            bin_items=bins[bin_idx],
            bin_load=bin_loads[bin_idx],
            capacity=capacity,
            sizes=sizes,
            n_total=n_total,
            size_rank=size_rank,
            remaining_ratio=remaining_ratio,
        )

    X: list[list[float]] = [_feat(best_bin)]
    y: list[int] = [1]

    negatives = [j for j in feasible if j != best_bin]
    if len(negatives) > max_negatives:
        negatives = list(rng.choice(negatives, size=max_negatives, replace=False))
    for j in negatives:
        X.append(_feat(j))
        y.append(0)

    return X, y


# ---------------------------------------------------------------------------
# Instance generation (mirrors train_repair_model.generate_instance)
# ---------------------------------------------------------------------------


def _generate_instance(rng: np.random.Generator, n_min: int, n_max: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# ALNS rollout with state capture
# ---------------------------------------------------------------------------


def _run_alns_and_capture(
    sizes: np.ndarray,
    model: Any,
    scaler: Any,
    max_iterations: int,
    max_negatives: int,
    rng: np.random.Generator,
) -> Tuple[list[list[float]], list[int]]:
    """Run ALNS on one instance; capture repair states and label with BFD.

    This mirrors the solver logic but intercepts each repair call to extract
    (bins_state, displaced_items) before the model makes its placement decisions.
    """
    sizes_list = [float(v) for v in sizes]
    n = len(sizes_list)
    capacity = 1.0
    denom = max(1, n)
    mf = make_features
    eps = 1e-9

    # FFD start solution
    order = sorted(range(n), key=lambda i: -sizes_list[i])
    size_rank: dict[int, int] = {item: rank for rank, item in enumerate(order)}

    bins: list[list[int]] = []
    bin_loads: list[float] = []
    item_to_bin: list[int] = [-1] * n

    for item in order:
        s = sizes_list[item]
        placed = False
        for j, load in enumerate(bin_loads):
            if load + s <= capacity:
                bins[j].append(item)
                bin_loads[j] += s
                item_to_bin[item] = j
                placed = True
                break
        if not placed:
            item_to_bin[item] = len(bins)
            bins.append([item])
            bin_loads.append(s)

    k_min = max(1, int(0.05 * n))
    k_max = max(k_min + 1, int(0.25 * n))

    X_captured: list[list[float]] = []
    y_captured: list[int] = []

    t0 = 1.0 / math.log(2.0)
    temperature = t0
    alpha = 0.9995

    for iteration in range(max_iterations):
        if len(bins) <= 1:
            break

        # Random destroy: remove one bin
        j_remove = int(rng.integers(0, len(bins)))
        displaced = list(bins[j_remove])
        bins.pop(j_remove)
        bin_loads.pop(j_remove)
        for item in displaced:
            item_to_bin[item] = -1
        # Rebuild item_to_bin
        for j, b in enumerate(bins):
            for item in b:
                item_to_bin[item] = j

        # Capture state and label each displaced item with BFD oracle
        remaining = len(displaced)
        for item in sorted(displaced, key=lambda i: -sizes_list[i]):
            x_rows, y_rows = _bfd_label(
                item=item,
                sizes=sizes_list,
                bins=bins,
                bin_loads=bin_loads,
                capacity=capacity,
                size_rank=size_rank,
                remaining_ratio=remaining / denom,
                max_negatives=max_negatives,
                rng=rng,
            )
            X_captured.extend(x_rows)
            y_captured.extend(y_rows)

            # Now actually place the item using the ML model (on-policy)
            feats: list[list[float]] = []
            idxs: list[int] = []
            for j, load in enumerate(bin_loads):
                if capacity - load + eps >= sizes_list[item]:
                    feats.append(
                        mf(
                            item=item,
                            item_size=sizes_list[item],
                            bin_items=bins[j],
                            bin_load=load,
                            capacity=capacity,
                            sizes=sizes_list,
                            n_total=n,
                            size_rank=size_rank,
                            remaining_ratio=remaining / denom,
                        )
                    )
                    idxs.append(j)

            if not feats:
                bins.append([item])
                bin_loads.append(sizes_list[item])
                item_to_bin[item] = len(bins) - 1
            else:
                feats_arr = np.asarray(feats, dtype=np.float64)
                if scaler is not None:
                    feats_arr = scaler.transform(feats_arr)
                scores = model.predict_proba(feats_arr)[:, 1]  # type: ignore[union-attr]
                best_j = idxs[int(np.argmax(scores))]
                bins[best_j].append(item)
                bin_loads[best_j] += sizes_list[item]
                item_to_bin[item] = best_j
            remaining -= 1

        temperature *= alpha

    return X_captured, y_captured


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect real ALNS repair states for training augmentation"
    )
    parser.add_argument(
        "--model-path", required=True, help="Existing ../models/repair_model.pkl"
    )
    parser.add_argument(
        "--instances", type=int, default=500, help="Synthetic instances to run"
    )
    parser.add_argument("--n-min", type=int, default=50)
    parser.add_argument("--n-max", type=int, default=200)
    parser.add_argument("--max-negatives", type=int, default=5)
    parser.add_argument(
        "--iterations", type=int, default=200, help="ALNS iterations per instance"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="alns_states.pkl")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Model feature_version={bundle.get('feature_version')} does not match "
            f"current FEATURE_VERSION={FEATURE_VERSION}. Retrain first."
        )

    model = bundle["model"]
    scaler = bundle["scaler"]

    rng = np.random.default_rng(args.seed)
    all_X: list[list[float]] = []
    all_y: list[int] = []

    print(f"Running ALNS on {args.instances} instances to collect repair states...")
    for i in tqdm(
        range(args.instances), desc="Collecting ALNS states", total=args.instances
    ):
        sizes = _generate_instance(rng, args.n_min, args.n_max)
        X_i, y_i = _run_alns_and_capture(
            sizes=sizes,
            model=model,
            scaler=scaler,
            max_iterations=args.iterations,
            max_negatives=args.max_negatives,
            rng=rng,
        )
        all_X.extend(X_i)
        all_y.extend(y_i)

    X_arr = np.asarray(all_X, dtype=np.float32)
    y_arr = np.asarray(all_y, dtype=np.int32)
    pos_rate = float(y_arr.mean()) if len(y_arr) > 0 else 0.0
    print(f"Collected {len(all_X)} rows  (pos_rate={pos_rate:.3f})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"X": X_arr, "y": y_arr, "feature_version": FEATURE_VERSION}, f)
    print(f"Saved to: {output_path}")
    print("Next step: retrain with --augment-with", output_path)


if __name__ == "__main__":
    main()
