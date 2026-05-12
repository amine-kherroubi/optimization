from __future__ import annotations

"""Train the optional learned repair model used by the hybrid ALNS solver.

This script performs offline imitation learning from Best-Fit Decreasing (BFD)
decisions. The resulting pickle can be passed to BinPackingSolver.solve via
`model_path`.
"""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class DatasetSummary:
    rows: int
    cols: int
    positive_rate: float


def generate_instance(rng: np.random.Generator, n_min: int, n_max: int) -> np.ndarray:
    """Generate one synthetic normalized 1D-BPP instance."""
    n = int(rng.integers(n_min, n_max + 1))
    return rng.uniform(0.1, 0.9, size=n)


def extract_training_examples(
    sizes: np.ndarray, max_negatives: int, rng: np.random.Generator
) -> tuple[list[list[float]], list[int]]:
    """Replay BFD and collect (features, label) rows for feasible bin choices.

    Labeling rule:
      - positive (1): the exact feasible bin BFD selected at that step
      - negative (0): other feasible bins (optionally subsampled)
    """
    n_total = int(sizes.shape[0])
    sizes_list = [float(v) for v in sizes]
    order = list(np.argsort(-sizes))
    size_rank = {item: rank for rank, item in enumerate(order)}

    replay_bins: list[list[int]] = []
    replay_loads: list[float] = []
    X: list[list[float]] = []
    y: list[int] = []

    for step, item in enumerate(order):
        item_size = float(sizes[item])
        remaining_ratio = (n_total - step) / max(1, n_total)

        # For the current insertion step, enumerate feasible existing bins and
        # identify the BFD choice (minimum slack after placement).
        feasible_bins: list[int] = []
        best_bin = -1
        best_slack = float("inf")
        for j, load in enumerate(replay_loads):
            rem = 1.0 - load
            if rem + 1e-9 < item_size:
                continue
            feasible_bins.append(j)
            slack = rem - item_size
            if slack < best_slack:
                best_slack = slack
                best_bin = j

        # If no existing bin can fit, BFD opens a new bin. There is no
        # bin-selection classification sample for this step.
        if best_bin == -1:
            replay_bins.append([item])
            replay_loads.append(item_size)
            continue

        X.append(
            _make_features(
                item,
                best_bin,
                replay_bins,
                replay_loads,
                sizes_list,
                size_rank,
                remaining_ratio,
            )
        )
        y.append(1)

        # Negatives are all alternative feasible bins, optionally capped to keep
        # class imbalance manageable and training memory bounded.
        negatives = [j for j in feasible_bins if j != best_bin]
        if len(negatives) > max_negatives:
            negatives = list(rng.choice(negatives, size=max_negatives, replace=False))
        for j in negatives:
            X.append(
                _make_features(
                    item,
                    j,
                    replay_bins,
                    replay_loads,
                    sizes_list,
                    size_rank,
                    remaining_ratio,
                )
            )
            y.append(0)

        replay_bins[best_bin].append(item)
        replay_loads[best_bin] += item_size

    return X, y


def _make_features(
    item: int,
    bin_idx: int,
    bins: list[list[int]],
    bin_loads: list[float],
    sizes: list[float],
    size_rank: dict[int, int],
    remaining_ratio: float,
) -> list[float]:
    """Feature contract for training (must match solver inference exactly)."""
    n_total = len(sizes)
    s = sizes[item]
    load = float(bin_loads[bin_idx])
    rem = 1.0 - load
    slack_after = rem - s
    members = bins[bin_idx]
    largest = max((sizes[k] for k in members), default=0.0)
    smallest = min((sizes[k] for k in members), default=0.0)
    return [
        s,
        s * s,
        s,
        size_rank[item] / max(1, n_total),
        remaining_ratio,
        load,
        rem,
        slack_after,
        slack_after,
        load,
        len(members) / max(1, n_total),
        largest,
        smallest,
        (s / rem) if rem > 1e-9 else 1.0,
    ]


def build_dataset(
    instances: int,
    n_min: int,
    n_max: int,
    max_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, DatasetSummary]:
    """Generate synthetic data and return X, y ready for sklearn.

    The synthetic distribution is intentionally simple (uniform sizes in [0.1, 0.9])
    so data generation remains fast while still exposing non-trivial packing choices.
    """
    if instances <= 0:
        raise ValueError("instances must be > 0")
    if n_min <= 0 or n_max <= 0 or n_min > n_max:
        raise ValueError("n_min and n_max must be positive with n_min <= n_max")
    if max_negatives < 0:
        raise ValueError("max_negatives must be >= 0")

    rng = np.random.default_rng(seed)
    all_x: list[list[float]] = []
    all_y: list[int] = []

    for i in range(instances):
        if i and i % 500 == 0:
            print(f"Generated {i}/{instances} instances...")
        sizes = generate_instance(rng, n_min=n_min, n_max=n_max)
        x_inst, y_inst = extract_training_examples(
            sizes, max_negatives=max_negatives, rng=rng
        )
        all_x.extend(x_inst)
        all_y.extend(y_inst)

    X = np.asarray(all_x, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.int32)
    summary = DatasetSummary(
        rows=int(X.shape[0]), cols=int(X.shape[1]), positive_rate=float(y.mean())
    )
    return X, y, summary


def main() -> None:
    # CLI flags are kept explicit so teammates can reproduce data/model artifacts
    # with identical settings when comparing approaches.
    parser = argparse.ArgumentParser(
        description="Train repair model for hybrid ALNS bin packing solver"
    )
    parser.add_argument("--instances", type=int, default=5000)
    parser.add_argument("--n-min", type=int, default=50)
    parser.add_argument("--n-max", type=int, default=200)
    parser.add_argument("--max-negatives", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=str, default="5_hybrid_ml_metaheuristics/repair_model.pkl"
    )
    args = parser.parse_args()

    X, y, summary = build_dataset(
        instances=args.instances,
        n_min=args.n_min,
        n_max=args.n_max,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )
    print(
        f"Dataset: rows={summary.rows}, cols={summary.cols}, pos_rate={summary.positive_rate:.3f}"
    )

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
    model.fit(x_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(x_val))
    print(f"Validation accuracy: {val_acc:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
