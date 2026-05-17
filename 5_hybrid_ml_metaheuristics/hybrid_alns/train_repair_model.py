"""Train the optional learned repair model used by the hybrid ALNS solver.

This script performs offline imitation learning from Best-Fit Decreasing (BFD)
decisions. The resulting pickle can be passed to BinPackingSolver.solve via
`model_path`.

Two complementary data sources are used to reduce distribution shift:

  1. Fresh BFD traces — items inserted one by one into an empty solution in
     decreasing size order. This captures clean, high-quality BFD decisions.
     At each step where at least one existing bin is feasible, the chosen bin
     (minimum slack) is labeled positive and any alternatives are labeled negative.

  2. Post-destruction repair traces — a full BFD solution is built first, a
     random fraction of bins is evicted, and the displaced items are re-inserted
     with BFD into the surviving bins. This mirrors the state the solver
     actually encounters during ALNS repair: existing bins hold arbitrary
     residual loads rather than freshly packed sequences.

The label at every step is binary:
  - 1 (positive): the bin BFD selected (minimum slack after placement)
  - 0 (negative): other feasible bins, subsampled to cap class imbalance

The saved pickle is a dict containing the fitted model, the fitted scaler,
and metadata needed by the solver to detect version mismatches:
  {"model": ..., "scaler": ..., "feature_version": ..., "n_features": ...}
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        """Minimal no-op fallback when tqdm is not installed."""
        desc = kwargs.get("desc", "")
        if desc:
            print(f"{desc}...")
        return iterable


from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from features import FEATURE_VERSION, N_FEATURES, make_features as _make_features  # noqa: E402


@dataclass(slots=True)
class DatasetSummary:
    rows: int
    cols: int
    positive_rate: float


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------



def generate_instance(rng: np.random.Generator, n_min: int, n_max: int) -> np.ndarray:
    """Return one synthetic normalized 1-D BPP instance.

    Sizes are drawn uniformly from [0.1, 0.9] so the bin capacity is implicitly
    1.0 throughout the training pipeline. Features produced here are therefore
    already in the same [0, 1] scale as those the solver computes by dividing
    integer sizes by integer capacity.
    """
    n = int(rng.integers(n_min, n_max + 1))
    return rng.uniform(0.1, 0.9, size=n)


# ---------------------------------------------------------------------------
# Example extraction — fresh BFD traces
# ---------------------------------------------------------------------------


def extract_training_examples(
    sizes: np.ndarray, max_negatives: int, rng: np.random.Generator
) -> tuple[list[list[float]], list[int]]:
    """Replay BFD on a fresh instance and collect (features, label) rows.

    Items are processed in decreasing size order (the BFD convention). At each
    step where at least one existing bin is feasible, the chosen bin (tightest
    fit) is labeled 1. Any other feasible bins at that step are labeled 0 and
    subsampled to at most max_negatives. Steps that open a new bin produce no
    classification sample because there is no bin-selection decision to learn.

    remaining_ratio at step k is (n_total - k) / n_total, counting the current
    item as still-remaining. Fresh traces effectively model a full-destruction
    repair scenario where n_displaced = n_total, so this formula is the special
    case of extract_repair_examples' (n_displaced - step) / n_total with
    n_displaced = n_total. The inference code in solver._repair_learned uses
    the same denominator (n_total) and the same numerator (displaced items still
    to place), so all three contexts are consistent.
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
        # Fraction of items (including this one) not yet inserted.
        remaining_ratio = (n_total - step) / max(1, n_total)

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

        if best_bin == -1:
            # No existing bin is feasible; open a new one and move on.
            replay_bins.append([item])
            replay_loads.append(item_size)
            continue

        X.append(
            _make_features(
                item=item,
                item_size=sizes_list[item],
                bin_items=replay_bins[best_bin],
                bin_load=replay_loads[best_bin],
                capacity=1.0,
                sizes=sizes_list,
                n_total=n_total,
                size_rank=size_rank,
                remaining_ratio=remaining_ratio,
            )
        )
        y.append(1)

        # Subsample negatives so the positive rate stays bounded regardless of
        # how many bins happen to be open. Steps with only one feasible bin
        # contribute a positive with no negatives, which is expected and correct.
        negatives = [j for j in feasible_bins if j != best_bin]
        if len(negatives) > max_negatives:
            negatives = list(rng.choice(negatives, size=max_negatives, replace=False))
        for j in negatives:
            X.append(
                _make_features(
                    item=item,
                    item_size=sizes_list[item],
                    bin_items=replay_bins[j],
                    bin_load=replay_loads[j],
                    capacity=1.0,
                    sizes=sizes_list,
                    n_total=n_total,
                    size_rank=size_rank,
                    remaining_ratio=remaining_ratio,
                )
            )
            y.append(0)

        replay_bins[best_bin].append(item)
        replay_loads[best_bin] += item_size

    return X, y


# ---------------------------------------------------------------------------
# Example extraction — post-destruction repair traces
# ---------------------------------------------------------------------------


def extract_repair_examples(
    sizes: np.ndarray,
    max_negatives: int,
    rng: np.random.Generator,
    destroy_fraction: float = 0.20,
) -> tuple[list[list[float]], list[int]]:
    """Build a BFD solution, destroy part of it, then collect re-insertion labels.

    The ALNS repair model is applied after destroy operators have evicted items
    from a partially optimised solution, not during a fresh BFD packing. The
    surviving bins therefore hold arbitrary residual loads that BFD never
    produced. Training on examples generated this way reduces the gap between
    the distribution the model was trained on and the distribution it sees at
    inference time (distribution shift / covariate shift).

    Labeling follows the same BFD rule: among feasible existing bins, the one
    with minimum post-placement slack is labeled 1. If no existing bin fits,
    a new bin is opened and no classification sample is emitted.

    remaining_ratio at step k is (n_displaced - k) / n_total, counting the
    current item as still-remaining. This convention matches extract_training_examples
    and solver._repair_learned.
    """
    n_total = int(sizes.shape[0])
    sizes_list = [float(v) for v in sizes]
    order = list(np.argsort(-sizes))
    size_rank = {item: rank for rank, item in enumerate(order)}

    # --- Phase 1: build a complete BFD solution. ---
    bins: list[list[int]] = []
    bin_loads: list[float] = []
    for item in order:
        item_size = float(sizes[item])
        best_bin = -1
        best_slack = float("inf")
        for j, load in enumerate(bin_loads):
            rem = 1.0 - load
            if rem + 1e-9 >= item_size:
                slack = rem - item_size
                if slack < best_slack:
                    best_slack = slack
                    best_bin = j
        if best_bin == -1:
            bins.append([item])
            bin_loads.append(item_size)
        else:
            bins[best_bin].append(item)
            bin_loads[best_bin] += item_size

    if len(bins) < 2:
        # Trivial instance — nothing meaningful to destroy.
        return [], []

    # --- Phase 2: randomly destroy some bins. ---
    # Always keep at least one bin so there are surviving bins to repack into.
    n_destroy = max(1, int(len(bins) * destroy_fraction))
    n_destroy = min(n_destroy, len(bins) - 1)
    destroy_idxs = sorted(
        rng.choice(len(bins), size=n_destroy, replace=False).tolist(),
        reverse=True,
    )
    displaced: list[int] = []
    for j in destroy_idxs:
        displaced.extend(bins[j])
        bins.pop(j)
        bin_loads.pop(j)

    if not displaced:
        return [], []

    # --- Phase 3: re-insert displaced items with BFD and collect examples. ---
    # Sort by decreasing size, matching the repair order in the solver.
    displaced_sorted = sorted(displaced, key=lambda i: -sizes[i])
    n_displaced = len(displaced_sorted)
    X: list[list[float]] = []
    y: list[int] = []

    for step, item in enumerate(displaced_sorted):
        item_size = float(sizes[item])
        # Fraction of displaced items (including this one) not yet re-inserted,
        # normalized by total instance size so the scale matches fresh traces.
        remaining_ratio = (n_displaced - step) / max(1, n_total)

        feasible_bins: list[int] = []
        best_bin = -1
        best_slack = float("inf")
        for j, load in enumerate(bin_loads):
            rem = 1.0 - load
            if rem + 1e-9 < item_size:
                continue
            feasible_bins.append(j)
            slack = rem - item_size
            if slack < best_slack:
                best_slack = slack
                best_bin = j

        if best_bin == -1:
            bins.append([item])
            bin_loads.append(item_size)
            continue

        X.append(
            _make_features(
                item=item,
                item_size=sizes_list[item],
                bin_items=bins[best_bin],
                bin_load=bin_loads[best_bin],
                capacity=1.0,
                sizes=sizes_list,
                n_total=n_total,
                size_rank=size_rank,
                remaining_ratio=remaining_ratio,
            )
        )
        y.append(1)

        negatives = [j for j in feasible_bins if j != best_bin]
        if len(negatives) > max_negatives:
            negatives = list(rng.choice(negatives, size=max_negatives, replace=False))
        for j in negatives:
            X.append(
                _make_features(
                    item=item,
                    item_size=sizes_list[item],
                    bin_items=bins[j],
                    bin_load=bin_loads[j],
                    capacity=1.0,
                    sizes=sizes_list,
                    n_total=n_total,
                    size_rank=size_rank,
                    remaining_ratio=remaining_ratio,
                )
            )
            y.append(0)

        bins[best_bin].append(item)
        bin_loads[best_bin] += item_size

    return X, y


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def build_dataset(
    instances: int,
    n_min: int,
    n_max: int,
    max_negatives: int,
    seed: int,
    destroy_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, DatasetSummary]:
    """Generate synthetic data and return X, y arrays ready for sklearn.

    Each synthetic instance contributes two batches of rows: one from a fresh
    BFD trace and one from a post-destruction repair trace (see module docstring
    for rationale). The two batches are pooled before splitting.
    """
    if instances <= 0:
        raise ValueError("instances must be > 0")
    if n_min <= 0 or n_max <= 0 or n_min > n_max:
        raise ValueError("n_min and n_max must be positive with n_min <= n_max")
    if max_negatives < 0:
        raise ValueError("max_negatives must be >= 0")
    if not 0.0 < destroy_fraction < 1.0:
        raise ValueError("destroy_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    all_x: list[list[float]] = []
    all_y: list[int] = []

    for _ in tqdm(
        range(instances), desc="Generating instances", unit="inst", dynamic_ncols=True
    ):
        sizes = generate_instance(rng, n_min=n_min, n_max=n_max)

        x_fresh, y_fresh = extract_training_examples(
            sizes, max_negatives=max_negatives, rng=rng
        )
        x_repair, y_repair = extract_repair_examples(
            sizes,
            max_negatives=max_negatives,
            rng=rng,
            destroy_fraction=destroy_fraction,
        )
        all_x.extend(x_fresh)
        all_y.extend(y_fresh)
        all_x.extend(x_repair)
        all_y.extend(y_repair)

    X = np.asarray(all_x, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.int32)
    summary = DatasetSummary(
        rows=int(X.shape[0]),
        cols=int(X.shape[1]),
        positive_rate=float(y.mean()),
    )
    return X, y, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train repair model for hybrid ALNS bin packing solver"
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=5000,
        help="Number of synthetic BPP instances to generate",
    )
    parser.add_argument(
        "--n-min", type=int, default=50, help="Minimum number of items per instance"
    )
    parser.add_argument(
        "--n-max", type=int, default=200, help="Maximum number of items per instance"
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=5,
        help="Maximum negative samples per positive (caps class imbalance)",
    )
    parser.add_argument(
        "--destroy-fraction",
        type=float,
        default=0.20,
        help="Fraction of bins destroyed in repair augmentation traces",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="repair_model.pkl",
        help="Output path for the saved model bundle",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    print("Generating dataset...")
    X, y, summary = build_dataset(
        instances=args.instances,
        n_min=args.n_min,
        n_max=args.n_max,
        max_negatives=args.max_negatives,
        destroy_fraction=args.destroy_fraction,
        seed=args.seed,
    )

    # The positive rate is always >= 1/(1+max_negatives) by construction: even
    # in the worst case where every step contributes exactly max_negatives
    # negatives, the ratio is 1/(1+max_negatives). Steps with fewer negatives
    # (including 0) can only raise it. Any observed rate below this value
    # indicates a bug in the extraction logic.
    expected_lower_bound = 1.0 / (1.0 + args.max_negatives)
    print(
        f"Dataset: rows={summary.rows}, cols={summary.cols}, "
        f"pos_rate={summary.positive_rate:.3f} "
        f"(theoretical lower bound: {expected_lower_bound:.3f})"
    )
    if summary.positive_rate > 0.70:
        raise RuntimeError(
            f"Positive rate {summary.positive_rate:.3f} is implausibly high — "
            "negative sampling may be broken. Check extraction logic."
        )
    if summary.positive_rate < expected_lower_bound:
        raise RuntimeError(
            f"Positive rate {summary.positive_rate:.3f} is below the theoretical "
            f"lower bound {expected_lower_bound:.3f}. Check extraction logic."
        )

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------

    # A held-out test set (never seen during CV) gives an unbiased estimate
    # of generalisation performance.
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    # Wrapping StandardScaler and LogisticRegressionCV in a Pipeline ensures
    # the scaler is re-fit within each CV fold rather than once on all of
    # x_trainval, preventing data leakage into the validation folds.
    #
    # saga supports class_weight and scales well to large datasets (stochastic
    # updates). class_weight='balanced' compensates for the positive-to-negative
    # imbalance without requiring manual weight tuning.
    #
    # We optimise for ROC-AUC rather than accuracy because the solver uses
    # predict_proba scores to rank candidate bins — calibrated ranking quality
    # matters more than hard-decision accuracy at the 0.5 threshold.
    n_cs = 4
    n_folds = 5
    n_l1_ratios = 1  # only (0,) — pure L2
    total_fits = n_cs * n_folds * n_l1_ratios
    print(
        f"Training model  (Pipeline · LogisticRegressionCV · saga · L2 · "
        f"{n_folds}-fold CV · {n_cs} C values = {total_fits} fits)..."
    )
    t_train_start = time.perf_counter()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=[0.01, 0.1, 1.0, 10.0],
                    cv=5,
                    solver="saga",
                    # penalty="l2" was deprecated in sklearn 1.8 (removed in 1.10).
                    # Use l1_ratios=(0,) instead: elasticnet with l1_ratio=0 is
                    # mathematically identical to pure L2 regularisation.
                    l1_ratios=(0,),
                    class_weight="balanced",
                    max_iter=2000,
                    scoring="roc_auc",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(x_trainval, y_trainval)
    t_train_elapsed = time.perf_counter() - t_train_start
    print(f"Training complete in {t_train_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Evaluation on held-out test set
    # ------------------------------------------------------------------

    test_proba = pipe.predict_proba(x_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    # C_ has shape (n_classes,) = (1,) for binary problems; use flat[0] to get
    # a scalar safely across all sklearn and NumPy 2.x versions.
    best_c = float(pipe.named_steps["clf"].C_.flat[0])
    print(f"Test ROC-AUC: {test_auc:.4f}  (selected C={best_c:.4g})")

    # A well-trained model on this task should achieve AUC > 0.75; lower values
    # suggest the feature contract has drifted or data generation is broken.
    if test_auc < 0.70:
        print(
            f"WARNING: test AUC {test_auc:.4f} is below 0.70. "
            "Consider more training instances or inspect the feature contract."
        )

    # ------------------------------------------------------------------
    # Save model bundle
    # ------------------------------------------------------------------

    # Extract fitted components from the pipeline. After pipe.fit(), the scaler
    # has been re-fit on all of x_trainval (the final fit after CV selects C),
    # which is what the solver needs for inference.
    #
    # The bundle includes everything the solver needs to reproduce the exact
    # inference pipeline. Versioning allows the solver to detect and reject
    # stale model files when the feature contract changes.
    payload = {
        "model": pipe.named_steps["clf"],
        "scaler": pipe.named_steps["scaler"],
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved model bundle to: {output_path}")


if __name__ == "__main__":
    main()
