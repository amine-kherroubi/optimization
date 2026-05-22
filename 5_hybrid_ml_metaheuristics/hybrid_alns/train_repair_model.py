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


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

import features
from features import (
    FEATURE_VERSION,
    N_FEATURES,
    make_features as _make_features,
)  # noqa: E402


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

    Three distributions are mixed so the model generalises beyond the uniform
    case. All sizes are clipped to (0.05, 0.95) to stay well within the
    implicit capacity of 1.0.

    Distribution probabilities (chosen to balance coverage):
      40% — Uniform[0.1, 0.9]  — classic benchmark distribution
      40% — Bimodal             — many small + many large items
      20% — Gaussian N(0.5,0.2) — items clustered around the midpoint
    """
    n = int(rng.integers(n_min, n_max + 1))
    dist = rng.integers(0, 5)  # 0-1 → uniform, 2-3 → bimodal, 4 → gaussian

    if dist < 2:
        sizes = rng.uniform(0.1, 0.9, size=n)
    elif dist < 4:
        half = n // 2
        small = rng.uniform(0.05, 0.35, size=half)
        large = rng.uniform(0.60, 0.95, size=n - half)
        sizes = np.concatenate([small, large])
        rng.shuffle(sizes)
    else:
        sizes = rng.normal(0.5, 0.2, size=n)

    return np.clip(sizes, 0.05, 0.95)


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
    # Prepare arrays for optional numba-jitted path (avoids per-call conversion)
    sizes_arr = np.array(sizes_list, dtype=np.float64)
    size_ranks_arr = np.array([size_rank[i] for i in range(n_total)], dtype=np.int64)
    eps = 1e-9

    # Prefer the numba jitted implementation when available. The wrapper keeps
    # the same keyword-only signature so call sites need not change.
    if getattr(features, "njit", None) is not None and hasattr(
        features, "make_features_jit"
    ):

        def mf(
            *,
            item,
            item_size,
            bin_items,
            bin_load,
            capacity,
            sizes,
            n_total,
            size_rank,
            remaining_ratio,
        ):
            bin_items_arr = np.array(bin_items, dtype=np.int64)
            return features.make_features_jit(
                item,
                float(item_size),
                bin_items_arr,
                bin_items_arr.shape[0],
                float(bin_load),
                float(capacity),
                sizes_arr,
                n_total,
                size_ranks_arr,
                float(remaining_ratio),
            )

    else:
        mf = _make_features

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
            if rem + eps < item_size:
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

        f = mf(
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
        # Ensure a Python list row is returned (type checkers expect list[list[float]]).
        if isinstance(f, np.ndarray):
            X.append(f.tolist())
        else:
            X.append(list(f))
        y.append(1)

        # Subsample negatives so the positive rate stays bounded regardless of
        # how many bins happen to be open. Steps with only one feasible bin
        # contribute a positive with no negatives, which is expected and correct.
        negatives = [j for j in feasible_bins if j != best_bin]
        if len(negatives) > max_negatives:
            negatives = list(rng.choice(negatives, size=max_negatives, replace=False))
        for j in negatives:
            g = mf(
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
            if isinstance(g, np.ndarray):
                X.append(g.tolist())
            else:
                X.append(list(g))
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
    # Prepare arrays for optional numba-jitted batch computation
    sizes_arr = np.array(sizes_list, dtype=np.float64)
    size_ranks_arr = np.array([size_rank[i] for i in range(n_total)], dtype=np.int64)
    eps = 1e-9

    # --- Phase 1: build a complete BFD solution. ---
    bins: list[list[int]] = []
    bin_loads: list[float] = []
    for item in order:
        item_size = float(sizes[item])
        best_bin = -1
        best_slack = float("inf")
        for j, load in enumerate(bin_loads):
            rem = 1.0 - load
            if rem + eps >= item_size:
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

    items_all: list[int] = []
    item_sizes_all: list[float] = []
    bin_items_flat: list[int] = []
    offsets: list[int] = [0]
    bin_loads_all: list[float] = []
    remaining_ratio_all: list[float] = []
    labels: list[int] = []

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
            if rem + eps < item_size:
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

        # Positive sample
        items_all.append(item)
        item_sizes_all.append(item_size)
        for k in bins[best_bin]:
            bin_items_flat.append(k)
        offsets.append(len(bin_items_flat))
        bin_loads_all.append(bin_loads[best_bin])
        remaining_ratio_all.append(remaining_ratio)
        labels.append(1)

        negatives = [j for j in feasible_bins if j != best_bin]
        if len(negatives) > max_negatives:
            negatives = list(rng.choice(negatives, size=max_negatives, replace=False))
        for j in negatives:
            items_all.append(item)
            item_sizes_all.append(item_size)
            for k in bins[j]:
                bin_items_flat.append(k)
            offsets.append(len(bin_items_flat))
            bin_loads_all.append(bin_loads[j])
            remaining_ratio_all.append(remaining_ratio)
            labels.append(0)

        bins[best_bin].append(item)
        bin_loads[best_bin] += item_size

    if not items_all:
        return [], []

    items_arr = np.array(items_all, dtype=np.int64)
    item_sizes_arr = np.array(item_sizes_all, dtype=np.float64)
    bin_items_flat_arr = np.array(bin_items_flat, dtype=np.int64)
    bin_offsets_arr = np.array(offsets, dtype=np.int64)
    bin_loads_arr = np.array(bin_loads_all, dtype=np.float64)
    remaining_ratio_arr = np.array(remaining_ratio_all, dtype=np.float64)

    if getattr(features, "njit", None) is not None and hasattr(
        features, "make_features_batch_jit"
    ):
        feats = features.make_features_batch_jit(
            items_arr,
            item_sizes_arr,
            bin_items_flat_arr,
            bin_offsets_arr,
            bin_loads_arr,
            1.0,
            sizes_arr,
            n_total,
            size_ranks_arr,
            remaining_ratio_arr,
        )
    else:
        feats = features.make_features_batch_py(
            items_arr,
            item_sizes_arr,
            bin_items_flat_arr,
            bin_offsets_arr,
            bin_loads_arr,
            1.0,
            sizes_arr,
            n_total,
            size_ranks_arr,
            remaining_ratio_arr,
        )

    # Convert to list-of-lists so the function matches its annotation
    # `tuple[list[list[float]], list[int]]` and downstream callers that
    # expect an iterable of rows (list.extend compatible).
    if hasattr(feats, "tolist"):
        feats_list = feats.tolist()
    else:
        feats_list = [list(row) for row in feats]

    return feats_list, labels


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def _generate_examples_for_seed(
    seed: int, n_min: int, n_max: int, max_negatives: int, destroy_fraction: float
) -> tuple[list[list[float]], list[int]]:
    """Generate examples for a single instance deterministically from `seed`.

    This helper is used by the optional parallel dataset generation path.
    """
    rng = np.random.default_rng(int(seed))
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
    return x_fresh + x_repair, y_fresh + y_repair


def build_dataset(
    instances: int,
    n_min: int,
    n_max: int,
    max_negatives: int,
    seed: int,
    destroy_fraction: float = 0.20,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, DatasetSummary]:
    """Generate synthetic data and return X, y arrays ready for sklearn.

    Each synthetic instance contributes two batches of rows: one from a fresh
    BFD trace and one from a post-destruction repair trace (see module docstring
    for rationale). The two batches are pooled before splitting.

    If ``workers`` > 1, instances are generated in parallel using
    ``concurrent.futures.ProcessPoolExecutor``. Deterministic per-instance
    seeds are derived from the provided ``seed`` to preserve reproducibility.
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

    if workers and workers > 1:
        seeds = rng.integers(0, 2**31 - 1, size=instances).tolist()
        with ProcessPoolExecutor(max_workers=workers) as exe:
            map_iter = exe.map(
                _generate_examples_for_seed,
                seeds,
                repeat(n_min),
                repeat(n_max),
                repeat(max_negatives),
                repeat(destroy_fraction),
            )
            for x_part, y_part in tqdm(
                map_iter,
                total=instances,
                desc="Generating instances",
                unit="inst",
                dynamic_ncols=True,
            ):
                all_x.extend(x_part)
                all_y.extend(y_part)
    else:
        for _ in tqdm(
            range(instances),
            desc="Generating instances",
            unit="inst",
            dynamic_ncols=True,
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
        rows=int(X.shape[0]), cols=int(X.shape[1]), positive_rate=float(y.mean())
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
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for dataset generation (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="repair_model.pkl",
        help="Output path for the saved model bundle",
    )
    parser.add_argument(
        "--augment-with",
        type=str,
        default=None,
        metavar="PKL",
        help=(
            "Path to a .pkl produced by collect_alns_states.py. "
            "Its (X, y) rows are appended to the BFD dataset before training, "
            "reducing covariate shift between training and ALNS inference states."
        ),
    )
    parser.add_argument(
        "--prewarm-numba",
        action="store_true",
        help="Pre-warm numba JIT for batch feature function before dataset generation",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    # Optional numba pre-warm: compile the batch JIT once so the first
    # heavy call later does not pay the compile cost during data generation.
    if args.prewarm_numba:
        if getattr(features, "njit", None) is not None and hasattr(
            features, "make_features_batch_jit"
        ):
            try:
                print("Pre-warming numba batch feature JIT...")
                a_items = np.array([0], dtype=np.int64)
                a_item_sizes = np.array([0.1], dtype=np.float64)
                a_bin_items_flat = np.array([], dtype=np.int64)
                a_bin_offsets = np.array([0, 0], dtype=np.int64)
                a_bin_loads = np.array([0.0], dtype=np.float64)
                a_sizes = np.array([0.1], dtype=np.float64)
                a_size_ranks = np.array([0], dtype=np.int64)
                a_remaining = np.array([1.0], dtype=np.float64)
                # single-call to trigger compilation
                features.make_features_batch_jit(
                    a_items,
                    a_item_sizes,
                    a_bin_items_flat,
                    a_bin_offsets,
                    a_bin_loads,
                    1.0,
                    a_sizes,
                    1,
                    a_size_ranks,
                    a_remaining,
                )
            except Exception:
                # Don't fail the whole script if pre-warm fails; fall back
                # to on-demand compilation later.
                print("Numba pre-warm failed; continuing without pre-warm")

    print("Generating dataset...")
    X, y, summary = build_dataset(
        instances=args.instances,
        n_min=args.n_min,
        n_max=args.n_max,
        max_negatives=args.max_negatives,
        destroy_fraction=args.destroy_fraction,
        seed=args.seed,
        workers=args.workers,
    )

    if args.augment_with:
        aug_path = Path(args.augment_with)
        if not aug_path.exists():
            raise FileNotFoundError(f"Augmentation file not found: {aug_path}")
        with aug_path.open("rb") as f:
            aug = pickle.load(f)
        if aug.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Augmentation file feature_version={aug.get('feature_version')} "
                f"does not match current FEATURE_VERSION={FEATURE_VERSION}."
            )
        X_aug = aug["X"].astype(np.float32)
        y_aug = aug["y"].astype(np.int32)
        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, y_aug], axis=0)
        print(
            f"Augmented with {len(X_aug)} ALNS states from {aug_path} → total rows: {len(X)}"
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

    # GradientBoostingClassifier captures non-linear feature interactions
    # (e.g. item_size × remaining_capacity) that Logistic Regression cannot.
    # The solver uses predict_proba scores to rank candidate bins, so ranking
    # quality (AUC) matters more than hard-decision accuracy.
    #
    # GBC does not support class_weight natively; we use compute_sample_weight
    # to achieve the same "balanced" effect: each sample is weighted inversely
    # proportional to its class frequency.
    #
    # StandardScaler is kept in the Pipeline for consistency with the saved
    # bundle (the solver always applies the scaler before predict_proba).
    print(
        "Training model  (Pipeline · GradientBoostingClassifier · "
        "n_estimators=300 · max_depth=4 · lr=0.05)..."
    )
    t_train_start = time.perf_counter()
    sample_weights = compute_sample_weight("balanced", y_trainval)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(x_trainval, y_trainval, clf__sample_weight=sample_weights)
    t_train_elapsed = time.perf_counter() - t_train_start
    print(f"Training complete in {t_train_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Evaluation on held-out test set
    # ------------------------------------------------------------------

    test_proba = pipe.predict_proba(x_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    # C_ has shape (n_classes,) = (1,) for binary problems; use flat[0] to get
    n_estimators = pipe.named_steps["clf"].n_estimators_
    print(f"Test ROC-AUC: {test_auc:.4f}  (n_estimators={n_estimators})")

    # A well-trained GBM on this task should achieve AUC > 0.80; lower values
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
