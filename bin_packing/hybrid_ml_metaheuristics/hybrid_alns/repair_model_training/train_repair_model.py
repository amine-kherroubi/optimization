"""Train the repair model with enhanced metrics and optional tuning.

Improvements over the original baseline training script:
    1. Full metrics (F1, precision, recall, confusion matrix)
    2. K-fold cross-validation
    3. Learning curves for overfitting detection
    4. Feature importance analysis
    5. Optional hyperparameter grid search
    6. Support for real Falkenauer data
    7. Early stopping based on validation loss
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import sys

try:
    # Prefer package-style import when invoked from project root
    from repair_model_training import features
except Exception:  # pragma: no cover - fallback for script execution
    import features

# Expose feature metadata constants for downstream code
FEATURE_VERSION = features.FEATURE_VERSION
N_FEATURES = features.N_FEATURES

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ============================================================================
# Enhanced evaluation metrics
# ============================================================================


class ModelEvaluator:
    """Comprehensive model evaluation toolkit."""

    def __init__(self, X_train, y_train, X_test, y_test, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

    def evaluate_all(self) -> dict[str, Any]:
        """Compute all metrics and return as dictionary."""
        metrics = {}

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Main metrics
        metrics["accuracy"] = float((y_pred == self.y_test).mean())
        metrics["precision"] = float(precision_score(self.y_test, y_pred))
        metrics["recall"] = float(recall_score(self.y_test, y_pred))
        metrics["f1"] = float(f1_score(self.y_test, y_pred))
        metrics["roc_auc"] = float(roc_auc_score(self.y_test, y_proba))
        metrics["average_precision"] = float(
            average_precision_score(self.y_test, y_proba)
        )

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        # Specificity & sensitivity
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Threshold analysis
        fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        metrics["optimal_threshold"] = float(thresholds[optimal_idx])

        return metrics

    def print_report(self, metrics: dict[str, Any]) -> None:
        """Pretty-print all metrics."""
        print("\n" + "=" * 70)
        print("FULL MODEL EVALUATION")
        print("=" * 70)

        print("\nClassification metrics:")
        print(f"  Accuracy      : {metrics['accuracy']:.4f}")
        print(
            f"  Precision     : {metrics['precision']:.4f}  (true positives / predicted positives)"
        )
        print(
            f"  Recall        : {metrics['recall']:.4f}    (true positives / actual positives)"
        )
        print(f"  F1-Score      : {metrics['f1']:.4f}")
        print(
            f"  Specificity   : {metrics['specificity']:.4f}  (true negatives / actual negatives)"
        )
        print(f"  Sensitivity   : {metrics['sensitivity']:.4f}  (= recall)")

        print("\nRanking metrics (the ones that matter most here):")
        print(f"  ROC-AUC              : {metrics['roc_auc']:.4f}  <-- main metric")
        print(f"  Average Precision    : {metrics['average_precision']:.4f}")
        print(f"  Optimal Threshold    : {metrics['optimal_threshold']:.4f}")

        cm = metrics["confusion_matrix"]
        print("\nConfusion matrix:")
        print(f"  TP: {cm['true_positives']:6d}  |  FP: {cm['false_positives']:6d}")
        print(f"  FN: {cm['false_negatives']:6d}  |  TN: {cm['true_negatives']:6d}")

        print("\nInterpretation for bin packing:")
        print(
            "  High precision means fewer placement mistakes and more reliable positive predictions."
        )
        print(
            "  High recall means the model finds most of the best bins and misses fewer good choices."
        )
        print(
            "  ROC-AUC measures ranking quality, which is the key objective for ALNS repair."
        )
        print("=" * 70 + "\n")


def plot_learning_curves(
    model,
    X_train,
    y_train,
    output_path: str = "learning_curves.png",
) -> None:
    """Plot learning curves to detect overfitting."""
    print("Computing learning curves (this can take some time)...")

    res = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        verbose=0,
    )
    train_sizes, train_scores, val_scores = res[0], res[1], res[2]

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Train AUC", linewidth=2)
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )
    plt.plot(train_sizes, val_mean, "s-", label="Val AUC", linewidth=2)
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
    )
    plt.xlabel("Training set size")
    plt.ylabel("ROC-AUC")
    plt.title("Learning Curves (Overfitting detection)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Learning curves saved: {output_path}")


def plot_roc_curve(
    y_test,
    y_proba,
    output_path: str = "roc_curve.png",
) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"ROC curve saved: {output_path}")


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 11,
    output_path: str = "feature_importance.png",
) -> None:
    """Plot feature importance from GradientBoosting."""
    if not hasattr(model, "feature_importances_"):
        print("WARNING: The model does not expose feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top Features - Importance for Bin Selection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Feature importance saved: {output_path}")

    # Print top features
    print("\nTop 5 features by importance:")
    for i, idx in enumerate(indices[:5], 1):
        print(f"  {i}. {feature_names[idx]:20s} : {importances[idx]:.4f}")


# ---------------------------------------------------------------------------
# Dataset generation utilities (self-contained)
# ---------------------------------------------------------------------------


@dataclass
class DatasetSummary:
    rows: int
    cols: int
    positive_rate: float


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


def build_dataset(
    *,
    instances: int,
    n_min: int,
    n_max: int,
    max_negatives: int,
    seed: int = 0,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, DatasetSummary]:
    """Generate a synthetic training dataset (X, y) and a short summary.

    This function mirrors the labeling logic used elsewhere: for each
    synthetic instance we compute a simple FFD start solution and label
    feasible bin placements using the BFD oracle (minimum post-placement slack).
    """
    rng = np.random.default_rng(seed)
    all_X: list[list[float]] = []
    all_y: list[int] = []

    for _ in tqdm(range(instances), desc="Generating dataset", total=instances):
        sizes = _generate_instance(rng, n_min, n_max)
        sizes_list = [float(v) for v in sizes]
        n = len(sizes_list)
        capacity = 1.0
        mf = features.make_features

        # Size ranking (as used by the feature creator)
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

        # Label each item with BFD oracle and collect positives + sampled negatives
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

            # Positive example (BFD)
            all_X.append(_feat(best_bin))
            all_y.append(1)

            # Negatives (other feasible bins), subsampled
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
        rows=int(X_arr.shape[0]), cols=int(N_FEATURES), positive_rate=pos_rate
    )
    return X_arr, y_arr, summary


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train repair model with comprehensive metrics"
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
        help="Maximum negative samples per positive",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducibility"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for dataset generation",
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
        help="Path to a .pkl file produced by collect_alns_states.py for data augmentation",
    )
    parser.add_argument(
        "--no-learning-curves",
        action="store_true",
        help="Skip learning curves (saves time)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plot generation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable hyperparameter grid search (warning: very slow)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show training progress for GradientBoosting",
    )
    args = parser.parse_args()

    # =====================================================================
    # 1. Data generation
    # =====================================================================
    print("=" * 70)
    print("PHASE 1: DATASET GENERATION")
    print("=" * 70)
    print(f"Generating {args.instances} synthetic instances...")
    print(f"  Item size range: [{args.n_min}, {args.n_max}]")
    print(f"  Maximum negatives per positive: {args.max_negatives}")
    print(f"  Workers: {args.workers}")

    X, y, summary = build_dataset(
        instances=args.instances,
        n_min=args.n_min,
        n_max=args.n_max,
        max_negatives=args.max_negatives,
        seed=args.seed,
        workers=args.workers,
    )

    if args.augment_with:
        aug_path = Path(args.augment_with)
        if aug_path.exists():
            with aug_path.open("rb") as f:
                aug = pickle.load(f)
            X_aug = aug["X"].astype(np.float32)
            y_aug = aug["y"].astype(np.int32)
            X = np.concatenate([X, X_aug], axis=0)
            y = np.concatenate([y, y_aug], axis=0)
            print(f"Augmented with {len(X_aug)} ALNS states -> total: {len(X)} rows")

    print(f"Dataset generated: {summary.rows} rows x {summary.cols} features")
    print(
        f"   Positive rate: {summary.positive_rate:.4f} (expected min: {1.0/(1+args.max_negatives):.4f})"
    )
    print(f"   Class 0 (negatives): {(y==0).sum():,} samples")
    print(f"   Class 1 (positives):  {(y==1).sum():,} samples")

    # =====================================================================
    # 2. Train/Test split
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TRAIN/TEST SPLIT")
    print("=" * 70)

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"Training set:  {len(x_trainval):,} samples")
    print(f"Test set:      {len(x_test):,} samples")

    # =====================================================================
    # 3. Model training with optional hyperparameter tuning
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: MODEL TRAINING")
    print("=" * 70)

    # Improved hyperparameters compared with the original baseline.
    hyperparams = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "validation_fraction": 0.1,
        "n_iter_no_change": 50,
        "verbose": 1 if args.verbose else 0,
        "random_state": 42,
    }

    if args.grid_search:
        print("WARNING: Grid search enabled (very long, 1-2 hours)...")
        param_grid = {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [4, 5, 6],
            "clf__learning_rate": [0.01, 0.03, 0.05],
        }
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        subsample=0.75,
                        min_samples_split=20,
                        min_samples_leaf=10,
                        verbose=1 if args.verbose else 0,
                        random_state=42,
                    ),
                ),
            ]
        )
        sample_weights = compute_sample_weight("balanced", y_trainval)
        gs = GridSearchCV(
            pipe,
            param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1 if args.verbose else 0,
        )
        print("Starting GridSearchCV...")
        t_train_start = time.perf_counter()
        gs.fit(x_trainval, y_trainval, clf__sample_weight=sample_weights)
        t_train_elapsed = time.perf_counter() - t_train_start
        pipe = gs.best_estimator_
        print(f"Best params: {gs.best_params_}")
        print(f"Best CV score: {gs.best_score_:.4f}")
    else:
        # Build classifier explicitly to satisfy static type checkers
        clf = GradientBoostingClassifier(
            n_estimators=int(hyperparams["n_estimators"]),
            max_depth=int(hyperparams["max_depth"]),
            learning_rate=float(hyperparams["learning_rate"]),
            subsample=float(hyperparams["subsample"]),
            min_samples_split=int(hyperparams["min_samples_split"]),
            min_samples_leaf=int(hyperparams["min_samples_leaf"]),
            validation_fraction=float(hyperparams["validation_fraction"]),
            n_iter_no_change=int(hyperparams["n_iter_no_change"]),
            verbose=int(hyperparams["verbose"]),
            random_state=int(hyperparams["random_state"]),
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
        print("Training GradientBoosting with tuned hyperparameters:")
        print(f"  n_estimators      : {hyperparams['n_estimators']}")
        print(f"  max_depth         : {hyperparams['max_depth']}")
        print(f"  learning_rate     : {hyperparams['learning_rate']}")
        print(f"  subsample         : {hyperparams['subsample']}")
        print(f"  min_samples_split : {hyperparams['min_samples_split']}")
        print(f"  min_samples_leaf  : {hyperparams['min_samples_leaf']}")
        print(f"  early_stopping    : YES (validation_fraction=0.1)")

        t_train_start = time.perf_counter()
        sample_weights = compute_sample_weight("balanced", y_trainval)
        pipe.fit(x_trainval, y_trainval, clf__sample_weight=sample_weights)
        t_train_elapsed = time.perf_counter() - t_train_start

    print(f"Training finished in {t_train_elapsed:.1f}s")
    n_estimators = pipe.named_steps["clf"].n_estimators_
    print(f"   Trees used: {n_estimators}")

    # =====================================================================
    # 4. Cross-validation
    # =====================================================================
    print(f"\nCross-validation ({args.cv_folds}-fold)...")
    cv_scores = cross_val_score(
        pipe,
        x_trainval,
        y_trainval,
        cv=args.cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"   CV scores: {cv_scores}")
    print(f"   Mean CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # =====================================================================
    # 5. Evaluation on test set
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: TEST SET EVALUATION")
    print("=" * 70)

    evaluator = ModelEvaluator(x_trainval, y_trainval, x_test, y_test, pipe)
    metrics = evaluator.evaluate_all()
    evaluator.print_report(metrics)

    # Check AUC threshold
    if metrics["roc_auc"] < 0.70:
        print(
            "WARNING: ROC-AUC < 0.70. Consider increasing the number of instances or features."
        )

    # =====================================================================
    # 6. Visualization
    # =====================================================================
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("PHASE 5: VISUALIZATIONS")
        print("=" * 70)

        # ROC curve
        y_proba = pipe.predict_proba(x_test)[:, 1]
        plot_roc_curve(y_test, y_proba)

        # Feature importance
        feature_names = [
            "item_size",
            "item_size_sq",
            "size_rank",
            "remaining",
            "bin_load",
            "bin_rem",
            "slack_after",
            "bin_count",
            "bin_largest",
            "bin_smallest",
            "fill_ratio",
        ]
        plot_feature_importance(pipe.named_steps["clf"], feature_names)

        # Learning curves
        if not args.no_learning_curves:
            plot_learning_curves(pipe, x_trainval, y_trainval)

    # =====================================================================
    # 7. Save model bundle
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: MODEL SAVING")
    print("=" * 70)

    payload = {
        "model": pipe.named_steps["clf"],
        "scaler": pipe.named_steps["scaler"],
        "feature_version": FEATURE_VERSION,
        "n_features": N_FEATURES,
        "metrics": metrics,
        "cv_scores": cv_scores.tolist(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Model saved: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   Test F1-Score: {metrics['f1']:.4f}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Dataset       : {summary.rows:,} samples")
    print(f"Features      : {summary.cols}")
    print(f"Instances     : {args.instances}")
    print(f"CV folds      : {args.cv_folds}")
    print(f"CV mean AUC   : {cv_scores.mean():.4f}")
    print(f"Test AUC      : {metrics['roc_auc']:.4f}")
    print(f"Test F1       : {metrics['f1']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"Model path    : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
