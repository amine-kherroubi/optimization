from __future__ import annotations

import pickle
import random
import time
from dataclasses import dataclass, field
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
    average_precision_score,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve,
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from . import features
    from .generate_dataset import DatasetSummary
except ImportError:  # pragma: no cover - fallback for direct script execution
    import features  # type: ignore[no-redef]
    from generate_dataset import DatasetSummary  # type: ignore[no-redef]

FEATURE_VERSION = features.FEATURE_VERSION
N_FEATURES = features.N_FEATURES

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ============================================================================
# Config
# ============================================================================


@dataclass
class TrainRepairModelConfig:
    data: list[str] = field(default_factory=list)
    """One or more .pkl dataset paths to load and merge before training.
    Each file must have been produced by generate_dataset.py or
    collect_alns_states.py (keys: X, y, feature_version).
    """
    output: str = "repair_model.pkl"
    no_learning_curves: bool = False
    no_plots: bool = False
    cv_folds: int = 5
    grid_search: bool = False
    verbose: bool = False

    # --- Reproducibility -------------------------------------------------
    seed: int = 42
    """Single source of randomness for train/test split and model training."""

    # --- Quality gates ---------------------------------------------------
    min_roc_auc: float = 0.70
    """Warn (and optionally fail) if holdout ROC-AUC falls below this."""
    min_average_precision: float = 0.50
    """Warn if holdout Average Precision falls below this."""
    min_top1_accuracy: float = 0.70
    """Warn if grouped top-1 repair accuracy falls below this when groups exist."""

    # --- Covariate-shift mitigation --------------------------------------
    require_alns_states: bool = True
    """When True, training will raise if no ALNS-state dataset is detected
    in `data`.  Set to False only for the v1 baseline model."""


# ============================================================================
# Enhanced evaluation metrics
# ============================================================================


class ModelEvaluator:
    """Comprehensive model evaluation toolkit."""

    def __init__(self, X_train, y_train, X_test, y_test, model, groups_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.groups_test = groups_test

    @staticmethod
    def _ranking_metrics(y_true, y_score, groups) -> dict[str, float | int]:
        """Evaluate whether each placement decision's top-ranked bin is the oracle.

        The repair model is used as a ranker: for one displaced item it scores all
        feasible bins and the solver selects ``argmax(score)``.  Row-wise AUC is
        useful but not sufficient, so grouped top-1 metrics are the closest offline
        proxy for runtime repair quality.
        """
        if groups is None:
            return {}

        groups_arr = np.asarray(groups)
        y_arr = np.asarray(y_true)
        score_arr = np.asarray(y_score)

        top1_hits = 0
        oracle_groups = 0
        reciprocal_ranks: list[float] = []
        candidate_counts: list[int] = []

        for group in np.unique(groups_arr):
            idx = np.flatnonzero(groups_arr == group)
            if idx.size == 0:
                continue
            labels = y_arr[idx]
            positives = np.flatnonzero(labels == 1)
            if positives.size == 0:
                continue

            oracle_groups += 1
            scores = score_arr[idx]
            order = np.argsort(scores)[::-1]
            top1_hits += int(labels[order[0]] == 1)
            first_positive_rank = min(
                int(np.flatnonzero(order == pos)[0]) + 1 for pos in positives
            )
            reciprocal_ranks.append(1.0 / first_positive_rank)
            candidate_counts.append(int(idx.size))

        if oracle_groups == 0:
            return {}

        return {
            "ranking_groups": int(oracle_groups),
            "top1_accuracy": float(top1_hits / oracle_groups),
            "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
            "mean_candidates_per_group": float(np.mean(candidate_counts)),
        }

    def evaluate_all(self) -> dict[str, Any]:
        """Compute all metrics and return as dictionary."""
        metrics = {}

        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        metrics["accuracy"] = float((y_pred == self.y_test).mean())
        metrics["precision"] = float(
            precision_score(self.y_test, y_pred, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(self.y_test, y_pred, zero_division=0)
        )
        metrics["f1"] = float(f1_score(self.y_test, y_pred, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(self.y_test, y_proba))
        metrics["average_precision"] = float(
            average_precision_score(self.y_test, y_proba)
        )
        metrics.update(self._ranking_metrics(self.y_test, y_proba, self.groups_test))

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
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
            f"  Precision     : {metrics['precision']:.4f}"
            "  (true positives / predicted positives)"
        )
        print(
            f"  Recall        : {metrics['recall']:.4f}"
            "    (true positives / actual positives)"
        )
        print(f"  F1-Score      : {metrics['f1']:.4f}")
        print(
            f"  Specificity   : {metrics['specificity']:.4f}"
            "  (true negatives / actual negatives)"
        )
        print(f"  Sensitivity   : {metrics['sensitivity']:.4f}  (= recall)")

        print("\nRanking metrics (the ones that matter most here):")
        print(f"  ROC-AUC              : {metrics['roc_auc']:.4f}  <-- main metric")
        print(f"  Average Precision    : {metrics['average_precision']:.4f}")
        print(f"  Optimal Threshold    : {metrics['optimal_threshold']:.4f}")
        if "top1_accuracy" in metrics:
            print(
                f"  Top-1 repair accuracy: {metrics['top1_accuracy']:.4f}  "
                "(argmax bin equals BFD oracle)"
            )
            print(
                f"  Mean reciprocal rank : {metrics['mean_reciprocal_rank']:.4f}"
            )
            print(
                f"  Ranking groups       : {metrics['ranking_groups']:,}  "
                f"(avg candidates={metrics['mean_candidates_per_group']:.2f})"
            )

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


# ============================================================================
# Plots
# ============================================================================


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

    print("\nTop 5 features by importance:")
    for i, idx in enumerate(indices[:5], 1):
        print(f"  {i}. {feature_names[idx]:20s} : {importances[idx]:.4f}")


# ============================================================================
# Data loading and integrity checks
# ============================================================================


def _load_and_merge(
    data_paths: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, DatasetSummary, dict[str, Any]]:
    """Load and merge one or more dataset .pkl files.

    Each file must contain keys: X, y, feature_version.
    Raises ValueError if:
      - a file's feature_version does not match FEATURE_VERSION
      - X has wrong number of features (N_FEATURES)
      - y contains values other than 0/1
      - X or y contain NaN / inf values

    Returns X, y, groups, summary, source_info
    where source_info holds per-file counts and ALNS proportion.
    """
    if not data_paths:
        raise ValueError("At least one data path is required.")

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    all_files_have_groups = True
    next_group_id = 0
    source_rows: dict[str, int] = {}   # source_tag -> row count
    alns_rows = 0
    total_rows = 0

    print(f"\n{'─' * 60}")
    print("DATASET INTEGRITY CHECKS")
    print(f"{'─' * 60}")

    for path_str in data_paths:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        with p.open("rb") as f:
            bundle = pickle.load(f)

        # ── feature_version check ──────────────────────────────────────
        fv = bundle.get("feature_version")
        if fv != FEATURE_VERSION:
            raise ValueError(
                f"{p}: feature_version={fv} does not match "
                f"current FEATURE_VERSION={FEATURE_VERSION}. Regenerate first."
            )

        X_i = bundle["X"].astype(np.float32)
        y_i = bundle["y"].astype(np.int32)
        groups_raw = bundle.get("groups")

        # ── feature count check ────────────────────────────────────────
        if X_i.ndim != 2 or X_i.shape[1] != N_FEATURES:
            raise ValueError(
                f"{p}: expected {N_FEATURES} features per row, "
                f"got shape {X_i.shape}"
            )

        # ── NaN / inf check ────────────────────────────────────────────
        if not np.isfinite(X_i).all():
            raise ValueError(f"{p}: X contains NaN or inf values — regenerate.")
        if not np.isfinite(y_i.astype(np.float32)).all():
            raise ValueError(f"{p}: y contains NaN or inf values — regenerate.")

        # ── label value check ──────────────────────────────────────────
        unique_labels = np.unique(y_i)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"{p}: y has invalid label values {unique_labels}; "
                "expected only 0 and 1."
            )

        rows = X_i.shape[0]
        pos_rate = float(y_i.mean()) if rows > 0 else 0.0
        source_tag = str(bundle.get("source", p.stem))

        if groups_raw is None:
            all_files_have_groups = False
            groups_i = None
        else:
            groups_arr = np.asarray(groups_raw)
            if groups_arr.shape != y_i.shape:
                raise ValueError(
                    f"{p}: groups shape {groups_arr.shape} does not match "
                    f"y shape {y_i.shape}. Regenerate the dataset."
                )
            # Factorize per file and offset globally so ids cannot collide when
            # synthetic and ALNS-state datasets are merged.
            _, groups_i = np.unique(groups_arr, return_inverse=True)
            groups_i = groups_i.astype(np.int64) + next_group_id
            if groups_i.size:
                next_group_id = int(groups_i.max()) + 1

        print(f"  ✓ {p.name}")
        print(f"      source       : {source_tag}")
        print(f"      rows         : {rows:,}")
        print(f"      feature_ver  : {fv}")
        print(f"      pos_rate     : {pos_rate:.4f}")
        print(f"      groups       : {'yes' if groups_i is not None else 'no'}")
        print(f"      NaN/inf      : none")
        print(f"      labels       : {sorted(unique_labels.tolist())}")

        X_parts.append(X_i)
        y_parts.append(y_i)
        if groups_i is not None:
            group_parts.append(groups_i)

        source_rows[source_tag] = source_rows.get(source_tag, 0) + rows
        if "alns" in source_tag.lower():
            alns_rows += rows
        total_rows += rows

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0) if all_files_have_groups else None

    # Global NaN guard on merged array (catches numeric edge-cases in concat)
    if not np.isfinite(X).all():
        raise ValueError("Merged X contains NaN/inf — check individual files.")

    pos_rate = float(y.mean()) if y.size > 0 else 0.0
    alns_ratio = alns_rows / total_rows if total_rows > 0 else 0.0

    summary = DatasetSummary(
        rows=int(X.shape[0]),
        cols=int(N_FEATURES),
        positive_rate=pos_rate,
    )

    source_info: dict[str, Any] = {
        "per_source": source_rows,
        "alns_rows": alns_rows,
        "alns_ratio": alns_ratio,
        "has_groups": groups is not None,
        "group_count": int(np.unique(groups).size) if groups is not None else 0,
    }

    print(f"\n  Merged totals")
    print(f"    rows         : {total_rows:,}")
    print(f"    positive rate: {pos_rate:.4f}")
    print(f"    ALNS rows    : {alns_rows:,}  ({alns_ratio:.1%} of total)")
    if groups is not None:
        print(f"    groups       : {source_info['group_count']:,} placement decisions")
    else:
        print("    groups       : unavailable (falling back to row-wise split)")
    for src, cnt in source_rows.items():
        print(f"    [{src}] : {cnt:,} rows")
    print(f"{'─' * 60}\n")

    return X, y, groups, summary, source_info


def _group_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split rows while keeping all candidates from one decision together."""
    if groups is None:
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=seed, stratify=y
        )
        return x_train, x_test, y_train, y_test, None, None

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        groups[train_idx],
        groups[test_idx],
    )


def _cv_splitter(groups_train: np.ndarray | None, cv_folds: int):
    """Return CV splitter/list compatible with cross_val_score."""
    if groups_train is None:
        return cv_folds

    unique_groups = np.unique(groups_train)
    n_splits = min(cv_folds, unique_groups.size)
    if n_splits < 2:
        return 2
    return list(
        GroupKFold(n_splits=n_splits).split(
            np.zeros(groups_train.shape[0]), groups=groups_train
        )
    )


# ============================================================================
# Main training function
# ============================================================================


def train_repair_model(config: TrainRepairModelConfig) -> dict[str, Any]:
    """Load pre-generated data, train a GradientBoostingClassifier, and save.

    Parameters
    ----------
    config : TrainRepairModelConfig

    Returns
    -------
    dict — the saved model bundle (model, scaler, metrics, …)
    """
    args = config

    # =====================================================================
    # 0. Seed logging
    # =====================================================================
    print("=" * 70)
    print("REPRODUCIBILITY — SEEDS")
    print("=" * 70)
    print(f"  config.seed          : {args.seed}  (train/test split + model)")
    print(f"  Python random seed   : set to {args.seed}")
    print(f"  numpy random seed    : set to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    # =====================================================================
    # 1. Load data
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LOADING DATASET(S)")
    print("=" * 70)

    X, y, groups, summary, source_info = _load_and_merge(args.data)

    print(f"Merged dataset : {summary.rows:,} rows x {summary.cols} features")
    print(f"  Positive rate : {summary.positive_rate:.4f}")
    print(f"  Class 0 (neg) : {(y == 0).sum():,}")
    print(f"  Class 1 (pos) : {(y == 1).sum():,}")
    print(f"  ALNS ratio    : {source_info['alns_ratio']:.1%}")
    print(
        "  Grouped split : "
        + (
            f"YES ({source_info['group_count']:,} placement decisions)"
            if source_info["has_groups"]
            else "NO (legacy row-wise dataset)"
        )
    )

    # =====================================================================
    # 1b. Covariate-shift gate: require ALNS states for v2+ models
    # =====================================================================
    if args.require_alns_states and source_info["alns_rows"] == 0:
        raise ValueError(
            "require_alns_states=True but no ALNS-state dataset was found "
            "in the provided data files.  Run collect_alns_states.py first "
            "and include its output in config.data, or set "
            "require_alns_states=False for the v1 baseline."
        )

    # =====================================================================
    # 2. Train/Test split
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"  random_state = {args.seed}")

    x_trainval, x_test, y_trainval, y_test, groups_trainval, groups_test = (
        _group_train_test_split(X, y, groups, seed=args.seed)
    )
    print(f"Training set : {len(x_trainval):,} samples")
    print(f"Test set     : {len(x_test):,} samples")
    if groups_trainval is not None and groups_test is not None:
        print(f"Train groups : {np.unique(groups_trainval).size:,}")
        print(f"Test groups  : {np.unique(groups_test).size:,}")

    # =====================================================================
    # 3. Model training with optional hyperparameter tuning
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: MODEL TRAINING")
    print("=" * 70)

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
        "random_state": args.seed,
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
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        sample_weights = compute_sample_weight("balanced", y_trainval)
        gs = GridSearchCV(
            pipe,
            param_grid,
            cv=_cv_splitter(groups_trainval, min(3, args.cv_folds)),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1 if args.verbose else 0,
        )
        print("Starting GridSearchCV...")
        t_train_start = time.perf_counter()
        gs.fit(x_trainval, y_trainval, clf__sample_weight=sample_weights)
        t_train_elapsed = time.perf_counter() - t_train_start
        pipe = gs.best_estimator_
        print(f"Best params   : {gs.best_params_}")
        print(f"Best CV score : {gs.best_score_:.4f}")
    else:
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
        print(f"  random_state      : {hyperparams['random_state']}")
        print(f"  early_stopping    : YES (validation_fraction=0.1)")

        t_train_start = time.perf_counter()
        sample_weights = compute_sample_weight("balanced", y_trainval)
        pipe.fit(x_trainval, y_trainval, clf__sample_weight=sample_weights)
        t_train_elapsed = time.perf_counter() - t_train_start

    print(f"Training finished in {t_train_elapsed:.1f}s")
    n_estimators_used = pipe.named_steps["clf"].n_estimators_
    print(f"  Trees used: {n_estimators_used}")

    # =====================================================================
    # 4. Cross-validation
    # =====================================================================
    print(f"\nCross-validation ({args.cv_folds}-fold)...")
    cv = _cv_splitter(groups_trainval, args.cv_folds)
    cv_scores = cross_val_score(
        pipe,
        x_trainval,
        y_trainval,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"  CV scores        : {cv_scores}")
    print(f"  Mean CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # =====================================================================
    # 5. Evaluation on test set
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: TEST SET EVALUATION")
    print("=" * 70)

    evaluator = ModelEvaluator(
        x_trainval, y_trainval, x_test, y_test, pipe, groups_test=groups_test
    )
    metrics = evaluator.evaluate_all()
    evaluator.print_report(metrics)

    # =====================================================================
    # 5b. Quality gates
    # =====================================================================
    print("=" * 70)
    print("QUALITY GATES")
    print("=" * 70)
    gate_passed = True

    roc = metrics["roc_auc"]
    ap = metrics["average_precision"]
    top1 = metrics.get("top1_accuracy")

    if roc < args.min_roc_auc:
        print(
            f"  ⚠  ROC-AUC {roc:.4f} is below threshold {args.min_roc_auc:.4f}. "
            "Consider more training data or features."
        )
        gate_passed = False
    else:
        print(f"  ✓  ROC-AUC {roc:.4f} ≥ {args.min_roc_auc:.4f}")

    if ap < args.min_average_precision:
        print(
            f"  ⚠  Average Precision {ap:.4f} is below threshold "
            f"{args.min_average_precision:.4f}."
        )
        gate_passed = False
    else:
        print(f"  ✓  Average Precision {ap:.4f} ≥ {args.min_average_precision:.4f}")

    if top1 is not None:
        if top1 < args.min_top1_accuracy:
            print(
                f"  ⚠  Top-1 repair accuracy {top1:.4f} is below threshold "
                f"{args.min_top1_accuracy:.4f}."
            )
            gate_passed = False
        else:
            print(
                f"  ✓  Top-1 repair accuracy {top1:.4f} ≥ "
                f"{args.min_top1_accuracy:.4f}"
            )

    if gate_passed:
        print("  All quality gates passed.")
    else:
        print(
            "  One or more quality gates failed — model saved but review is recommended."
        )
    print("=" * 70)

    # =====================================================================
    # 6. Visualization
    # =====================================================================
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("PHASE 5: VISUALIZATIONS")
        print("=" * 70)

        y_proba = pipe.predict_proba(x_test)[:, 1]
        plot_roc_curve(y_test, y_proba)

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
        # Reproducibility provenance saved in the bundle
        "seed": args.seed,
        "dataset_summary": {
            "rows": summary.rows,
            "cols": summary.cols,
            "positive_rate": summary.positive_rate,
            "alns_rows": source_info["alns_rows"],
            "alns_ratio": source_info["alns_ratio"],
            "has_groups": source_info["has_groups"],
            "group_count": source_info["group_count"],
            "per_source": source_info["per_source"],
        },
        "quality_gates": {
            "min_roc_auc": args.min_roc_auc,
            "min_average_precision": args.min_average_precision,
            "min_top1_accuracy": args.min_top1_accuracy,
            "passed": gate_passed,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Model saved : {output_path}")
    print(f"  Size       : {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Test AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Test F1    : {metrics['f1']:.4f}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Dataset files : {args.data}")
    print(f"Total samples : {summary.rows:,}")
    print(f"ALNS ratio    : {source_info['alns_ratio']:.1%}")
    print(f"Features      : {summary.cols}")
    print(f"Seed          : {args.seed}")
    print(f"CV folds      : {args.cv_folds}")
    print(f"CV mean AUC   : {cv_scores.mean():.4f}")
    print(f"Test AUC      : {metrics['roc_auc']:.4f}")
    print(f"Test F1       : {metrics['f1']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"Quality gates : {'PASSED' if gate_passed else 'FAILED (see warnings)'}")
    print(f"Model path    : {output_path}")
    print("=" * 70)
    return payload


# ============================================================================
