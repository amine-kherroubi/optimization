"""Optuna hyperparameter tuning for the Hybrid ALNS solver — 1D Bin Packing.

Usage:
    from tune_with_optuna import HybridALNSTuner

    tuner = HybridALNSTuner(seed=42)
    tuner.run_isolation_steps()                 # optional warm-up
    tuner.run_stage1_coarse(n_trials=100)
    tuner.run_stage2_fine(n_trials=50)
    tuner.print_best_config()

Or run directly:
    python tune_with_optuna.py --help
    python tune_with_optuna.py --stage1-trials 30 --stage2-trials 20
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from tqdm import tqdm

from bin_packing_optimization.datasets.registry import DATASET_REGISTRY
from bin_packing_optimization.datasets.types import Instance
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver import (
    BinPackingSolver,
)
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.models import (
    load_repair_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETER DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# --- 9 Tunable metaheuristic parameters (the tuning target) ---
DEFAULT_PARAMS: dict[str, Any] = {
    "initial_temperature": 1.0 / math.log(2.0),  # ≈ 1.4427
    "alpha_cool": 0.9995,
    "k_min_frac": 0.05,
    "k_max_frac": 0.25,
    "bandit_alpha": 0.3,
    "warmup_calls": 300,
    "no_improve_frac": 0.05,
    "reheat_soft_mult": 0.35,
    "reheat_hard_mult": 0.20,
}

# --- Structural parameters (safety floors / design constants, held fixed) ---
STRUCTURAL_DEFAULTS: dict[str, Any] = {
    "min_no_improve_limit": 250,
    "temp_precision_floor": 1e-12,
    "reheat_check_interval_divisor": 4.0,
    "reheat_check_min_interval": 50,
    "hard_restart_min_limit": 100,
    "patience_shrink_factor": 2.0 / 3.0,
}

# --- Pipeline-level configuration (not passed to the solver) ---
PIPELINE_CONFIG: dict[str, Any] = {
    "quality_weight": 0.9,
    "time_weight": 0.1,
    "time_limit": 60,
}

# --- Isolation-step small bank size ---
# Isolation steps use a smaller bank (20 instances vs 20+40 for Optuna stages).
# This keeps the grid coarse-and-fast before the expensive Optuna stages.
ISOLATION_BANK_SIZE: int = 20

# --- Isolation-step grid definitions ---
# Each isolation step tests one group of params with uniform-random destruction
# (bandit disabled) to measure the group's effect without confounding.
# Grids are kept small (3 values per param) for speed — the goal is a rough
# directional signal, not precise tuning.
ISOLATION_GRIDS: dict[str, list[dict[str, Any]]] = {
    "step1_sa_thermal": [
        {
            "initial_temperature": t,
            "alpha_cool": a,
            "reheat_soft_mult": s,
            "reheat_hard_mult": h,
        }
        for t in [0.5, 1.44, 5.0]
        for a in [0.99, 0.999, 0.9999]
        for s in [0.10, 0.35, 0.70]
        for h in [0.05, 0.20, 0.50]
    ],  # 3×3×3×3 = 81 configs
    "step2_destruction": [
        {"k_min_frac": kmin, "k_max_frac": kmax, "no_improve_frac": ni}
        for kmin in [0.01, 0.05, 0.20]
        for kmax in [0.10, 0.25, 0.50]
        for ni in [0.01, 0.05, 0.20]
    ],  # 3×3×3 = 27 configs
    "step3_bandit": [
        {"bandit_alpha": a, "warmup_calls": w}
        for a in [0.01, 0.30, 2.00]
        for w in [0, 300, 1000]
    ],  # 3×3 = 9 configs
}

# --- Default instance bank composition (total = 100) ---
# Each entry: (dataset_key, proportional_weight)
BANK_COMPOSITION: list[tuple[str, int]] = [
    ("falkenauer-t", 5),
    ("falkenauer-u", 40),
    ("scholl-1", 10),
    ("scholl-2", 35),
    ("scholl-3", 10),
]
BANK_COMPOSITION_TOTAL = sum(w for _, w in BANK_COMPOSITION)

# --- Optuna Stage 1 search ranges (tightened) ---
STAGE1_RANGES: dict[str, dict[str, Any]] = {
    "initial_temperature": {"low": 0.5, "high": 5.0, "log": True},
    "alpha_cool": {"low": 0.998, "high": 0.9999},
    "reheat_soft_mult": {"low": 0.10, "high": 0.90},
    "reheat_hard_mult": {"low": 0.05, "high": 0.80},
    "k_min_frac": {"low": 0.01, "high": 0.20},
    "k_max_frac": {"low": 0.10, "high": 0.40},
    "no_improve_frac": {"low": 0.01, "high": 0.20},
    "bandit_alpha": {"low": 0.05, "high": 1.0, "log": True},
    "warmup_calls": {"low": 50, "high": 600, "type": "int"},
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TuningResult:
    """Metrics collected from evaluating a parameter configuration on the bank."""

    mean_relative_gap: float
    mean_normalized_time: float
    mean_time_s: float
    total_time_s: float
    per_dataset_relative_gap: dict[str, float]
    arm_pulls: list[int]
    n_instances: int


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _make_solver_kwargs(
    user_params: dict[str, Any], include_structural: bool = False
) -> dict[str, Any]:
    """Merge *user_params* on top of ``DEFAULT_PARAMS`` (and optionally structural)."""
    kwargs = dict(DEFAULT_PARAMS)
    kwargs.update(user_params)
    if include_structural:
        for k, v in STRUCTURAL_DEFAULTS.items():
            kwargs.setdefault(k, v)
    return kwargs


def _lower_bound(inst: Instance) -> int:
    """Continuous lower bound LB = ceil(sum(sizes) / bin_capacity)."""
    return int(math.ceil(sum(inst.sizes) / inst.bin_capacity))


def _params_diff(
    base: dict[str, Any], changed: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return ``{param: (base_val, changed_val)}`` for keys that differ."""
    return {
        k: (base[k], changed[k]) for k in changed if k in base and base[k] != changed[k]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PARALLEL WORKER (module-level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_one(
    inst_dict: dict[str, Any],
    model_name: str,
    seed: int,
    max_iter: int,
    time_limit: float,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one instance. Module-level so ProcessPoolExecutor can pickle it."""
    import math
    import time

    from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver import (
        BinPackingSolver,
    )
    from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.models import (
        load_repair_model,
    )

    model_bundle = load_repair_model(model_name)
    sizes = list(inst_dict["sizes"])
    solver = BinPackingSolver(sizes, inst_dict["bin_capacity"], seed=seed)

    t0 = time.perf_counter()
    solver.solve(model_bundle=model_bundle, max_iterations=max_iter, **kwargs)
    elapsed = time.perf_counter() - t0

    sol = solver.get_solution()
    lb = int(math.ceil(sum(sizes) / inst_dict["bin_capacity"]))
    relative_gap = (sol.total_bins_used - lb) / max(lb, 1)

    arm_pulls = [0, 0, 0]
    if hasattr(solver, "_bandit"):
        for arm_idx in range(3):
            arm_pulls[arm_idx] = solver._bandit.arm_counts[arm_idx]

    normalized_time = elapsed / max(time_limit, 1e-9)

    return {
        "dataset_key": inst_dict["dataset_key"],
        "relative_gap": relative_gap,
        "normalized_time": normalized_time,
        "elapsed": elapsed,
        "arm_pulls": arm_pulls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TUNER CLASS
# ─────────────────────────────────────────────────────────────────────────────


class HybridALNSTuner:
    """Optuna-based hyperparameter tuner for the Hybrid ALNS 1D-BPP solver.

    Pipeline
    --------
    1. **Baseline** — evaluate default config →  :file:`baseline_defaults.json`
    2. **Isolation steps** (optional) — grid-search each param group independently
       with uniform-random bandit →  :file:`isolation_step*.json`
    3. **Stage 1 (coarse)** — wide-range Bayesian search (2 000 iter/inst)
       →  :file:`alns_stage1_coarse_study.json`
    4. **Stage 2 (fine)** — narrow-range search anchored on Stage-1 best
       (5 000 iter/inst) →  :file:`alns_stage2_fine_study.json`
    5. **Final** — evaluate merged best →  :file:`best_config_evaluation.json`
    """

    def __init__(
        self,
        counts_per_key: dict[str, int] | None = None,
        seed: int = 42,
        model_name: str = "repair_model_v2.pkl",
        output_dir: str = "tuning_results",
        bank_size: int = 20,
        stage2_bank_size: int = 40,
        quality_weight: float = 0.9,
        time_weight: float = 0.1,
        time_limit: float = 60,
    ):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.model_bundle = load_repair_model(model_name)

        self.quality_weight = quality_weight
        self.time_weight = time_weight
        self.time_limit = time_limit

        # Build instance bank proportionally from BANK_COMPOSITION
        if counts_per_key is None:
            scale = bank_size / BANK_COMPOSITION_TOTAL
            counts_per_key = {
                key: max(1, round(w * scale)) for key, w in BANK_COMPOSITION
            }
        self.counts_per_key = counts_per_key

        # Stage 2 uses a separate (larger) composition
        scale2 = stage2_bank_size / BANK_COMPOSITION_TOTAL
        self.counts_per_key_stage2: dict[str, int] = {
            key: max(1, round(w * scale2)) for key, w in BANK_COMPOSITION
        }

        self.tuning_bank: list[Instance] = self._build_bank()
        print(
            f"\nTuning bank: {len(self.tuning_bank)} instances  "
            f"(isolation + Stage 1)"
        )
        self._log_bank_summary()

        # Smaller bank for fast isolation steps — stratified across datasets
        self.isolation_bank: list[Instance] = self._build_isolation_bank()
        print(
            f"  Isolation bank: {len(self.isolation_bank)} instances "
            f"(stratified subset, for fast grid search)"
        )

        # Larger bank for Stage 2 (more robust evaluation)
        self.stage2_bank: list[Instance] = self._build_bank(self.counts_per_key_stage2)
        print(f"  Stage 2 bank: {len(self.stage2_bank)} instances")

    # ── Instance bank ──────────────────────────────────────────────────────

    def _build_bank(
        self,
        counts_per_key: dict[str, int] | None = None,
    ) -> list[Instance]:
        """Build a size-stratified instance bank from the dataset registry.

        Uses strided sampling (with jitter) so each dataset contributes instances
        that span its full item-count range.
        """
        if counts_per_key is None:
            counts_per_key = self.counts_per_key
        bank: list[Instance] = []
        rng = np.random.default_rng(self.seed)

        for key, count in counts_per_key.items():
            if count <= 0:
                continue
            config = DATASET_REGISTRY[key]
            files = sorted(config.directory.glob(config.glob))

            if count > len(files):
                print(
                    f"  Warning: {key} requested {count} but only {len(files)} "
                    f"available. Using all {len(files)}."
                )
                count = len(files)

            step = max(1, len(files) // count)
            indices = list(range(0, len(files), step))[:count]
            for i in indices:
                j = int(rng.integers(i, min(i + step, len(files))))
                inst = config.parser(files[j], config.key)
                bank.append(inst)

        rng.shuffle(bank)
        return bank

    def _build_isolation_bank(self) -> list[Instance]:
        """Build a small stratified bank for fast isolation grid search.

        Picks ``ISOLATION_BANK_SIZE // n_datasets`` instances per dataset
        so all datasets are represented while keeping the total small.
        """
        bank: list[Instance] = []
        rng = np.random.default_rng(self.seed)
        n_per = max(1, ISOLATION_BANK_SIZE // max(len(self.counts_per_key), 1))
        for key in sorted(self.counts_per_key):
            pool = [inst for inst in self.tuning_bank if inst.dataset_key == key]
            if not pool:
                continue
            picked = rng.choice(
                pool, size=min(n_per, len(pool)), replace=False
            ).tolist()
            bank.extend(picked)
        # Shuffle so training order doesn't bias per-trial timing
        rng.shuffle(bank)
        return bank

    def _log_bank_summary(self) -> None:
        """Print the bank composition per dataset."""
        by_key: dict[str, list[int]] = {}
        for inst in self.tuning_bank:
            by_key.setdefault(inst.dataset_key, []).append(inst.num_items)
        print(f"  Composition ({len(self.tuning_bank)} instances):")
        for key, sizes in sorted(by_key.items()):
            n = len(sizes)
            min_s, max_s = min(sizes), max(sizes)
            print(f"    {key}: {n} instances (items {min_s}–{max_s})")

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate_config(
        self,
        solver_kwargs: dict[str, Any],
        max_iter: int = 2000,
        time_limit: float | None = None,
        force_uniform_random: bool = False,
        bank: list[Instance] | None = None,
        n_jobs: int = 1,
    ) -> TuningResult:
        """Run the solver on *bank* instances and return aggregate metrics.

        Parameters
        ----------
        solver_kwargs:
            Parameter overrides (e.g. ``{"alpha_cool": 0.9993}``).
            Missing params are filled from ``DEFAULT_PARAMS``.
        max_iter:
            ALNS iterations per instance.
        time_limit:
            Optional per-instance wall-clock limit in seconds.
        force_uniform_random:
            If True, the bandit is bypassed and destroy operators are chosen
            uniformly at random.
        bank:
            Instances to evaluate on. Defaults to ``self.tuning_bank``.
        n_jobs:
            Number of parallel workers for instance evaluation.
            -1 = use all CPUs. 1 = sequential (default).
        """
        if bank is None:
            bank = self.tuning_bank

        if n_jobs == -1 or n_jobs is None:
            n_jobs = os.cpu_count() or 1
        n_jobs = max(1, int(n_jobs))

        kwargs = _make_solver_kwargs(solver_kwargs, include_structural=True)
        kwargs["force_uniform_random"] = force_uniform_random

        # Prepare serializable instance dicts for the worker
        inst_dicts = [
            {
                "sizes": list(i.sizes),
                "bin_capacity": i.bin_capacity,
                "dataset_key": i.dataset_key,
            }
            for i in bank
        ]

        if n_jobs == 1:
            results = [
                _evaluate_one(
                    d, self.model_name, self.seed, max_iter, self.time_limit, kwargs
                )
                for d in tqdm(inst_dicts, desc="  Solving", leave=False)
            ]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = [
                    ex.submit(
                        _evaluate_one,
                        d,
                        self.model_name,
                        self.seed,
                        max_iter,
                        self.time_limit,
                        kwargs,
                    )
                    for d in inst_dicts
                ]
                results = [
                    f.result()
                    for f in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="  Solving",
                        leave=False,
                    )
                ]

        total_relative_gap = 0.0
        total_normalized_time = 0.0
        total_raw_time = 0.0
        per_ds_relative_gap: dict[str, list[float]] = {}
        arm_pulls = [0, 0, 0]

        for r in results:
            total_relative_gap += r["relative_gap"]
            total_normalized_time += r["normalized_time"]
            total_raw_time += r["elapsed"]
            per_ds_relative_gap.setdefault(r["dataset_key"], []).append(
                r["relative_gap"]
            )
            for arm_idx in range(3):
                arm_pulls[arm_idx] += r["arm_pulls"][arm_idx]

        n = len(results)
        return TuningResult(
            mean_relative_gap=total_relative_gap / max(n, 1),
            mean_normalized_time=total_normalized_time / max(n, 1),
            mean_time_s=total_raw_time / max(n, 1),
            total_time_s=total_raw_time,
            per_dataset_relative_gap={
                k: float(np.mean(v)) for k, v in per_ds_relative_gap.items()
            },
            arm_pulls=arm_pulls,
            n_instances=n,
        )

    # ── Logging ────────────────────────────────────────────────────────────

    def save_log(
        self,
        filename: str,
        run_type: str,
        params: dict[str, Any],
        result: TuningResult,
        search_ranges: dict[str, dict[str, Any]] | None = None,
        baseline_params: dict[str, Any] | None = None,
    ) -> None:
        """Save a structured JSON result.

        Parameters
        ----------
        filename:
            Output file name (e.g. ``"baseline_defaults.json"``).
        run_type:
            A label such as ``"baseline"``, ``"isolation_step1"``, etc.
        params:
            The parameter dictionary that was evaluated.
        result:
            The metrics returned by :meth:`evaluate_config`.
        search_ranges:
            If provided, records the search space for reproducibility.
        baseline_params:
            If provided, records what changed relative to this reference
            (usually ``DEFAULT_PARAMS``).
        """
        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "run_type": run_type,
            "seed": self.seed,
            "bank_size": len(self.tuning_bank),
            "parameters": dict(params),
            "structural_defaults": dict(STRUCTURAL_DEFAULTS),
            "pipeline_config": {
                "quality_weight": self.quality_weight,
                "time_weight": self.time_weight,
                "time_limit": self.time_limit,
            },
            "metrics": {
                "mean_relative_gap": result.mean_relative_gap,
                "mean_normalized_time": result.mean_normalized_time,
                "mean_time_s": result.mean_time_s,
                "total_time_s": result.total_time_s,
                "per_dataset_relative_gap": result.per_dataset_relative_gap,
                "arm_pulls": result.arm_pulls,
                "n_instances": result.n_instances,
            },
        }
        if search_ranges is not None:
            payload["search_ranges"] = search_ranges
        if baseline_params is not None:
            payload["changes_from_baseline"] = _params_diff(baseline_params, params)

        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  -> {path}")

    # ── Baseline ───────────────────────────────────────────────────────────

    def evaluate_defaults(self, max_iter: int = 2000, n_jobs: int = 1) -> TuningResult:
        """Evaluate the default config as a baseline reference."""
        print("\n" + "=" * 70)
        print("  BASELINE  —  default parameters  ".center(70, " "))
        print("=" * 70)
        result = self.evaluate_config({}, max_iter=max_iter, n_jobs=n_jobs)
        self.save_log(
            "baseline_defaults.json",
            "baseline",
            DEFAULT_PARAMS,
            result,
            baseline_params=DEFAULT_PARAMS,
        )
        print(f"\n  mean_relative_gap  = {result.mean_relative_gap:.4f}")
        print(
            f"  mean_normalized_time = {result.mean_normalized_time:.4f}  ({result.mean_time_s:.3f} s / {self.time_limit} s)"
        )
        return result

    # ── Isolation steps ────────────────────────────────────────────────────

    def run_isolation_steps(
        self,
        max_iter: int = 2000,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Grid-search each parameter group independently (uniform random bandit).

        Isolation steps measure the effect of one group of params while the
        bandit is forced to uniform random (so it does not confound the result).
        The best values from each step are merged into a *pre-tuned* anchor.

        Returns
        -------
        dict
            Merged best config across all isolation steps.
        """
        bank = self.isolation_bank
        print(f"\n{'=' * 70}")
        print("  ISOLATION STEPS  —  group-wise grid search  ".center(70, " "))
        print(
            f"  Bank: {len(bank)} instances  (subset of main tuning bank)".center(
                70, " "
            )
        )
        print(f"{'=' * 70}")

        best_merged: dict[str, Any] = {}
        all_step_logs: dict[str, Any] = {}

        for step_key, grid in ISOLATION_GRIDS.items():
            label = step_key.replace("step", "Step ").replace("_", " ").title()
            n_cfg = len(grid)
            est_h = n_cfg * len(bank) * 5 / 3600
            print(
                f"\n  ── {label}  ({n_cfg} configs × {len(bank)} instances ~ {est_h:.1f}h) ──"
            )

            if step_key == "step3_bandit":
                step_uniform = False  # bandit params need the bandit ON
            else:
                step_uniform = True  # disable bandit confounding

            best_gap = float("inf")
            best_params = {}

            for trial_idx, candidate in enumerate(
                tqdm(grid, desc=f"  {label}", leave=False)
            ):
                result = self.evaluate_config(
                    candidate,
                    max_iter=max_iter,
                    force_uniform_random=step_uniform,
                    bank=bank,
                    n_jobs=n_jobs,
                )
                if result.mean_relative_gap < best_gap:
                    best_gap = result.mean_relative_gap
                    best_params = dict(candidate)

                n_save = max(1, len(grid) // 3)
                if trial_idx % n_save == 0 or trial_idx == len(grid) - 1:
                    tqdm.write(
                        f"    gap={result.mean_relative_gap:.4f}  "
                        f"best={best_gap:.4f}  |  "
                        + ", ".join(f"{k}={v}" for k, v in candidate.items())
                    )

            # Log best of this step
            step_result = self.evaluate_config(
                best_params,
                max_iter=max_iter,
                force_uniform_random=step_uniform,
                bank=bank,
                n_jobs=n_jobs,
            )
            self.save_log(
                f"isolation_{step_key}.json",
                f"isolation_{step_key}",
                best_params,
                step_result,
                search_ranges={"grid": grid, "bank_size": len(bank)},
                baseline_params=DEFAULT_PARAMS,
            )
            print(f"  >> Best {label}: gap={best_gap:.4f}")
            for k, v in best_params.items():
                print(f"      {k:25s} = {v}")

            best_merged.update(best_params)
            all_step_logs[step_key] = {
                "best_params": best_params,
                "best_gap": best_gap,
            }

        # Log the merged config
        merged = {**DEFAULT_PARAMS, **best_merged}
        merged_result = self.evaluate_config(
            best_merged,
            max_iter=max_iter,
            force_uniform_random=False,
            bank=bank,
            n_jobs=n_jobs,
        )
        self.save_log(
            "isolation_merged.json",
            "isolation_merged",
            merged,
            merged_result,
            baseline_params=DEFAULT_PARAMS,
        )
        print(
            f"\n  >> Merged best from isolation: gap={merged_result.mean_relative_gap:.4f}"
        )
        print(f"\n{'=' * 70}")

        return merged

    # ── Stage 1: Coarse ───────────────────────────────────────────────────

    def run_stage1_coarse(
        self,
        n_trials: int = 100,
        max_iter: int = 2000,
        study_name: str = "alns_stage1_coarse",
        anchor: dict[str, Any] | None = None,
        n_jobs: int = 1,
    ) -> optuna.Study:
        """Stage 1: wide-range Bayesian search over all 9 parameters.

        If *anchor* (e.g. from isolation steps) is provided, it is enqueued as
        trial 0 alongside the default config.
        """
        print(f"\n{'=' * 70}")
        print("  STAGE 1  —  coarse Bayesian search  ".center(70, " "))
        print(f"  {n_trials} trials, {max_iter} iter/inst".center(70, " "))
        print(f"{'=' * 70}")

        # Print search ranges for transparency
        print("\n  Search ranges passed to Optuna:")
        for k, spec in STAGE1_RANGES.items():
            lo = spec["low"]
            hi = spec["high"]
            typ = spec.get("type", "float")
            log_ = spec.get("log", False)
            log_str = " [log]" if log_ else ""
            print(f"    {k:25s}  {typ:5s}  [{lo}, {hi}]{log_str}")

        # ── Objective ──────────────────────────────────────────────────────
        _n_jobs = n_jobs  # capture for closure
        _qw = self.quality_weight
        _tw = self.time_weight

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for k, spec in STAGE1_RANGES.items():
                lo, hi = spec["low"], spec["high"]
                log_ = spec.get("log", False)
                if spec.get("type") == "int":
                    params[k] = trial.suggest_int(k, int(lo), int(hi))
                else:
                    params[k] = trial.suggest_float(k, lo, hi, log=log_)
            result = self.evaluate_config(params, max_iter=max_iter, n_jobs=_n_jobs)
            composite = (
                _qw * result.mean_relative_gap + _tw * result.mean_normalized_time
            )
            trial.set_user_attr(
                "metrics",
                {
                    "mean_relative_gap": result.mean_relative_gap,
                    "mean_normalized_time": result.mean_normalized_time,
                    "mean_time_s": result.mean_time_s,
                    "per_dataset_relative_gap": result.per_dataset_relative_gap,
                    "arm_pulls": result.arm_pulls,
                },
            )
            return composite

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=None,
        )

        # Enqueue defaults plus optional anchor as early trials
        study.enqueue_trial(DEFAULT_PARAMS)
        if anchor is not None:
            study.enqueue_trial(anchor)

        study.optimize(objective, n_trials=n_trials)
        self._report_study(study, "Stage 1 (coarse)")
        self._save_study(study, study_name, search_ranges=STAGE1_RANGES)
        return study

    # ── Stage 2: Fine ──────────────────────────────────────────────────────

    def run_stage2_fine(
        self,
        coarse_study: optuna.Study | None = None,
        n_trials: int = 50,
        max_iter: int = 5000,
        study_name: str = "alns_stage2_fine",
        bank: list[Instance] | None = None,
        n_jobs: int = 1,
    ) -> optuna.Study:
        """Stage 2: narrow-range search anchored around the Stage-1 best.

        Windows are ±15–25% of Stage-1 best (or defaults if no study given).
        """
        anchor = (
            {**DEFAULT_PARAMS, **coarse_study.best_params}
            if coarse_study is not None
            else dict(DEFAULT_PARAMS)
        )

        # Build the narrow Stage-2 ranges (for logging)
        def _window(
            name: str, center: float, rel: float = 0.15, abs_: float = 0.0
        ) -> tuple[float, float]:
            if abs_ > 0:
                low = center - abs_
            else:
                low = center * (1.0 - rel)
            high = center + (abs_ if abs_ > 0 else center * rel)
            return max(1e-6, low), high

        STAGE2_RANGES = {
            "initial_temperature": {
                "low": max(1e-6, anchor["initial_temperature"] * 0.85),
                "high": anchor["initial_temperature"] * 1.15,
                "log": True,
            },
            "alpha_cool": {
                "low": anchor["alpha_cool"] - 0.0005,
                "high": anchor["alpha_cool"] + 0.0005,
            },
            "reheat_soft_mult": {
                "low": anchor["reheat_soft_mult"] - 0.10,
                "high": anchor["reheat_soft_mult"] + 0.10,
            },
            "reheat_hard_mult": {
                "low": anchor["reheat_hard_mult"] - 0.08,
                "high": anchor["reheat_hard_mult"] + 0.08,
            },
            "k_min_frac": {
                "low": max(1e-6, anchor["k_min_frac"] * 0.75),
                "high": anchor["k_min_frac"] * 1.25,
            },
            "k_max_frac": {
                "low": max(1e-6, anchor["k_max_frac"] * 0.75),
                "high": anchor["k_max_frac"] * 1.25,
            },
            "no_improve_frac": {
                "low": max(1e-6, anchor["no_improve_frac"] * 0.75),
                "high": anchor["no_improve_frac"] * 1.25,
            },
            "bandit_alpha": {
                "low": max(1e-6, anchor["bandit_alpha"] * 0.80),
                "high": anchor["bandit_alpha"] * 1.20,
                "log": True,
            },
            "warmup_calls": {
                "low": max(0, anchor["warmup_calls"] - 150),
                "high": anchor["warmup_calls"] + 150,
                "type": "int",
            },
        }

        print(f"\n{'=' * 70}")
        print("  STAGE 2  —  fine Bayesian search  ".center(70, " "))
        print(f"  {n_trials} trials, {max_iter} iter/inst".center(70, " "))
        print(f"  Anchored around Stage-1 best: {json.dumps(anchor)}")
        print(f"{'=' * 70}")

        print("\n  Narrow search ranges passed to Optuna:")
        for k, spec in sorted(STAGE2_RANGES.items()):
            lo = spec["low"]
            hi = spec["high"]
            log_ = spec.get("log", False)
            log_str = " [log]" if log_ else ""
            print(f"    {k:25s}  [{lo:.6g}, {hi:.6g}]{log_str}")

        # ── Objective ──────────────────────────────────────────────────────
        _bank = bank or self.stage2_bank
        _n_jobs = n_jobs  # capture for closure
        _qw = self.quality_weight
        _tw = self.time_weight

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for k, spec in STAGE2_RANGES.items():
                lo, hi = spec["low"], spec["high"]
                log_ = spec.get("log", False)
                if spec.get("type") == "int":
                    params[k] = trial.suggest_int(k, int(lo), int(hi))
                else:
                    params[k] = trial.suggest_float(k, lo, hi, log=log_)
            result = self.evaluate_config(
                params, max_iter=max_iter, bank=_bank, n_jobs=_n_jobs
            )
            composite = (
                _qw * result.mean_relative_gap + _tw * result.mean_normalized_time
            )
            trial.set_user_attr(
                "metrics",
                {
                    "mean_relative_gap": result.mean_relative_gap,
                    "mean_normalized_time": result.mean_normalized_time,
                    "mean_time_s": result.mean_time_s,
                    "per_dataset_relative_gap": result.per_dataset_relative_gap,
                    "arm_pulls": result.arm_pulls,
                },
            )
            return composite

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=None,
        )
        study.enqueue_trial(anchor)

        study.optimize(objective, n_trials=n_trials)
        self._report_study(study, "Stage 2 (fine)")
        self._save_study(study, study_name, search_ranges=STAGE2_RANGES)
        return study

    # ── Reporting helpers ──────────────────────────────────────────────────

    def _report_study(self, study: optuna.Study, label: str) -> None:
        print(f"\n{'=' * 70}")
        print(f"  {label}  —  best objective: {study.best_value:.4f}")
        print(f"{'=' * 70}")
        for k, v in study.best_params.items():
            print(f"    {k:25s} = {v}")

    def _save_study(
        self,
        study: optuna.Study,
        name: str,
        search_ranges: dict[str, Any] | None = None,
    ) -> None:
        trials_data = []
        for t in study.trials:
            if t.value is None:
                continue
            trials_data.append(
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "metrics": t.user_attrs.get("metrics"),
                }
            )
        payload: dict[str, Any] = {
            "study_name": name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(trials_data),
            "trials": trials_data,
        }
        if search_ranges is not None:
            payload["search_ranges"] = search_ranges
        path = self.output_dir / f"{name}_study.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  -> {path}")

    def print_best_config(
        self,
        stage1: optuna.Study | None = None,
        stage2: optuna.Study | None = None,
        isolation_best: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Print a clean summary showing the evolution of every parameter."""
        src = stage2 if stage2 is not None else stage1

        # Gather available stages
        stages = {"default (initial)": dict(DEFAULT_PARAMS)}
        if isolation_best is not None:
            stages["after isolation"] = {**DEFAULT_PARAMS, **isolation_best}
        if stage1 is not None:
            stages["after Stage 1 (coarse)"] = {**DEFAULT_PARAMS, **stage1.best_params}
        if src is not None:
            stages["after Stage 2 (fine, final)"] = {
                **DEFAULT_PARAMS,
                **src.best_params,
            }

        all_params = sorted(set(k for s in stages.values() for k in s))

        print(f"\n{'=' * 110}")
        header = "PARAMETER EVOLUTION  —  every stage tracked  ".center(110)
        print(f"  {header}")
        print(f"{'=' * 110}")

        # Column headers
        cols = ["Parameter"] + list(stages.keys()) + ["Search range (Stage 1)"]
        col_w = max(len(c) for c in cols) + 2
        fmt = f"{{:<{28}}}" + "".join(f"{{:<{24}}}" for _ in stages) + "{{}}"
        print(fmt.format(*cols))

        print("-" * 110)

        for p in all_params:
            if p in STRUCTURAL_DEFAULTS:
                continue  # structural params are always at default
            vals = []
            for label, cfg in stages.items():
                v = cfg.get(p, "—")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            rng = STAGE1_RANGES.get(p, {})
            if rng.get("type") == "int":
                lo, hi = int(rng["low"]), int(rng["high"])
                range_str = f"[{lo}, {hi}]"
            else:
                lo, hi = rng.get("low", "?"), rng.get("high", "?")
                log_ = "log" if rng.get("log") else ""
                range_str = f"[{lo}, {hi}] {log_}".strip()
            print(
                f"  {p:25s}  "
                + "  ".join(f"{v:>22s}" for v in vals)
                + f"    {range_str}"
            )

        print("=" * 110)

        # Final best merged
        if src is not None:
            best = {**DEFAULT_PARAMS, **STRUCTURAL_DEFAULTS, **PIPELINE_CONFIG, **src.best_params}
            print(f"\n  Best configuration found:")
            print(f"  {'=' * 45}")
            for k, v in best.items():
                print(f"    {k:30s} = {v}")
            return best

        return dict(DEFAULT_PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune Hybrid ALNS hyperparameters with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Balanced run (~2.5 h on 8 cores):\n"
            "  python tune_with_optuna.py --n-jobs 8\n\n"
            "  # Quick smoke test (~30 min on 8 cores):\n"
            "  python tune_with_optuna.py --quick --n-jobs 8\n\n"
            "  # Heavy run (~5 h on 8 cores):\n"
            "  python tune_with_optuna.py --bank-size 40 --stage2-bank-size 80 "
            "--stage1-trials 60 --stage2-trials 40 --n-jobs 8\n\n"
            "  # Baseline only:\n"
            "  python tune_with_optuna.py --baseline-only\n"
        ),
    )
    parser.add_argument(
        "--stage1-trials",
        type=int,
        default=30,
        help="Stage 1 (coarse) trials (default: 30)",
    )
    parser.add_argument(
        "--stage2-trials",
        type=int,
        default=20,
        help="Stage 2 (fine) trials (default: 20)",
    )
    parser.add_argument(
        "--iter-stage1",
        type=int,
        default=2000,
        help="ALNS iterations per instance in Stage 1 (default: 2000)",
    )
    parser.add_argument(
        "--iter-stage2",
        type=int,
        default=3000,
        help="ALNS iterations per instance in Stage 2 (default: 3000)",
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=20,
        help="Instance bank size for isolation + Stage 1, proportionally sampled (default: 20)",
    )
    parser.add_argument(
        "--stage2-bank-size",
        type=int,
        default=40,
        help="Instance bank size for Stage 2 (default: 40)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers. -1 = all CPUs (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_results",
        help="Output directory (default: tuning_results)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only evaluate defaults, skip tuning",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Skip isolation grid-search steps; go directly to Optuna",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Minimal: 5 trials S1, 3 trials S2, no isolation, bank=10, stage2-bank=20",
    )

    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.9,
        help="Quality coefficient in composite objective (default: 0.9)",
    )
    parser.add_argument(
        "--time-weight",
        type=float,
        default=0.1,
        help="Time coefficient in composite objective (default: 0.1)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60,
        help="Time normaliser in seconds for elapsed / time_limit (default: 60)",
    )

    args = parser.parse_args()

    # Apply --quick override
    if args.quick:
        args.stage1_trials = 5
        args.stage2_trials = 3
        args.no_isolation = True
        args.iter_stage1 = 500
        args.iter_stage2 = 1000
        args.bank_size = 10
        args.stage2_bank_size = 20

    # Build tuner with specified bank sizes
    counts_per_key = None
    scale = args.bank_size / BANK_COMPOSITION_TOTAL
    counts_per_key = {key: max(1, round(w * scale)) for key, w in BANK_COMPOSITION}

    tuner = HybridALNSTuner(
        seed=args.seed,
        output_dir=args.output_dir,
        counts_per_key=counts_per_key,
        stage2_bank_size=args.stage2_bank_size,
        quality_weight=args.quality_weight,
        time_weight=args.time_weight,
        time_limit=args.time_limit,
    )
    n_jobs = args.n_jobs
    if n_jobs == -1 or n_jobs is None:
        n_jobs = os.cpu_count() or 1

    n_inst = len(tuner.tuning_bank)
    n_s2 = len(tuner.stage2_bank)
    bank_s = n_inst * 5.0  # ~5 s / inst at 2000 iter
    bank_s2 = n_s2 * 15.0  # ~15 s / inst at 3000 iter

    # Print time budget upfront
    speedup = min(n_jobs, os.cpu_count() or 1)
    print(f"\n{'─' * 60}")
    print(
        f"  TIME BUDGET ESTIMATE  ({n_jobs} worker{'s' if n_jobs > 1 else ''})".center(
            60
        )
    )
    print(f"{'─' * 60}")
    total_est_h = 0.0
    baseline_h = 1 * bank_s / 3600 / speedup
    total_est_h += baseline_h
    print(f"  Baseline (1 eval × {n_inst} inst)               {baseline_h:.2f} h")

    if not args.no_isolation:
        iso_n = len(tuner.isolation_bank)
        iso_total = sum(len(g) for g in ISOLATION_GRIDS.values())
        iso_h = iso_total * iso_n * 5.0 / 3600 / speedup
        total_est_h += iso_h
        print(f"  Isolation ({iso_total} cfgs × {iso_n} inst)         {iso_h:.1f} h")

    s1_h = args.stage1_trials * bank_s / 3600 / speedup
    total_est_h += s1_h
    print(
        f"  Stage 1 ({args.stage1_trials} trials × {n_inst} inst)          {s1_h:.1f} h"
    )

    s2_h = args.stage2_trials * bank_s2 / 3600 / speedup
    total_est_h += s2_h
    print(f"  Stage 2 ({args.stage2_trials} trials × {n_s2} inst)         {s2_h:.1f} h")
    print(f"{'─' * 60}")
    print(
        f"  TOTAL ESTIMATED                        {total_est_h:.1f} h  ({total_est_h/24:.1f} days)"
    )
    print(f"{'─' * 60}")
    if total_est_h > 3 and n_jobs == 1:
        print("  Use --n-jobs N to parallelize across N CPU cores.")
        print("  Or use --quick for a fast smoke-test (~30 min).")
    print()

    # ── 1. Baseline ────────────────────────────────────────────────────────
    baseline = tuner.evaluate_defaults(
        max_iter=min(2000, args.iter_stage1), n_jobs=n_jobs
    )

    if args.baseline_only:
        tuner.print_best_config()
        return

    # ── 2. Isolation steps (optional) ──────────────────────────────────────
    isolation_best: dict[str, Any] | None = None
    if not args.no_isolation:
        isolation_best = tuner.run_isolation_steps(
            max_iter=min(2000, args.iter_stage1),
            n_jobs=n_jobs,
        )

    # ── 3. Stage 1: coarse ─────────────────────────────────────────────────
    stage1 = tuner.run_stage1_coarse(
        n_trials=args.stage1_trials,
        max_iter=args.iter_stage1,
        anchor=isolation_best,
        n_jobs=n_jobs,
    )

    # ── 4. Stage 2: fine ───────────────────────────────────────────────────
    stage2 = tuner.run_stage2_fine(
        coarse_study=stage1,
        n_trials=args.stage2_trials,
        max_iter=args.iter_stage2,
        bank=tuner.stage2_bank,
        n_jobs=n_jobs,
    )

    # ── 5. Final comparison ────────────────────────────────────────────────
    tuner.print_best_config(
        stage1=stage1,
        stage2=stage2,
        isolation_best=isolation_best,
    )

    print("\n  Re-evaluating best config on stage-2 bank...")
    best_params = {**DEFAULT_PARAMS, **stage2.best_params}
    best_result = tuner.evaluate_config(
        best_params,
        max_iter=args.iter_stage2,
        bank=tuner.stage2_bank,
        n_jobs=n_jobs,
    )
    tuner.save_log(
        "best_config_evaluation.json",
        "best_final",
        best_params,
        best_result,
        search_ranges=STAGE1_RANGES,
        baseline_params=DEFAULT_PARAMS,
    )
    # Also evaluate on the tuning bank for apples-to-apples against baseline
    best_result_tb = tuner.evaluate_config(
        best_params,
        max_iter=args.iter_stage1,
        bank=tuner.tuning_bank,
        n_jobs=n_jobs,
    )
    print(f"\n  {'=' * 45}")
    print(
        f"    Baseline relative gap:  {baseline.mean_relative_gap:.4f}  ({len(tuner.tuning_bank)} inst)"
    )
    print(
        f"    Best     relative gap:  {best_result_tb.mean_relative_gap:.4f}  ({len(tuner.tuning_bank)} inst)"
    )
    if best_result_tb.mean_relative_gap < baseline.mean_relative_gap:
        impr = baseline.mean_relative_gap - best_result_tb.mean_relative_gap
        pct = impr / max(baseline.mean_relative_gap, 1e-9) * 100
        print(f"    Improvement:   {impr:+.4f}  ({pct:+.1f}%)")
    else:
        print(
            f"    Change:        {baseline.mean_relative_gap - best_result_tb.mean_relative_gap:+.4f}"
        )
    print(f"  {'=' * 45}")
    print(f"\n  All results saved in:  {tuner.output_dir}/")


if __name__ == "__main__":
    main()
