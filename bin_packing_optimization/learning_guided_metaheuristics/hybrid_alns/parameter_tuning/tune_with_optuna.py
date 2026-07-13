"""Optuna hyperparameter tuning for the Hybrid ALNS solver — 1D Bin Packing.

Usage:
    from tune_with_optuna import HybridALNSTuner

    tuner = HybridALNSTuner()
    tuner.run_isolation_steps()                 # optional warm-up
    tuner.run_stage1_coarse()
    tuner.run_stage2_fine()
    tuner.print_best_config()

Or run directly:
    python tune_with_optuna.py --help
    python tune_with_optuna.py --stage1-trials 50 --stage2-trials 50
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from tqdm import tqdm

from bin_packing_optimization.datasets.registry import DATASET_REGISTRY
from bin_packing_optimization.datasets.types import Instance

from tuning_config import (
    BANK_COMPOSITION,
    BANK_COMPOSITION_TOTAL,
    BASELINE_FILENAME,
    BEST_EVAL_FILENAME,
    DEFAULT_BANK_SIZE,
    DEFAULT_MODEL,
    DEFAULT_N_JOBS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARAMS,
    DEFAULT_SEED,
    DEFAULT_QUALITY_WEIGHT,
    DEFAULT_TIME_WEIGHT,
    DEFAULT_TIME_LIMIT,
    ISOLATION_BANK_SIZE,
    ISOLATION_GRIDS,
    ISOLATION_MERGED_FILENAME,
    ISOLATION_STEP_TEMPLATE,
    ITER_BASELINE,
    ITER_ISOLATION,
    ITER_STAGE1,
    ITER_STAGE2,
    N_DESTROY_ARMS,
    PIPELINE_SUMMARY_FILENAME,
    QUICK_BANK_SIZE,
    QUICK_ITER_STAGE1,
    QUICK_ITER_STAGE2,
    QUICK_STAGE1_TRIALS,
    QUICK_STAGE2_TRIALS,
    SECONDS_PER_HOUR,
    SECONDS_PER_INST_2000ITER,
    SECONDS_PER_INST_5000ITER,
    STAGE1_RANGES,
    STAGE1_STUDY_NAME,
    STAGE1_TRIALS,
    STAGE2_RANGE_OFFSETS,
    STAGE2_STUDY_NAME,
    STAGE2_TRIALS,
    STRUCTURAL_DEFAULTS,
    STUDY_FILENAME_TEMPLATE,
    WARNING_HOURS_THRESHOLD,
)

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


def _params_diff(
    base: dict[str, Any], changed: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return ``{param: (base_val, changed_val)}`` for keys that differ."""
    return {
        k: (base[k], changed[k]) for k in changed if k in base and base[k] != changed[k]
    }


def _save_json_atomic(data: dict[str, Any], path: str | Path) -> None:
    """Atomically write *data* as JSON to *path*.

    Writes to a temporary file first, then renames (``os.replace``) so the
    target file is never left in a partially-written state.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    print(f"  -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PARALLEL WORKER (module-level for multiprocessing pickling)
# ─────────────────────────────────────────────────────────────────────────────

# Global model cache shared across worker processes — avoids reloading
# the repair model from disk for every single instance evaluation.
_WORKER_MODEL_CACHE: dict[str, Any] = {}


def _worker_init(model_name: str) -> None:
    """Load the repair model once per worker process."""
    from bin_packing_optimization.learning_guided_metaheuristics.hybrid_alns.models import (
        load_repair_model,
    )

    _WORKER_MODEL_CACHE["bundle"] = load_repair_model(model_name)


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

    from bin_packing_optimization.learning_guided_metaheuristics.hybrid_alns.hybrid_alns_solver import (
        BinPackingSolver,
    )

    model_bundle = _WORKER_MODEL_CACHE.get("bundle")
    if model_bundle is None:
        from bin_packing_optimization.learning_guided_metaheuristics.hybrid_alns.models import (
            load_repair_model,
        )

        model_bundle = load_repair_model(model_name)
        _WORKER_MODEL_CACHE["bundle"] = model_bundle

    sizes = list(inst_dict["sizes"])
    solver = BinPackingSolver(sizes, inst_dict["bin_capacity"], seed=seed)

    # Build a local copy so we never mutate the caller's shared kwargs dict.
    call_kwargs = {**kwargs, "time_limit_seconds": time_limit}
    t0 = time.perf_counter()
    try:
        solver.solve(model_bundle=model_bundle, max_iterations=max_iter, **call_kwargs)
        elapsed = time.perf_counter() - t0

        sol = solver.get_solution()
        lb = int(math.ceil(sum(sizes) / inst_dict["bin_capacity"]))
        relative_gap = (sol.total_bins_used - lb) / max(lb, 1)

        arm_pulls = [0] * N_DESTROY_ARMS
        if hasattr(solver, "_bandit"):
            for arm_idx in range(N_DESTROY_ARMS):
                arm_pulls[arm_idx] = solver._bandit.arm_counts[arm_idx]

        normalized_time = elapsed / max(time_limit, 1e-9)
    except Exception:
        elapsed = 0.0
        relative_gap = float("inf")
        normalized_time = float("inf")
        arm_pulls = [0, 0, 0]

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
    3. **Stage 1 (coarse)** — wide-range Bayesian search (5 000 iter/inst)
       →  :file:`alns_stage1_coarse_study.json`
    4. **Stage 2 (fine)** — narrow-range search anchored on Stage-1 best
       (5 000 iter/inst) →  :file:`alns_stage2_fine_study.json`
    5. **Final** — evaluate merged best →  :file:`best_config_evaluation.json`
    """

    def __init__(
        self,
        counts_per_key: dict[str, int] | None = None,
        seed: int = DEFAULT_SEED,
        model_name: str = DEFAULT_MODEL,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        bank_size: int = DEFAULT_BANK_SIZE,
        quality_weight: float = DEFAULT_QUALITY_WEIGHT,
        time_weight: float = DEFAULT_TIME_WEIGHT,
        time_limit: float = DEFAULT_TIME_LIMIT,
    ):
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name

        self.quality_weight = quality_weight
        self.time_weight = time_weight
        self.time_limit = time_limit

        # Build instance bank proportionally from BANK_COMPOSITION
        if counts_per_key is None:
            scale = bank_size / BANK_COMPOSITION_TOTAL
            counts_per_key = {
                key: c
                for key, c in ((key, round(w * scale)) for key, w in BANK_COMPOSITION)
                if c > 0
            }
        self.counts_per_key = counts_per_key

        # Main evaluation bank — used by baseline, Stage 1, Stage 2, and final
        self.tuning_bank: list[Instance] = self._build_bank()
        print(f"\nTuning bank: {len(self.tuning_bank)} instances")
        self._log_bank_summary()

        # Smaller bank for fast isolation steps — stratified across datasets
        self.isolation_bank: list[Instance] = self._build_isolation_bank()
        print(
            f"  Isolation bank: {len(self.isolation_bank)} instances "
            f"(stratified subset, for fast grid search)"
        )

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
        max_iter: int = ITER_BASELINE,
        force_uniform_random: bool = False,
        bank: list[Instance] | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> TuningResult:
        """Run the solver on *bank* instances and return aggregate metrics.

        Parameters
        ----------
        solver_kwargs:
            Parameter overrides (e.g. ``{"alpha_cool": 0.9993}``).
            Missing params are filled from ``DEFAULT_PARAMS``.
        max_iter:
            ALNS iterations per instance.
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
                    d,
                    self.model_name,
                    self.seed + idx,
                    max_iter,
                    self.time_limit,
                    kwargs,
                )
                for idx, d in enumerate(
                    tqdm(inst_dicts, desc="  Solving", leave=False, position=1)
                )
            ]
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_worker_init,
                initargs=(self.model_name,),
            ) as ex:
                futures = [
                    ex.submit(
                        _evaluate_one,
                        d,
                        self.model_name,
                        self.seed + idx,
                        max_iter,
                        self.time_limit,
                        kwargs,
                    )
                    for idx, d in enumerate(inst_dicts)
                ]
                results = []
                for f in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="  Solving",
                    leave=False,
                    position=1,
                ):
                    try:
                        result = f.result(timeout=600)
                        results.append(result)
                    except Exception:
                        results.append({
                            "dataset_key": "unknown",
                            "relative_gap": float("inf"),
                            "normalized_time": float("inf"),
                            "elapsed": 0.0,
                            "arm_pulls": [0, 0, 0],
                        })

        total_relative_gap = 0.0
        total_normalized_time = 0.0
        total_raw_time = 0.0
        per_ds_relative_gap: dict[str, list[float]] = {}
        arm_pulls = [0] * N_DESTROY_ARMS

        for r in results:
            total_relative_gap += r["relative_gap"]
            total_normalized_time += r["normalized_time"]
            total_raw_time += r["elapsed"]
            per_ds_relative_gap.setdefault(r["dataset_key"], []).append(
                r["relative_gap"]
            )
            for arm_idx in range(N_DESTROY_ARMS):
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
        bank_size_override: int | None = None,
        *,
        max_iter: int | None = None,
        n_jobs: int | None = None,
        force_uniform_random: bool | None = None,
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
        bank_size_override:
            If provided, overrides ``bank_size`` in the saved JSON
            (e.g. for isolation steps that use the smaller isolation bank).
        max_iter:
            ALNS iterations per instance used for this evaluation.
        n_jobs:
            Number of parallel workers used.
        force_uniform_random:
            Whether the bandit was forced to uniform random.
        """
        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "run_type": run_type,
            "seed": self.seed,
            "model_name": self.model_name,
            "bank_size": (
                bank_size_override
                if bank_size_override is not None
                else len(self.tuning_bank)
            ),
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
        if max_iter is not None:
            payload["max_iter"] = max_iter
        if n_jobs is not None:
            payload["n_jobs"] = n_jobs
        if force_uniform_random is not None:
            payload["force_uniform_random"] = force_uniform_random

        _save_json_atomic(payload, self.output_dir / filename)

    # ── Baseline ───────────────────────────────────────────────────────────

    def evaluate_defaults(
        self, max_iter: int = ITER_BASELINE, n_jobs: int = DEFAULT_N_JOBS
    ) -> TuningResult:
        """Evaluate the default config as a baseline reference."""
        print("\n" + "=" * 70)
        print("  BASELINE  —  default parameters  ".center(70, " "))
        print("=" * 70)
        result = self.evaluate_config({}, max_iter=max_iter, n_jobs=n_jobs)
        self.save_log(
            BASELINE_FILENAME,
            "baseline",
            DEFAULT_PARAMS,
            result,
            baseline_params=DEFAULT_PARAMS,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        print(f"\n  mean_relative_gap  = {result.mean_relative_gap:.4f}")
        print(
            f"  mean_normalized_time = {result.mean_normalized_time:.4f}  ({result.mean_time_s:.3f} s / {self.time_limit} s)"
        )
        return result

    # ── Isolation steps ────────────────────────────────────────────────────

    def run_isolation_steps(
        self,
        max_iter: int = ITER_ISOLATION,
        n_jobs: int = DEFAULT_N_JOBS,
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
            _ISOLATION_LABELS: dict[str, str] = {
                "step1_sa_thermal": "Step 1 — SA Thermal",
                "step2_destruction": "Step 2 — Destruction",
                "step3_bandit": "Step 3 — Bandit",
            }
            label = _ISOLATION_LABELS.get(step_key, step_key)
            n_cfg = len(grid)
            print(f"\n  ── {label}  ({n_cfg} configs × {len(bank)} instances) ──")

            if step_key == "step3_bandit":
                step_uniform = False  # bandit params need the bandit ON
            else:
                step_uniform = True  # disable bandit confounding

            best_gap = float("inf")
            best_params = {}
            best_result: TuningResult | None = None

            for trial_idx, candidate in enumerate(
                tqdm(grid, desc=f"  {label}", leave=False, position=0)
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
                    best_result = result

                n_save = max(1, len(grid) // 3)
                if trial_idx % n_save == 0 or trial_idx == len(grid) - 1:
                    tqdm.write(
                        f"    gap={result.mean_relative_gap:.4f}  "
                        f"best={best_gap:.4f}  |  "
                        + ", ".join(f"{k}={v}" for k, v in candidate.items())
                    )

            # Log best of this step — reuse cached best_result, avoid re-eval
            step_result = best_result
            self.save_log(
                ISOLATION_STEP_TEMPLATE.format(step_key=step_key),
                f"isolation_{step_key}",
                best_params,
                step_result,
                search_ranges={"grid": grid, "bank_size": len(bank)},
                baseline_params=DEFAULT_PARAMS,
                bank_size_override=len(bank),
                max_iter=max_iter,
                n_jobs=n_jobs,
                force_uniform_random=step_uniform,
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
            ISOLATION_MERGED_FILENAME,
            "isolation_merged",
            merged,
            merged_result,
            baseline_params=DEFAULT_PARAMS,
            bank_size_override=len(bank),
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        print(
            f"\n  >> Merged best from isolation: gap={merged_result.mean_relative_gap:.4f}"
        )
        print(f"\n{'=' * 70}")

        return merged

    # ── Stage 1: Coarse ───────────────────────────────────────────────────

    def run_stage1_coarse(
        self,
        n_trials: int = STAGE1_TRIALS,
        max_iter: int = ITER_STAGE1,
        study_name: str = STAGE1_STUDY_NAME,
        anchor: dict[str, Any] | None = None,
        n_jobs: int = DEFAULT_N_JOBS,
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
            if params.get("k_min_frac", 0) >= params.get("k_max_frac", 1):
                raise optuna.TrialPruned()
            # hard restart uses a *smaller* temperature multiplier than soft reheat
            # (hard = stronger cooldown).  Invert that constraint → invalid config.
            if params.get("reheat_hard_mult", 0) >= params.get("reheat_soft_mult", 1):
                raise optuna.TrialPruned()
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

        t0 = time.perf_counter()
        study.optimize(objective, n_trials=n_trials)
        elapsed = time.perf_counter() - t0
        self._report_study(study, "Stage 1 (coarse)")
        self._save_study(
            study, study_name,
            search_ranges=STAGE1_RANGES,
            elapsed=elapsed,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        return study

    # ── Stage 2: Fine ──────────────────────────────────────────────────────

    def run_stage2_fine(
        self,
        coarse_study: optuna.Study | None = None,
        n_trials: int = STAGE2_TRIALS,
        max_iter: int = ITER_STAGE2,
        study_name: str = STAGE2_STUDY_NAME,
        n_jobs: int = DEFAULT_N_JOBS,
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
        # Per-parameter hard ceilings: some params have strict solver-enforced upper
        # bounds that the offset arithmetic could violate when the Stage-1 best sits
        # near the edge of the Stage-1 range.  Clamp hi to stay inside valid bounds.
        _PARAM_HARD_CEIL: dict[str, float] = {
            "alpha_cool": 1.0 - 1e-7,  # solver requires alpha_cool < 1
        }
        STAGE2_RANGES: dict[str, dict[str, Any]] = {}
        for k, offsets in STAGE2_RANGE_OFFSETS.items():
            anchor_val = anchor[k]
            lo: float | int
            hi: float | int
            if "low_factor" in offsets:
                lo = anchor_val * offsets["low_factor"]
                hi = anchor_val * offsets["high_factor"]
            else:
                lo = anchor_val + offsets["low_offset"]
                hi = anchor_val + offsets["high_offset"]
            floor = 0 if offsets.get("type") == "int" else 1e-6
            lo = max(floor, lo)
            if k in _PARAM_HARD_CEIL:
                hi = min(hi, _PARAM_HARD_CEIL[k])
            spec: dict[str, Any] = {"low": lo, "high": hi}
            if offsets.get("log"):
                spec["log"] = True
            if offsets.get("type"):
                spec["type"] = offsets["type"]
            STAGE2_RANGES[k] = spec

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
            if params.get("k_min_frac", 0) >= params.get("k_max_frac", 1):
                raise optuna.TrialPruned()
            if params.get("reheat_hard_mult", 0) >= params.get("reheat_soft_mult", 1):
                raise optuna.TrialPruned()
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
        study.enqueue_trial(anchor)

        t0 = time.perf_counter()
        study.optimize(objective, n_trials=n_trials)
        elapsed = time.perf_counter() - t0
        self._report_study(study, "Stage 2 (fine)")
        self._save_study(
            study, study_name,
            search_ranges=STAGE2_RANGES,
            elapsed=elapsed,
            anchor_params=anchor,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
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
        *,
        elapsed: float | None = None,
        anchor_params: dict[str, Any] | None = None,
        max_iter: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        sampler = study.sampler
        sampler_info: dict[str, Any] = {"name": sampler.__class__.__name__}
        if hasattr(sampler, "n_startup_trials"):
            sampler_info["n_startup_trials"] = sampler.n_startup_trials
        if hasattr(sampler, "seed"):
            sampler_info["seed"] = sampler.seed

        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        n_failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)

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
            "direction": study.direction.name,
            "sampler": sampler_info,
            "timestamp": time.time(),
            "seed": self.seed,
            "pipeline_config": {
                "quality_weight": self.quality_weight,
                "time_weight": self.time_weight,
                "time_limit": self.time_limit,
                "max_iter": max_iter,
                "n_jobs": n_jobs,
            },
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials_total": len(study.trials),
            "n_trials_completed": len(trials_data),
            "n_trials_pruned": n_pruned,
            "n_trials_failed": n_failed,
            "trials": trials_data,
        }
        if elapsed is not None:
            payload["elapsed_s"] = elapsed
        if search_ranges is not None:
            payload["search_ranges"] = search_ranges
        if anchor_params is not None:
            payload["anchor_params"] = anchor_params
        _save_json_atomic(payload, self.output_dir / STUDY_FILENAME_TEMPLATE.format(name=name))

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

        # Column headers — widths are fixed: 28 chars for param name, 24 per stage column.
        cols = ["Parameter"] + list(stages.keys()) + ["Search range (Stage 1)"]
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
            best = {**DEFAULT_PARAMS, **STRUCTURAL_DEFAULTS, **src.best_params}
            print(f"\n  Best configuration found:")
            print(f"  {'=' * 45}")
            for k, v in best.items():
                print(f"    {k:30s} = {v}")
            return best

        return dict(DEFAULT_PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def _save_pipeline_summary(
    tuner: HybridALNSTuner,
    args: Any,
    timing: dict[str, float],
    *,
    baseline: TuningResult | None = None,
    best_result: TuningResult | None = None,
) -> None:
    """Save a pipeline-summary JSON tying all steps together.

    Written to ``{output_dir}/pipeline_summary.json``.
    """
    output_files: dict[str, str] = {
        "baseline": BASELINE_FILENAME,
        "isolation_step_template": ISOLATION_STEP_TEMPLATE,
        "isolation_merged": ISOLATION_MERGED_FILENAME,
        "stage1": STUDY_FILENAME_TEMPLATE.format(name=STAGE1_STUDY_NAME),
        "stage2": STUDY_FILENAME_TEMPLATE.format(name=STAGE2_STUDY_NAME),
        "best_eval": BEST_EVAL_FILENAME,
    }
    results_payload: dict[str, Any] = {}
    if baseline is not None:
        results_payload["baseline_mean_relative_gap"] = baseline.mean_relative_gap
        results_payload["baseline_mean_normalized_time"] = baseline.mean_normalized_time
    if best_result is not None:
        results_payload["best_mean_relative_gap"] = best_result.mean_relative_gap
        results_payload["best_mean_normalized_time"] = best_result.mean_normalized_time
        impr = baseline.mean_relative_gap - best_result.mean_relative_gap if baseline else 0
        results_payload["improvement_abs"] = round(impr, 6)
        results_payload["improvement_pct"] = round(
            impr / max(baseline.mean_relative_gap, 1e-9) * 100, 2
        ) if baseline else None

    cli_args = {
        k: v for k, v in vars(args).items()
        if k != "func"  # in case subparsers are added later
    }
    summary = {
        "timestamp": time.time(),
        "cli_args": cli_args,
        "timing_seconds": timing,
        "results": results_payload,
        "output_files": output_files,
    }
    _save_json_atomic(summary, tuner.output_dir / PIPELINE_SUMMARY_FILENAME)


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
            "  python tune_with_optuna.py --bank-size 100 "
            "--stage1-trials 80 --stage2-trials 60 --n-jobs 8\n\n"
            "  # Baseline only:\n"
            "  python tune_with_optuna.py --baseline-only\n\n"
            "  # Resume: skip to Stage 2 using a Stage-1 JSON on disk:\n"
            "  python tune_with_optuna.py --stage2-only "
            "--stage1-json tuning_results/alns_stage1_coarse_study.json\n"
        ),
    )
    parser.add_argument(
        "--stage1-trials",
        type=int,
        default=STAGE1_TRIALS,
        help=f"Stage 1 (coarse) trials (default: {STAGE1_TRIALS})",
    )
    parser.add_argument(
        "--stage2-trials",
        type=int,
        default=STAGE2_TRIALS,
        help=f"Stage 2 (fine) trials (default: {STAGE2_TRIALS})",
    )
    parser.add_argument(
        "--iter-stage1",
        type=int,
        default=ITER_STAGE1,
        help=f"ALNS iterations per instance in Stage 1 (default: {ITER_STAGE1})",
    )
    parser.add_argument(
        "--iter-stage2",
        type=int,
        default=ITER_STAGE2,
        help=f"ALNS iterations per instance in Stage 2 (default: {ITER_STAGE2})",
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=DEFAULT_BANK_SIZE,
        help=f"Instance bank size (default: {DEFAULT_BANK_SIZE})",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Parallel workers. -1 = all CPUs (default: {DEFAULT_N_JOBS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
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
        "--stage2-only",
        action="store_true",
        help=(
            "Skip baseline, isolation, and Stage 1; run Stage 2 only. "
            "Requires --stage1-json to load the Stage-1 best params."
        ),
    )
    parser.add_argument(
        "--stage1-json",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a Stage-1 study JSON (produced by a previous run). "
            "Used with --stage2-only to resume after an interrupted Stage 1."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Minimal: 5 trials S1, 3 trials S2, no isolation, bank=10",
    )

    parser.add_argument(
        "--quality-weight",
        type=float,
        default=DEFAULT_QUALITY_WEIGHT,
        help=f"Quality coefficient in composite objective (default: {DEFAULT_QUALITY_WEIGHT})",
    )
    parser.add_argument(
        "--time-weight",
        type=float,
        default=DEFAULT_TIME_WEIGHT,
        help=f"Time coefficient in composite objective (default: {DEFAULT_TIME_WEIGHT})",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=f"Time normaliser in seconds for elapsed / time_limit (default: {DEFAULT_TIME_LIMIT})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Repair model filename loaded by the solver (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    # Apply --quick override
    if args.quick:
        args.stage1_trials = QUICK_STAGE1_TRIALS
        args.stage2_trials = QUICK_STAGE2_TRIALS
        args.no_isolation = True
        args.iter_stage1 = QUICK_ITER_STAGE1
        args.iter_stage2 = QUICK_ITER_STAGE2
        args.bank_size = QUICK_BANK_SIZE

    # Build proportional per-dataset counts from the requested bank size
    scale = args.bank_size / BANK_COMPOSITION_TOTAL
    counts_per_key = {
        key: c
        for key, c in ((key, round(w * scale)) for key, w in BANK_COMPOSITION)
        if c > 0
    }

    tuner = HybridALNSTuner(
        seed=args.seed,
        model_name=args.model_name,
        output_dir=args.output_dir,
        counts_per_key=counts_per_key,
        quality_weight=args.quality_weight,
        time_weight=args.time_weight,
        time_limit=args.time_limit,
    )
    # ── 1. Validate mutually exclusive mode flags ──────────────────────────
    if args.stage2_only and args.baseline_only:
        parser.error("--stage2-only and --baseline-only are mutually exclusive.")
    if args.stage2_only and args.stage1_json is None:
        parser.error("--stage2-only requires --stage1-json <path>.")

    n_inst = len(tuner.tuning_bank)
    # ~5 s/inst at 2000 iter (baseline / isolation); ~12.5 s/inst at 5000 iter (Optuna stages)
    bank_s = n_inst * SECONDS_PER_INST_5000ITER
    n_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
    speedup = min(n_workers, os.cpu_count() or 1)
    print(f"\n{'─' * 60}")
    print(
        f"  TIME BUDGET ESTIMATE  ({n_workers} worker{'s' if n_workers > 1 else ''})".center(
            60
        )
    )
    print(f"{'─' * 60}")
    total_est_h = 0.0

    if not args.stage2_only:
        baseline_h = bank_s / SECONDS_PER_HOUR / speedup
        total_est_h += baseline_h
        print(f"  Baseline (1 eval × {n_inst} inst)               {baseline_h:.2f} h")

        if not args.no_isolation:
            iso_n = len(tuner.isolation_bank)
            iso_total = sum(len(g) for g in ISOLATION_GRIDS.values())
            iso_h = (
                iso_total
                * iso_n
                * SECONDS_PER_INST_2000ITER
                / SECONDS_PER_HOUR
                / speedup
            )
            total_est_h += iso_h
            print(
                f"  Isolation ({iso_total} cfgs × {iso_n} inst)         {iso_h:.1f} h"
            )

        if not args.baseline_only:
            s1_h = args.stage1_trials * bank_s / SECONDS_PER_HOUR / speedup
            total_est_h += s1_h
            print(
                f"  Stage 1 ({args.stage1_trials} trials × {n_inst} inst)          {s1_h:.1f} h"
            )

    s2_h = args.stage2_trials * bank_s / SECONDS_PER_HOUR / speedup
    total_est_h += s2_h
    print(
        f"  Stage 2 ({args.stage2_trials} trials × {n_inst} inst)          {s2_h:.1f} h"
    )
    print(f"{'─' * 60}")
    print(
        f"  TOTAL ESTIMATED                        {total_est_h:.1f} h  ({total_est_h/24:.1f} days)"
    )
    print(f"{'─' * 60}")
    if total_est_h > WARNING_HOURS_THRESHOLD and n_workers == 1:
        print("  Use --n-jobs N to parallelize across N CPU cores.")
        print("  Or use --quick for a fast smoke-test (~30 min).")
    print()

    # ── --stage2-only: load Stage-1 results from disk and jump straight to S2 ──
    if args.stage2_only:
        stage1_path = Path(args.stage1_json)
        if not stage1_path.exists():
            parser.error(f"--stage1-json path not found: {stage1_path}")
        with open(stage1_path) as f:
            stage1_data = json.load(f)
        stage1_best_params: dict[str, Any] = stage1_data.get("best_params", {})
        print(
            f"  Loaded Stage-1 best params from {stage1_path}  "
            f"(best_value={stage1_data.get('best_value', 'n/a')})"
        )

        # Reconstruct a minimal Optuna study so run_stage2_fine receives the
        # standard coarse_study argument without any special-casing inside the method.
        _s1_stub = optuna.create_study(
            direction="minimize", study_name="stage1_stub_from_json"
        )
        _s1_stub.add_trial(
            optuna.trial.create_trial(
                params=stage1_best_params,
                distributions={
                    k: optuna.distributions.FloatDistribution(0.0, 1.0)
                    for k in stage1_best_params
                },
                value=stage1_data.get("best_value", 0.0),
            )
        )

        _timing: dict[str, float] = {}
        t0 = time.perf_counter()
        stage2 = tuner.run_stage2_fine(
            coarse_study=_s1_stub,
            n_trials=args.stage2_trials,
            max_iter=args.iter_stage2,
            n_jobs=args.n_jobs,
        )
        _timing["stage2_s"] = round(time.perf_counter() - t0, 3)
        tuner.print_best_config(stage2=stage2)
        _save_pipeline_summary(tuner, args, _timing)
        print(f"\n  All results saved in:  {tuner.output_dir}/")
        return

    # ── Track actual execution time for each stage ────────────────────────
    timing: dict[str, float] = {}

    # ── 1. Baseline ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    baseline = tuner.evaluate_defaults(
        max_iter=min(ITER_BASELINE, args.iter_stage1), n_jobs=args.n_jobs
    )
    timing["baseline_s"] = round(time.perf_counter() - t0, 3)

    if args.baseline_only:
        tuner.print_best_config()
        # Save minimal summary
        _save_pipeline_summary(
            tuner, args, timing, baseline=baseline, best_result=None,
        )
        return

    # ── 2. Isolation steps (optional) ──────────────────────────────────────
    isolation_best: dict[str, Any] | None = None
    if not args.no_isolation:
        t0 = time.perf_counter()
        isolation_best = tuner.run_isolation_steps(
            max_iter=min(ITER_ISOLATION, args.iter_stage1),
            n_jobs=args.n_jobs,
        )
        timing["isolation_s"] = round(time.perf_counter() - t0, 3)

    # ── 3. Stage 1: coarse ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    stage1 = tuner.run_stage1_coarse(
        n_trials=args.stage1_trials,
        max_iter=args.iter_stage1,
        anchor=isolation_best,
        n_jobs=args.n_jobs,
    )
    timing["stage1_s"] = round(time.perf_counter() - t0, 3)

    # ── 4. Stage 2: fine ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    stage2 = tuner.run_stage2_fine(
        coarse_study=stage1,
        n_trials=args.stage2_trials,
        max_iter=args.iter_stage2,
        n_jobs=args.n_jobs,
    )
    timing["stage2_s"] = round(time.perf_counter() - t0, 3)

    # ── 5. Final comparison ────────────────────────────────────────────────
    tuner.print_best_config(
        stage1=stage1,
        stage2=stage2,
        isolation_best=isolation_best,
    )

    print("\n  Re-evaluating best config...")
    best_params = {**DEFAULT_PARAMS, **stage2.best_params}
    # Evaluate best on the same tuning bank as baseline for apples-to-apples
    t0 = time.perf_counter()
    best_result = tuner.evaluate_config(
        best_params,
        max_iter=args.iter_stage2,
        n_jobs=args.n_jobs,
    )
    timing["best_eval_s"] = round(time.perf_counter() - t0, 3)
    tuner.save_log(
        BEST_EVAL_FILENAME,
        "best_final",
        best_params,
        best_result,
        search_ranges=STAGE1_RANGES,
        baseline_params=DEFAULT_PARAMS,
        max_iter=args.iter_stage2,
        n_jobs=args.n_jobs,
    )
    print(f"\n  {'=' * 45}")
    print(
        f"    Baseline relative gap:  {baseline.mean_relative_gap:.4f}  ({n_inst} inst)"
    )
    print(
        f"    Best     relative gap:  {best_result.mean_relative_gap:.4f}  ({n_inst} inst)"
    )
    if best_result.mean_relative_gap < baseline.mean_relative_gap:
        impr = baseline.mean_relative_gap - best_result.mean_relative_gap
        pct = impr / max(baseline.mean_relative_gap, 1e-9) * 100
        print(f"    Improvement:   {impr:+.4f}  ({pct:+.1f}%)")
    else:
        print(
            f"    Change:        {baseline.mean_relative_gap - best_result.mean_relative_gap:+.4f}"
        )
    print(f"  {'=' * 45}")
    print(f"\n  All results saved in:  {tuner.output_dir}/")

    # ── Save pipeline summary ──────────────────────────────────────────────
    _save_pipeline_summary(
        tuner, args, timing,
        baseline=baseline,
        best_result=best_result,
    )


if __name__ == "__main__":
    main()
