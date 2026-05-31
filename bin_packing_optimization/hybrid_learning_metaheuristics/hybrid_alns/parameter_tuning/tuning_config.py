"""Single source of truth for all tuning configuration.

Import this module in both ``tune_with_optuna.py`` and
``parameter_tuning.ipynb`` to keep parameter definitions, defaults,
ranges, and grid compositions consistent across the pipeline.
"""

from __future__ import annotations

import math
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TUNABLE PARAMETERS — 9 metaheuristic knobs
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# 2.  STRUCTURAL PARAMETERS — safety floors / design constants (held fixed)
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURAL_DEFAULTS: dict[str, Any] = {
    "min_no_improve_limit": 250,
    "temp_precision_floor": 1e-12,
    "reheat_check_interval_divisor": 4.0,
    "reheat_check_min_interval": 50,
    "hard_restart_min_limit": 100,
    "patience_shrink_factor": 2.0 / 3.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PIPELINE CONFIGURATION — objective weighting & time limit
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_QUALITY_WEIGHT: float = 0.9
DEFAULT_TIME_WEIGHT: float = 0.1
DEFAULT_TIME_LIMIT: float = 60

PIPELINE_CONFIG: dict[str, Any] = {
    "quality_weight": DEFAULT_QUALITY_WEIGHT,
    "time_weight": DEFAULT_TIME_WEIGHT,
    "time_limit": DEFAULT_TIME_LIMIT,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  INSTANCE BANK — dataset composition
# ─────────────────────────────────────────────────────────────────────────────

BANK_COMPOSITION: list[tuple[str, int]] = [
    ("falkenauer-t", 5),
    ("falkenauer-u", 40),
    ("scholl-1", 10),
    ("scholl-2", 35),
    ("scholl-3", 10),
]
BANK_COMPOSITION_TOTAL: int = sum(w for _, w in BANK_COMPOSITION)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  ISOLATION STEPS — fast grid-search config
# ─────────────────────────────────────────────────────────────────────────────

# Each isolation step tests one group of params with uniform-random destruction
# (bandit disabled) to measure the group's effect without confounding.
# Grids are kept small (3 values per param) for speed.

ISOLATION_BANK_SIZE: int = 20

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
        if h < s  # hard restart must be stronger cooldown than soft reheat
    ],  # 3×3×3×3 = 81, filtered to 54 (excludes h ≥ s)
    "step2_destruction": [
        {"k_min_frac": kmin, "k_max_frac": kmax, "no_improve_frac": ni}
        for kmin in [0.01, 0.05, 0.20]
        for kmax in [0.10, 0.25, 0.50]
        for ni in [0.01, 0.05, 0.20]
        if kmax > kmin
    ],  # 3×3×3 = 27, filtered to valid (excludes k_max ≤ k_min)
    "step3_bandit": [
        {"bandit_alpha": a, "warmup_calls": w}
        for a in [0.01, 0.30, 2.00]
        for w in [0, 300, 1000]
    ],  # 3×3 = 9 configs
}

# ─────────────────────────────────────────────────────────────────────────────
# 6.  ITERATION & TRIAL BUDGETS
# ─────────────────────────────────────────────────────────────────────────────

ITER_BASELINE: int = 2000  # ALNS iterations per instance — baseline eval
ITER_ISOLATION: int = 2000  # ALNS iterations per instance — isolation steps
ITER_STAGE1: int = 5000  # ALNS iterations per instance — Stage 1 (coarse)
ITER_STAGE2: int = 5000  # ALNS iterations per instance — Stage 2 (fine)

STAGE1_TRIALS: int = 50  # Number of Optuna trials — Stage 1
STAGE2_TRIALS: int = 50  # Number of Optuna trials — Stage 2

STAGE1_STUDY_NAME: str = "alns_stage1_coarse"
STAGE2_STUDY_NAME: str = "alns_stage2_fine"

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TUNER CONSTRUCTOR DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SEED: int = 42
DEFAULT_MODEL: str = "repair_model_v2.pkl"
DEFAULT_OUTPUT_DIR: str = "tuning_results"
DEFAULT_BANK_SIZE: int = 100
DEFAULT_N_JOBS: int = 1

# ─────────────────────────────────────────────────────────────────────────────
# 8.  OUTPUT FILENAMES
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_FILENAME: str = "baseline_defaults.json"
ISOLATION_STEP_TEMPLATE: str = "isolation_{step_key}.json"
ISOLATION_MERGED_FILENAME: str = "isolation_merged.json"
STUDY_FILENAME_TEMPLATE: str = "{name}_study.json"
BEST_EVAL_FILENAME: str = "best_config_evaluation.json"
PIPELINE_SUMMARY_FILENAME: str = "pipeline_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# 9.  STAGE 2 RANGE OFFSETS  —  narrow windows around Stage-1 best
# ─────────────────────────────────────────────────────────────────────────────

STAGE2_RANGE_OFFSETS: dict[str, dict[str, Any]] = {
    "initial_temperature": {"low_factor": 0.85, "high_factor": 1.15, "log": True},
    "alpha_cool": {"low_offset": -0.0005, "high_offset": 0.0005},
    "reheat_soft_mult": {"low_offset": -0.10, "high_offset": 0.10},
    "reheat_hard_mult": {"low_offset": -0.08, "high_offset": 0.08},
    "k_min_frac": {"low_factor": 0.75, "high_factor": 1.25},
    "k_max_frac": {"low_factor": 0.75, "high_factor": 1.25},
    "no_improve_frac": {"low_factor": 0.75, "high_factor": 1.25},
    "bandit_alpha": {"low_factor": 0.80, "high_factor": 1.20, "log": True},
    "warmup_calls": {"low_offset": -150, "high_offset": 150, "type": "int"},
}

# ─────────────────────────────────────────────────────────────────────────────
# 10.  DESTROY OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

N_DESTROY_ARMS: int = 3

# ─────────────────────────────────────────────────────────────────────────────
# 11.  QUICK-MODE OVERRIDES  —  ``--quick`` flag values
# ─────────────────────────────────────────────────────────────────────────────

QUICK_STAGE1_TRIALS: int = 5
QUICK_STAGE2_TRIALS: int = 3
QUICK_ITER_STAGE1: int = 500
QUICK_ITER_STAGE2: int = 1000
QUICK_BANK_SIZE: int = 10

# ─────────────────────────────────────────────────────────────────────────────
# 12.  TIME-ESTIMATE CONSTANTS  —  rough per-instance timing for budget hints
# ─────────────────────────────────────────────────────────────────────────────

SECONDS_PER_INST_2000ITER: float = 5.0
SECONDS_PER_INST_5000ITER: float = 12.5
SECONDS_PER_HOUR: int = 3600
WARNING_HOURS_THRESHOLD: int = 3

# ─────────────────────────────────────────────────────────────────────────────
# 13.  OPTUNA SEARCH RANGES — Stage 1 (coarse)
# ─────────────────────────────────────────────────────────────────────────────

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
