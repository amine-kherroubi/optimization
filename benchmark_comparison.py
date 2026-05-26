"""Benchmark multi-méthodes : comparaison de toutes les stratégies de bandit.

Méthodes comparées
------------------
1. TS          — Thompson Sampling Beta-Bernoulli (sans contexte)
2. LinUCB-0.3  — LinUCB contextuel, alpha=0.3  (meilleur alpha trouvé)
3. LinUCB-1.0  — LinUCB contextuel, alpha=1.0  (valeur par défaut originale)
4. WS-20       — TS 200 iters warm-up → LinUCB alpha=0.3
5. WS-30       — TS 300 iters warm-up → LinUCB alpha=0.3

Usage :
    python3 benchmark_comparison.py
    python3 benchmark_comparison.py --iters 1000 --instances 5
    python3 benchmark_comparison.py --iters 1000 --instances 999   # tous
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

# ── Imports projet ────────────────────────────────────────────────────────────
import bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver as _solver_mod
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver import (
    BinPackingSolver,
    LinUCBBandit,
    WarmStartLinUCBBandit,
)
from bin_packing_optimization.datasets.Falkenauer.Falkenauer_T.configure import DATASET_CONFIG as FALK_T
from bin_packing_optimization.datasets.Scholl.Scholl_1.configure import DATASET_CONFIG as SCHOLL1
from bin_packing_optimization.datasets.Scholl.Scholl_2.configure import DATASET_CONFIG as SCHOLL2
from bin_packing_optimization.datasets.Scholl.Scholl_3.configure import DATASET_CONFIG as SCHOLL3


# ── Chargement du modèle GBT ─────────────────────────────────────────────────
_MODEL_PATH = Path("bin_packing_optimization/hybrid_learning_metaheuristics/hybrid_alns/models/repair_model_v2.pkl")

def _load_bundle() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(_MODEL_PATH, "rb") as f:
            return pickle.load(f)

REAL_BUNDLE = _load_bundle()


# ── Définition des méthodes ───────────────────────────────────────────────────

class _ThompsonSampling:
    """Beta-Bernoulli TS — ignore le contexte (baseline)."""
    __slots__ = ("_alpha", "_beta", "_rng")
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self._alpha = np.ones(n_arms, dtype=np.float64)
        self._beta  = np.ones(n_arms, dtype=np.float64)
        self._rng   = np.random.default_rng(999)
    def select_arm(self, context: np.ndarray) -> int:
        return int(np.argmax(self._rng.beta(self._alpha, self._beta)))
    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        if self._rng.random() < reward:
            self._alpha[arm] += 1.0
        else:
            self._beta[arm] += 1.0


class _WarmStart30:
    """TS warm-up 300 iters → LinUCB alpha=0.3 (défaut du solver)."""
    def __init__(self, n_arms: int, n_features: int, alpha: float = 0.3):
        self._b = WarmStartLinUCBBandit(n_arms=n_arms, n_features=n_features,
                                        alpha=0.3, warmup_calls=300)
    def select_arm(self, context): return self._b.select_arm(context)
    def update(self, arm, context, reward): self._b.update(arm, context, reward)


# ── Registre des méthodes ─────────────────────────────────────────────────────
METHODS: dict[str, type] = {
    "TS":         _ThompsonSampling,   # baseline (sans contexte)
    "LinUCB-0.3": LinUCBBandit,        # LinUCB contextuel alpha=0.3
    "WS-30":      _WarmStart30,        # TS 300 iters → LinUCB-0.3  ← DÉFAUT SOLVER
}


# ── Chargement des instances ──────────────────────────────────────────────────
def _load_instances(config, n: int) -> list:
    files = sorted(config.directory.glob(config.glob))[:n]
    return [config.parser(f, config.key) for f in files]


# ── Exécution d'une instance avec un bandit donné ────────────────────────────
def _run(instance, bandit_cls, max_iterations: int, seed: int = 42) -> dict:
    _solver_mod.LinUCBBandit = bandit_cls
    solver = BinPackingSolver(instance.sizes, instance.bin_capacity, seed=seed)
    t0 = time.perf_counter()
    solver.solve(model_bundle=REAL_BUNDLE, max_iterations=max_iterations)
    elapsed = time.perf_counter() - t0
    sol = solver.get_solution()
    lb  = math.ceil(sum(instance.sizes) / instance.bin_capacity)
    gap = (sol.total_bins_used - lb) / max(lb, 1) * 100
    _solver_mod.LinUCBBandit = LinUCBBandit
    return {"bins": sol.total_bins_used, "lb": lb, "gap": gap, "time": elapsed}


# ── Affichage ─────────────────────────────────────────────────────────────────
_MW = 11   # method column width
_NW = 5    # numeric column width

def _header(dataset_label: str, method_names: list[str]) -> None:
    print(f"\n{'═'*100}", flush=True)
    print(f"  Dataset : {dataset_label}", flush=True)
    print(f"{'═'*100}", flush=True)
    # Header row: Instance | n | LB | method1_bins | method1_time | method2_bins | ...
    hdr = f"  {'Instance':<22}  {'n':>5}  {'LB':>4}"
    for m in method_names:
        hdr += f"  {m+' bins':>{_MW}}  {m+' time':>{_MW}}"
    print(hdr, flush=True)
    sep = f"  {'-'*22}  {'-'*5}  {'-'*4}"
    for _ in method_names:
        sep += f"  {'-'*_MW}  {'-'*_MW}"
    print(sep, flush=True)


def _row(inst, results: dict[str, dict]) -> None:
    line = f"  {inst.name:<22}  {inst.num_items:>5}  {list(results.values())[0]['lb']:>4}"
    bins_list = [r["bins"] for r in results.values()]
    best_bins = min(bins_list)
    for name, r in results.items():
        mark = "✓" if r["bins"] == best_bins and bins_list.count(best_bins) < len(bins_list) else " "
        line += f"  {r['bins']:>{_MW}}{mark}  {r['time']:>{_MW}.3f}s"
    print(line, flush=True)


def _summary(method_names: list[str], all_results: dict[str, list[dict]]) -> None:
    n = len(list(all_results.values())[0])
    avg = lambda lst, k: sum(r[k] for r in lst) / n

    print(f"\n  {'Moyenne':<22}  {'':>5}  {'':>4}", end="", flush=True)
    for m in method_names:
        print(f"  {avg(all_results[m], 'bins'):>{_MW}.2f}   {avg(all_results[m], 'time'):>{_MW}.3f}s", end="", flush=True)
    print(flush=True)

    # Wins matrix: count how many times each method has strictly the fewest bins
    print(f"\n  Victoires (moins de bins) :", flush=True)
    wins = {m: 0 for m in method_names}
    for i in range(n):
        bins_i = {m: all_results[m][i]["bins"] for m in method_names}
        best   = min(bins_i.values())
        # Only award win if it's uniquely best
        winners = [m for m, b in bins_i.items() if b == best]
        if len(winners) == 1:
            wins[winners[0]] += 1
    for m in method_names:
        print(f"    {m:<14}: {wins[m]:>3}/{n}", flush=True)

    # Fastest on average
    avg_times = {m: avg(all_results[m], "time") for m in method_names}
    fastest = min(avg_times, key=avg_times.get)
    print(f"\n  Plus rapide en moyenne : {fastest} ({avg_times[fastest]:.3f}s)", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multi-méthodes de bandit")
    parser.add_argument("--iters",     type=int, default=500,  help="Itérations ALNS (défaut: 500)")
    parser.add_argument("--instances", type=int, default=5,    help="Instances par dataset (défaut: 5)")
    parser.add_argument("--seed",      type=int, default=42,   help="Graine aléatoire (défaut: 42)")
    args = parser.parse_args()

    method_names = list(METHODS.keys())

    DATASETS = [
        ("Falkenauer T (n=60,   20 inst)",  FALK_T,  args.instances),
        ("Scholl 1     (n=50,  120 inst)",  SCHOLL1, args.instances),
        ("Scholl 2     (n=50,  480 inst)",  SCHOLL2, args.instances),
        ("Scholl 3     (n=200,  10 inst)",  SCHOLL3, args.instances),
    ]

    print(f"\n  Méthodes : {', '.join(method_names)}")
    print(f"  Itérations ALNS : {args.iters}   |   Instances : {args.instances}   |   Seed : {args.seed}")
    print(f"  Modèle : GBT réel ({_MODEL_PATH.name})")
    total_inst = sum(min(n, len(list(c.directory.glob(c.glob)))) for _, c, n in DATASETS)
    print(f"  Total : {total_inst} instances × {len(METHODS)} méthodes = {total_inst * len(METHODS)} runs\n")

    grand_all: dict[str, list[dict]] = {m: [] for m in method_names}

    for label, config, n_inst in DATASETS:
        instances = _load_instances(config, n_inst)
        if not instances:
            print(f"\n  [!] Aucune instance pour {label}", flush=True)
            continue

        _header(label, method_names)
        dataset_results: dict[str, list[dict]] = {m: [] for m in method_names}

        for i, inst in enumerate(instances, 1):
            inst_results: dict[str, dict] = {}
            for m_name, m_cls in METHODS.items():
                print(f"  [{i}/{len(instances)}] {inst.name} — {m_name}...",
                      file=sys.stderr, end="\r", flush=True)
                inst_results[m_name] = _run(inst, m_cls, args.iters, args.seed)
            print(" " * 70, file=sys.stderr, end="\r", flush=True)
            _row(inst, inst_results)
            for m_name in method_names:
                dataset_results[m_name].append(inst_results[m_name])
                grand_all[m_name].append(inst_results[m_name])

        _summary(method_names, dataset_results)

    # Résumé global (tous datasets confondus)
    if sum(len(v) for v in grand_all.values()) > 0:
        n_total = len(grand_all[method_names[0]])
        if n_total > 0:
            print(f"\n{'═'*100}", flush=True)
            print(f"  RÉSUMÉ GLOBAL — {n_total} instances, {args.iters} itérations", flush=True)
            print(f"{'═'*100}", flush=True)
            _summary(method_names, grand_all)

    print(f"\n{'═'*80}\n", flush=True)


if __name__ == "__main__":
    main()
