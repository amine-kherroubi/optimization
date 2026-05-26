"""Hybrid ALNS solver for 1D Bin Packing.

This module provides a single, deterministic implementation path:
- destroy/repair ALNS with simulated annealing acceptance
- LinUCB contextual bandit to choose destroy operators (Chu et al., ICML 2011)
- mandatory learned repair model for reconstruction
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.features import (
    FEATURE_VERSION as _EXPECTED_FEATURE_VERSION,
)
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.features import (
    N_FEATURES as _EXPECTED_N_FEATURES,
)
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.features import (
    make_features as _make_features,
)
import bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.repair_model_training.features as _features


@dataclass(slots=True)
class BinPackingSolution:
    """Compact representation of a bin packing solution."""

    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[float]


class _WorkingSolution:
    """Mutable internal solution used by ALNS with O(1) item-to-bin lookups."""

    __slots__ = ("sizes", "capacity", "bins", "bin_loads", "item_to_bin")

    def __init__(self, sizes: Sequence[float], capacity: int):
        self.sizes = list(sizes)
        self.capacity = capacity
        self.bins: list[list[int]] = []
        self.bin_loads: list[int] = []
        self.item_to_bin: list[int] = [-1] * len(sizes)

    def copy(self) -> _WorkingSolution:
        new = _WorkingSolution(self.sizes, self.capacity)
        new.bins = [list(b) for b in self.bins]
        new.bin_loads = list(self.bin_loads)
        new.item_to_bin = list(self.item_to_bin)
        return new

    def cost(self) -> int:
        return len(self.bins)

    def rebuild_item_to_bin(self) -> None:
        self.item_to_bin = [-1] * len(self.sizes)
        for j, items in enumerate(self.bins):
            for item in items:
                self.item_to_bin[item] = j


N_CONTEXT_FEATURES = 5  # [T/T0, stagnation/limit, cost/LB, k/n, iter/max_iter]


class _FastPredictor:
    """Thin wrapper around GBT + StandardScaler that bypasses sklearn's
    per-call input validation (validate_data), which accounts for ~33% of
    total solve time in profiling.

    Applies scaling manually (pure numpy) then calls model._raw_predict
    directly, converting logits to probabilities via sigmoid.
    Only works with GradientBoostingClassifier + StandardScaler.
    Falls back to the standard path for any other model type.
    """

    __slots__ = ("_model", "_mean", "_scale", "_fast")

    def __init__(self, model: Any, scaler: Any) -> None:
        self._model = model
        self._fast = False
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler

            if isinstance(model, GradientBoostingClassifier) and isinstance(
                scaler, StandardScaler
            ):
                self._mean = scaler.mean_.astype(np.float64)
                self._scale = scaler.scale_.astype(np.float64)
                self._fast = True
        except Exception:
            pass
        if not self._fast:
            self._mean = None
            self._scale = None

    def predict_proba_col1(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for each row of X — fast path or safe fallback."""
        if self._fast:
            X_scaled = ((X - self._mean) / self._scale).astype(
                np.float32
            )  # GBT needs float32
            raw = self._model._raw_predict(X_scaled)  # shape (n, 1)
            return 1.0 / (1.0 + np.exp(-raw[:, 0]))  # sigmoid → P(1)
        # fallback: standard sklearn path
        scaler = getattr(self, "_scaler_fallback", None)
        if scaler is not None:
            X = scaler.transform(X)
        return self._model.predict_proba(X)[:, 1]


class LinUCBBandit:
    """Disjoint LinUCB contextual bandit for destroy-operator selection.

    Each arm k maintains A_k^{-1} (cached inverse) and b_k (reward vector).
    Selection: argmax_k [ θ_k^T x + α √(x^T A_k^{-1} x) ]
    Update:    A_k^{-1} via Sherman-Morrison (O(d²) instead of O(d³))
               b_k += r * x

    Reference: Chu et al., ICML 2011.
    Sherman-Morrison: (A + uv^T)^{-1} = A^{-1} - (A^{-1}u v^T A^{-1}) / (1 + v^T A^{-1} u)
    """

    __slots__ = ("alpha", "_A_inv", "_b", "_n_arms")

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.alpha = float(alpha)
        self._n_arms = n_arms
        # Store A_inv directly — identity matrix is its own inverse
        self._A_inv: list[np.ndarray] = [np.eye(n_features) for _ in range(n_arms)]
        self._b: list[np.ndarray] = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """Return arm with highest UCB score for the given context vector."""
        scores = np.empty(self._n_arms)
        for k in range(self._n_arms):
            A_inv = self._A_inv[k]
            theta = A_inv @ self._b[k]
            scores[k] = theta @ context + self.alpha * np.sqrt(
                context @ A_inv @ context
            )
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update arm k using Sherman-Morrison rank-1 inverse update (O(d²))."""
        A_inv = self._A_inv[arm]
        Ax = A_inv @ context  # d-vector, O(d²)
        denom = 1.0 + context @ Ax  # scalar
        self._A_inv[arm] = A_inv - np.outer(Ax, Ax) / denom  # rank-1 update
        self._b[arm] += reward * context


# Private reference to the real LinUCBBandit — immune to monkey-patching in tests/benchmarks.
_LinUCBBanditImpl = LinUCBBandit


class WarmStartLinUCBBandit:
    """TS warm-up phase followed by LinUCB contextual exploitation.

    For the first `warmup_calls` iterations, uses Beta-Bernoulli Thompson
    Sampling (fast, context-free exploration).  After that, switches to the
    full LinUCB bandit which exploits the context vector.

    This avoids the cold-start problem of pure LinUCB on short runs: TS
    quickly finds good arms, then LinUCB refines choices using context.
    """

    __slots__ = ("_linucb", "_ts_alpha", "_ts_beta", "_rng", "_calls", "_warmup_calls")

    def __init__(
        self, n_arms: int, n_features: int, alpha: float = 0.3, warmup_calls: int = 200
    ):
        # Use _LinUCBBanditImpl (private, not monkey-patchable) to avoid infinite
        # recursion when the benchmark replaces the module-level LinUCBBandit name.
        self._linucb = _LinUCBBanditImpl(
            n_arms=n_arms, n_features=n_features, alpha=alpha
        )
        self._ts_alpha = np.ones(n_arms, dtype=np.float64)
        self._ts_beta = np.ones(n_arms, dtype=np.float64)
        self._rng = np.random.default_rng(42)
        self._calls = 0
        self._warmup_calls = int(warmup_calls)

    def select_arm(self, context: np.ndarray) -> int:
        if self._calls < self._warmup_calls:
            return int(np.argmax(self._rng.beta(self._ts_alpha, self._ts_beta)))
        return self._linucb.select_arm(context)

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        if self._calls < self._warmup_calls:
            if self._rng.random() < reward:
                self._ts_alpha[arm] += 1.0
            else:
                self._ts_beta[arm] += 1.0
        else:
            self._linucb.update(arm, context, reward)
        self._calls += 1


class BinPackingSolver:
    """Single-path hybrid ALNS solver: LinUCB contextual bandit + learned repair."""

    __slots__ = (
        "_item_sizes",
        "_bin_capacity",
        "_final_solution",
        "_model",
        "_scaler",
        "_predictor",
        "_rng",
    )

    def __init__(self, item_sizes: list[int], bin_capacity: int, seed: int | None = 42):
        if bin_capacity <= 0:
            raise ValueError("bin_capacity must be a positive integer.")
        if any(size <= 0 for size in item_sizes):
            raise ValueError("All item sizes must be strictly positive.")
        if any(size > bin_capacity for size in item_sizes):
            raise ValueError("No single item size can exceed the bin capacity.")
        self._item_sizes = list(item_sizes)
        self._bin_capacity = int(bin_capacity)
        self._final_solution: BinPackingSolution | None = None
        self._model: Any = None
        self._scaler: Any = None
        self._predictor: _FastPredictor | None = None
        self._rng = np.random.default_rng(seed)

    def solve(self, method: str | None = None, **params) -> None:
        if method is not None:
            warnings.warn(
                f"method={method!r} is ignored; BinPackingSolver runs a single ALNS pipeline.",
                stacklevel=2,
            )

        model_bundle = params.get("model_bundle")
        model = params.get("model")
        scaler = params.get("scaler")
        if model_bundle is not None:
            self._model, self._scaler = self._validate_model_bundle(model_bundle)
        elif model is not None and scaler is not None:
            self._model, self._scaler = self._validate_model_components(model, scaler)
        else:
            raise ValueError(
                "Provide model_bundle or both model and scaler objects. "
                "Path-based model loading is no longer supported."
            )
        self._predictor = _FastPredictor(self._model, self._scaler)

        max_iterations = int(params.get("max_iterations", 5_000))
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        t0 = float(params.get("initial_temperature", 1.0 / math.log(2.0)))
        if t0 <= 0.0:
            raise ValueError("initial_temperature must be > 0.")
        alpha_cool = float(params.get("alpha_cool", 0.9995))
        if not 0.0 < alpha_cool <= 1.0:
            raise ValueError("alpha_cool must be in (0, 1].")

        raw_tl = params.get("time_limit_seconds")
        deadline: float | None = (
            time.perf_counter() + float(raw_tl) if raw_tl is not None else None
        )

        start = self._build_ffd_start_solution()
        best = start.copy()
        current = start.copy()

        n = len(self._item_sizes)
        k_min = max(1, int(0.05 * n))
        k_max = max(k_min + 1, int(0.25 * n))
        lower_bound = math.ceil(sum(self._item_sizes) / self._bin_capacity)
        bandit = WarmStartLinUCBBandit(
            n_arms=3, n_features=N_CONTEXT_FEATURES, alpha=0.3, warmup_calls=300
        )

        temperature = t0
        no_improve_limit = max(250, max_iterations // 20)
        iterations_since_improvement = 0
        for iteration in range(max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            _ctx = self._build_context(
                temperature,
                t0,
                iterations_since_improvement,
                no_improve_limit,
                current.cost(),
                lower_bound,
                k_min,
                n,
                iteration,
                max_iterations,
            )
            arm = bandit.select_arm(_ctx)
            candidate = current.copy()

            # Adaptive k: grow the destruction radius when stagnating.
            # stagnation_ratio goes from 0 → 1 as iterations_since_improvement
            # increases, linearly expanding k toward k_max.
            # Denominator is no_improve_limit (the stagnation budget), NOT
            # max_iterations — using max_iterations kept the ratio near 0 for
            # the entire run, making adaptive-k effectively a no-op.
            stagnation_ratio = min(
                1.0, iterations_since_improvement / max(1, no_improve_limit)
            )
            k_adaptive_max = k_min + int(stagnation_ratio * (k_max - k_min))
            k_items = int(self._rng.integers(k_min, max(k_min + 1, k_adaptive_max + 1)))

            if arm == 0:
                displaced = self._destroy_random(candidate, k_items)
            elif arm == 1:
                displaced = self._destroy_worst(candidate, k_items)
            else:
                displaced = self._destroy_related(candidate, k_items)

            if displaced:
                self._repair_learned(candidate, displaced)
                self._consolidate(candidate)

            delta = candidate.cost() - current.cost()
            accepted = delta <= 0 or self._rng.random() < math.exp(
                -delta / max(temperature, 1e-12)
            )
            if accepted:
                current = candidate

            improved = False
            if current.cost() < best.cost():
                best = current.copy()
                improved = True
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1

            # LinUCB reward: improvement normalised by gap to lower bound.
            # Saving bins near the optimum (gap small) earns more than saving
            # the same number when far away (gap large). Accepted-without-saving
            # earns a small signal (0.2) to keep exploration; rejected earns 0.
            _reward = self._linucb_reward(
                delta, accepted, improved, current.cost(), lower_bound
            )
            bandit.update(arm, _ctx, _reward)
            # Cool first, then conditionally reheat — wrong order caused each
            # reheat to be immediately eroded by the multiplication that followed.
            temperature *= alpha_cool
            if iterations_since_improvement > 0 and (
                iterations_since_improvement % max(50, no_improve_limit // 4) == 0
            ):
                # Soft reheat tied to no_improve_limit (not max_iterations) so
                # the interval scales with the actual stagnation budget.
                temperature = max(temperature, t0 * 0.35)
            if iterations_since_improvement >= no_improve_limit:
                # Diversification restart: return to best, reheat partially, and
                # shrink the patience window so successive restarts get shorter.
                # The outer range(max_iterations) guarantees termination.
                current = best.copy()
                iterations_since_improvement = 0
                temperature = max(temperature, t0 * 0.20)
                no_improve_limit = max(100, no_improve_limit * 2 // 3)

        self._final_solution = self._to_presentable_solution(best)

    def get_solution(self) -> BinPackingSolution:
        if self._final_solution is None:
            raise RuntimeError("No solution available. Call solve() first.")
        return self._final_solution

    def _build_ffd_start_solution(self) -> _WorkingSolution:
        sol = _WorkingSolution(self._item_sizes, self._bin_capacity)
        order = sorted(
            range(len(self._item_sizes)), key=lambda i: (-self._item_sizes[i], i)
        )
        # Best-Fit Decreasing (BFD): choose the feasible bin with minimum slack.
        # This provides a stronger warm start for ALNS than looser fit policies.
        for item in order:
            size = self._item_sizes[item]
            best_j = -1
            best_slack = self._bin_capacity + 1
            for j, load in enumerate(sol.bin_loads):
                slack = self._bin_capacity - load - size
                if slack < 0:
                    continue
                if slack < best_slack:
                    best_slack = slack
                    best_j = j
            if best_j >= 0:
                j = best_j
                sol.bins[j].append(item)
                sol.bin_loads[j] += size
                sol.item_to_bin[item] = j
            else:
                j = len(sol.bins)
                sol.bins.append([item])
                sol.bin_loads.append(size)
                sol.item_to_bin[item] = j
        return sol

    def _destroy_random(self, sol: _WorkingSolution, k_items: int) -> list[int]:
        all_items = [item for b in sol.bins for item in b]
        if len(all_items) <= 1:
            return []
        # Item-level selection: sample exactly min(k_items, n-1) items at random.
        # Previously this removed whole bins, which could displace 6–30× k_items
        # when a randomly chosen bin happened to be large.
        k_effective = min(k_items, len(all_items) - 1)
        to_remove = [
            int(x) for x in self._rng.choice(all_items, size=k_effective, replace=False)
        ]
        return self._remove_items(sol, to_remove)

    def _destroy_worst(self, sol: _WorkingSolution, k_items: int) -> list[int]:
        all_items = [item for b in sol.bins for item in b]
        if len(all_items) <= 1:
            return []
        # Item-level selection: collect items from least-loaded bins first,
        # stopping once we reach exactly k_items items.  The tiny random nudge
        # on bin load breaks ties without changing the overall ranking.
        order = sorted(
            range(len(sol.bins)),
            key=lambda j: sol.bin_loads[j] + float(self._rng.uniform(0.0, 1e-6)),
        )
        candidates: list[int] = []
        for j in order:
            candidates.extend(sol.bins[j])
            if len(candidates) >= k_items:
                break
        k_effective = min(k_items, len(all_items) - 1)
        to_remove = candidates[:k_effective]
        return self._remove_items(sol, to_remove)

    def _destroy_related(self, sol: _WorkingSolution, k_items: int) -> list[int]:
        all_items = [item for b in sol.bins for item in b]
        if len(all_items) <= 1:
            return []
        seed = int(self._rng.choice(all_items))
        candidates = [i for i in all_items if i != seed]
        candidates.sort(key=lambda i: abs(self._item_sizes[i] - self._item_sizes[seed]))
        # Cap displacement so at least one item stays placed.  Without this cap,
        # k_items ≥ len(all_items) would evict every item, reducing the warm-
        # started solution to an empty state on every related-destroy call.
        k_effective = min(k_items, len(all_items) - 1)
        to_remove = [seed] + candidates[: max(0, k_effective - 1)]
        return self._remove_items(sol, to_remove)

    def _remove_items(self, sol: _WorkingSolution, to_remove: list[int]) -> list[int]:
        """Remove specific items from their bins (swap-with-last O(1)), prune
        empty bins, rebuild item_to_bin.  Shared by all three destroy operators."""
        for item in to_remove:
            j = sol.item_to_bin[item]
            if j == -1:
                continue
            lst = sol.bins[j]
            idx = lst.index(item)
            lst[idx] = lst[-1]
            lst.pop()
            sol.bin_loads[j] -= self._item_sizes[item]
            sol.item_to_bin[item] = -1
        empty = [j for j, b in enumerate(sol.bins) if not b]
        for j in reversed(empty):
            sol.bins.pop(j)
            sol.bin_loads.pop(j)
        sol.rebuild_item_to_bin()
        return to_remove

    def _repair_learned(self, sol: _WorkingSolution, displaced: list[int]) -> None:
        n = len(self._item_sizes)
        sizes_float = [float(v) for v in self._item_sizes]
        global_order = sorted(range(n), key=lambda i: -self._item_sizes[i])
        rank = {item: idx for idx, item in enumerate(global_order)}
        mf = _make_features
        model = self._model
        scaler = self._scaler
        predictor = self._predictor
        denom = max(1, n)
        cap_float = float(self._bin_capacity)
        eps = 1e-9

        # Precompute arrays used by the batch feature API
        sizes_arr = np.array(sizes_float, dtype=np.float64)
        size_ranks_arr = np.array([rank[i] for i in range(n)], dtype=np.int64)

        use_batch = getattr(_features, "njit", None) is not None and hasattr(
            _features, "make_features_batch_jit"
        )

        remaining = len(displaced)
        for item in sorted(displaced, key=lambda i: -self._item_sizes[i]):
            size = self._item_sizes[item]
            size_float = float(size)

            items_all: list[int] = []
            item_sizes_all: list[float] = []
            bin_items_flat: list[int] = []
            offsets: list[int] = [0]
            bin_loads_all: list[float] = []
            idxs: list[int] = []

            for j, load in enumerate(sol.bin_loads):
                capacity_left = self._bin_capacity - load
                if capacity_left + eps < size:
                    continue
                items_all.append(item)
                item_sizes_all.append(size_float)
                for k in sol.bins[j]:
                    bin_items_flat.append(k)
                offsets.append(len(bin_items_flat))
                bin_loads_all.append(float(load))
                idxs.append(j)

            if not items_all:
                sol.bins.append([item])
                sol.bin_loads.append(size)
                sol.item_to_bin[item] = len(sol.bins) - 1
                remaining -= 1
                continue

            assert model is not None

            items_arr = np.array(items_all, dtype=np.int64)
            item_sizes_arr = np.array(item_sizes_all, dtype=np.float64)
            bin_items_flat_arr = np.array(bin_items_flat, dtype=np.int64)
            bin_offsets_arr = np.array(offsets, dtype=np.int64)
            bin_loads_arr = np.array(bin_loads_all, dtype=np.float64)
            remaining_ratio_arr = np.full(
                items_arr.shape[0], remaining / denom, dtype=np.float64
            )

            if use_batch:
                feats_arr = _features.make_features_batch_jit(
                    items_arr,
                    item_sizes_arr,
                    bin_items_flat_arr,
                    bin_offsets_arr,
                    bin_loads_arr,
                    cap_float,
                    sizes_arr,
                    n,
                    size_ranks_arr,
                    remaining_ratio_arr,
                )
            else:
                feats_arr = _features.make_features_batch_py(
                    items_arr,
                    item_sizes_arr,
                    bin_items_flat_arr,
                    bin_offsets_arr,
                    bin_loads_arr,
                    cap_float,
                    sizes_arr,
                    n,
                    size_ranks_arr,
                    remaining_ratio_arr,
                )

            scores = predictor.predict_proba_col1(feats_arr)
            best_idx = int(scores.argmax())
            best_j = idxs[best_idx]
            sol.bins[best_j].append(item)
            sol.bin_loads[best_j] += size
            sol.item_to_bin[item] = best_j
            remaining -= 1

    def _consolidate(self, sol: _WorkingSolution) -> None:
        """Post-repair local search: merge under-loaded bins into others.

        Iterates bins from least-loaded to most-loaded. For each item in a
        lightly-loaded bin, tries to move it to another bin that has room.
        Empty bins are pruned at the end. A single rebuild_item_to_bin() call
        is deferred until after all moves to avoid redundant O(n) scans.

        The sorted order is computed once before the outer loop and only
        recomputed after a pass that made at least one move — avoids an
        unnecessary O(B log B) sort when the first pass makes no moves.
        """
        changed = False
        order = sorted(range(len(sol.bins)), key=lambda j: sol.bin_loads[j])
        for _ in range(len(sol.bins)):
            moved_any = False
            for src in order:
                if not sol.bins[src]:
                    continue
                for item in list(sol.bins[src]):
                    size = self._item_sizes[item]
                    for dst, dst_load in enumerate(sol.bin_loads):
                        if dst == src:
                            continue
                        if dst_load + size <= self._bin_capacity:
                            # Move item from src to dst
                            lst = sol.bins[src]
                            idx = lst.index(item)
                            lst[idx] = lst[-1]
                            lst.pop()
                            sol.bin_loads[src] -= size
                            sol.bins[dst].append(item)
                            sol.bin_loads[dst] += size
                            changed = True
                            moved_any = True
                            break
            if not moved_any:
                break
            # Re-sort only when loads changed — skip unnecessary work on the
            # final pass where moved_any was False (caught by the break above).
            order = sorted(range(len(sol.bins)), key=lambda j: sol.bin_loads[j])

        if changed:
            empty = [j for j, b in enumerate(sol.bins) if not b]
            for j in reversed(empty):
                sol.bins.pop(j)
                sol.bin_loads.pop(j)
            sol.rebuild_item_to_bin()

    def _to_presentable_solution(self, sol: _WorkingSolution) -> BinPackingSolution:
        return BinPackingSolution(
            total_bins_used=len(sol.bins),
            bin_assignments={i: list(items) for i, items in enumerate(sol.bins)},
            final_bin_loads=list(sol.bin_loads),
        )

    @staticmethod
    def _build_context(
        temperature: float,
        t0: float,
        iterations_since_improvement: int,
        no_improve_limit: int,
        current_cost: int,
        lower_bound: int,
        k_items: int,
        n: int,
        iteration: int,
        max_iterations: int,
    ) -> np.ndarray:
        """Build the 5-D context vector for LinUCB arm selection.

        Features (all normalised to comparable ranges):
          0  T / T0                        — SA phase (1→0, hot→cold)
                                             Hot = exploration phase, cold = exploitation
          1  iter_no_improve / limit       — stagnation (0→1)
                                             Signals when diversification is needed
          2  current_cost / LB            — gap to optimum (≥1)
                                             1.0 = optimal; >1 = room for improvement
          3  k_items / n                  — destruction radius (0.05–0.25)
                                             Captures the current neighbourhood size
          4  iteration / max_iterations   — search progress (0→1)
                                             Early vs late stage of the search
        """
        return np.array(
            [
                temperature / max(t0, 1e-12),
                iterations_since_improvement / max(no_improve_limit, 1),
                current_cost / max(lower_bound, 1),
                k_items / max(n, 1),
                iteration / max(max_iterations, 1),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _linucb_reward(
        delta: int,
        accepted: bool,
        improved: bool,
        current_cost: int,
        lower_bound: int,
    ) -> float:
        """Normalised reward in [0, 1] for LinUCB update.

        Improvement is divided by the gap to LB so that gains near the
        optimum are rewarded more than the same gain when far away.

          r = min(1, bins_saved / gap)   if bins were saved (improved=True)
          r = 0.2                        if accepted without saving bins
          r = 0.0                        if rejected

        Note: credit assignment — reward is attributed to the destroy operator
        but the GBT repair model also contributes. This is an accepted
        limitation in hybrid ALNS + RL literature.
        """
        gap = max(1, current_cost - lower_bound)
        bins_saved = max(0, -delta)
        if bins_saved > 0:
            return min(1.0, bins_saved / gap)
        return 0.2 if accepted else 0.0

    @staticmethod
    def _validate_model_components(model: Any, scaler: Any) -> tuple[Any, Any]:
        if not hasattr(model, "predict_proba"):
            raise TypeError("Loaded model must expose predict_proba(features).")
        return model, scaler

    @staticmethod
    def _validate_model_bundle(bundle: Any) -> tuple[Any, Any]:
        """Validate an in-memory model bundle produced by train_repair_model.py."""
        if not isinstance(bundle, dict):
            raise TypeError(
                f"Expected a model bundle dict, got {type(bundle).__name__}. "
                "Re-train with the current train_repair_model.py."
            )

        required_keys = {"model", "scaler", "feature_version", "n_features"}
        missing = required_keys - bundle.keys()
        if missing:
            raise KeyError(
                f"Model bundle is missing keys: {missing}. "
                "Re-train with the current train_repair_model.py."
            )

        version = bundle["feature_version"]
        if version != _EXPECTED_FEATURE_VERSION:
            raise ValueError(
                f"Model feature_version={version} does not match solver's "
                f"expected version={_EXPECTED_FEATURE_VERSION}. "
                "Re-train with the current train_repair_model.py."
            )

        n_features = bundle["n_features"]
        if n_features != _EXPECTED_N_FEATURES:
            raise ValueError(
                f"Model n_features={n_features} does not match solver's "
                f"expected n_features={_EXPECTED_N_FEATURES}. "
                "Re-train with the current train_repair_model.py."
            )

        model = bundle["model"]
        scaler = bundle["scaler"]
        return BinPackingSolver._validate_model_components(model, scaler)
