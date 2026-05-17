"""Hybrid ALNS solver for 1D Bin Packing.

This module provides a single, deterministic implementation path:
- destroy/repair ALNS with simulated annealing acceptance
- Thompson Sampling to choose destroy operators
- mandatory learned repair model for reconstruction
"""

from __future__ import annotations

import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure features.py (sibling module) is importable regardless of how this
# file is loaded (direct script, importlib from benchmark.py, notebook, etc.).
_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from features import FEATURE_VERSION as _EXPECTED_FEATURE_VERSION  # noqa: E402
from features import N_FEATURES as _EXPECTED_N_FEATURES  # noqa: E402
from features import make_features as _make_features  # noqa: E402


@dataclass(slots=True)
class BinPackingSolution:
    """Compact representation of a bin packing solution."""

    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[int]


class _WorkingSolution:
    """Mutable internal solution used by ALNS with O(1) item-to-bin lookups."""

    __slots__ = ("sizes", "capacity", "bins", "bin_loads", "item_to_bin")

    def __init__(self, sizes: list[int], capacity: int):
        self.sizes = sizes
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


class ThompsonSamplingBandit:
    """Beta-Bernoulli Thompson Sampling for destroy-operator selection."""

    __slots__ = ("alpha", "beta")

    def __init__(self, n_arms: int):
        self.alpha = np.ones(n_arms, dtype=np.float64)
        self.beta = np.ones(n_arms, dtype=np.float64)

    def select_arm(self, rng: np.random.Generator) -> int:
        return int(np.argmax(rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: int) -> None:
        if reward == 1:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0


class BinPackingSolver:
    """Single-path hybrid ALNS solver: Thompson bandit + learned repair."""

    __slots__ = (
        "_item_sizes",
        "_bin_capacity",
        "_final_solution",
        "_model",
        "_scaler",
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
        self._rng = np.random.default_rng(seed)

    def solve(self, method: str | None = None, **params) -> None:
        _ = method

        model_path = params.get("model_path")
        if not model_path:
            raise ValueError(
                "model_path is required for this approach. "
                "Train it first with train_repair_model.py."
            )
        self._model, self._scaler = self._load_model(str(model_path))

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
        bandit = ThompsonSamplingBandit(n_arms=3)

        temperature = t0
        iterations_since_improvement = 0
        for iteration in range(max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            arm = bandit.select_arm(self._rng)
            candidate = current.copy()

            # Adaptive k: grow the destruction radius when stagnating.
            # stagnation_ratio goes from 0 → 1 as iterations_since_improvement
            # increases, linearly expanding k toward k_max.
            stagnation_ratio = min(1.0, iterations_since_improvement / max(1, max_iterations))
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

            # 3-level reward: new best (1.0) > accepted (0.5) > rejected (0.0).
            # Thompson Sampling expects binary feedback, so we convert the
            # continuous reward probabilistically: update with 1 if a uniform
            # draw falls below the reward, 0 otherwise. This preserves the
            # expected value while keeping the Beta posterior well-calibrated.
            reward = 1.0 if improved else (0.5 if accepted else 0.0)
            bandit.update(arm, 1 if self._rng.random() < reward else 0)
            temperature *= alpha_cool

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
        for item in order:
            size = self._item_sizes[item]
            for j, load in enumerate(sol.bin_loads):
                if load + size <= self._bin_capacity:
                    sol.bins[j].append(item)
                    sol.bin_loads[j] += size
                    sol.item_to_bin[item] = j
                    break
            else:
                sol.bins.append([item])
                sol.bin_loads.append(size)
                sol.item_to_bin[item] = len(sol.bins) - 1
        return sol

    def _destroy_random(self, sol: _WorkingSolution, k_items: int) -> list[int]:
        if len(sol.bins) <= 1:
            return []
        # Select bins in random order until cumulative displaced items reach k_items.
        bin_order = self._rng.permutation(len(sol.bins)).tolist()
        selected: list[int] = []
        removed_count = 0
        for j in bin_order:
            if removed_count >= k_items:
                break
            if len(selected) >= len(sol.bins) - 1:
                break
            selected.append(j)
            removed_count += len(sol.bins[j])
        # Defensive fallback: unreachable in practice because the len(sol.bins) <= 1
        # guard above ensures at least 2 bins exist, so the first loop iteration
        # always appends before either break condition can fire.
        if not selected:
            selected = [int(bin_order[0])]
        return self._remove_bins(sol, selected)

    def _destroy_worst(self, sol: _WorkingSolution, k_items: int) -> list[int]:
        if len(sol.bins) <= 1:
            return []
        # Greedily pick least-loaded bins until cumulative displaced items reach k_items.
        order = sorted(
            range(len(sol.bins)),
            key=lambda j: sol.bin_loads[j] + float(self._rng.uniform(0.0, 1e-6)),
        )
        selected: list[int] = []
        removed_count = 0
        for j in order:
            if removed_count >= k_items:
                break
            if len(selected) >= len(sol.bins) - 1:
                break
            selected.append(j)
            removed_count += len(sol.bins[j])
        # Defensive fallback: same reasoning as _destroy_random — unreachable in practice.
        if not selected:
            selected = [order[0]]
        return self._remove_bins(sol, selected)

    def _destroy_related(self, sol: _WorkingSolution, k_items: int) -> list[int]:
        all_items = [item for b in sol.bins for item in b]
        if len(all_items) <= 1:
            return []
        seed = int(self._rng.choice(all_items))
        candidates = [i for i in all_items if i != seed]
        candidates.sort(key=lambda i: abs(self._item_sizes[i] - self._item_sizes[seed]))
        # Cap displacement so at least one item (and therefore one bin) stays
        # placed, matching the ≥1-bin-preserved guarantee enforced by
        # _destroy_random and _destroy_worst. Without this cap, k_items ≥
        # len(all_items) would evict every item from every bin, reducing the
        # warm-started solution to an empty state and forcing a restart from
        # scratch on every related-destroy call — defeating the purpose of ALNS.
        k_effective = min(k_items, len(all_items) - 1)
        to_remove = [seed] + candidates[: max(0, k_effective - 1)]

        for item in to_remove:
            j = sol.item_to_bin[item]
            if j == -1:
                continue
            sol.bins[j].remove(item)
            sol.bin_loads[j] -= self._item_sizes[item]
            sol.item_to_bin[item] = -1

        empty = [j for j, b in enumerate(sol.bins) if not b]
        for j in reversed(empty):
            sol.bins.pop(j)
            sol.bin_loads.pop(j)
        sol.rebuild_item_to_bin()
        return to_remove

    def _remove_bins(self, sol: _WorkingSolution, bin_indices: list[int]) -> list[int]:
        displaced: list[int] = []
        for j in sorted(bin_indices, reverse=True):
            displaced.extend(sol.bins[j])
            sol.bins.pop(j)
            sol.bin_loads.pop(j)
        for item in displaced:
            sol.item_to_bin[item] = -1
        sol.rebuild_item_to_bin()
        return displaced

    def _repair_learned(self, sol: _WorkingSolution, displaced: list[int]) -> None:
        n = len(self._item_sizes)
        sizes_float = [float(v) for v in self._item_sizes]
        global_order = sorted(range(n), key=lambda i: -self._item_sizes[i])
        rank = {item: idx for idx, item in enumerate(global_order)}

        # remaining_ratio = (items_in_batch_still_to_place) / n_total.
        # This matches the convention in extract_repair_examples:
        #   (n_displaced - step) / n_total
        # Fresh BFD traces in training set n_displaced = n_total (full
        # destruction), so their ratio of 1.0 at step 0 is the special case
        # of this formula when all items are displaced — not a mismatch.
        remaining = len(displaced)
        for item in sorted(displaced, key=lambda i: -self._item_sizes[i]):
            size = self._item_sizes[item]
            feats: list[list[float]] = []
            idxs: list[int] = []

            for j, load in enumerate(sol.bin_loads):
                capacity_left = self._bin_capacity - load
                if capacity_left + 1e-9 < size:
                    continue
                feats.append(
                    _make_features(
                        item=item,
                        item_size=float(size),
                        bin_items=sol.bins[j],
                        bin_load=float(load),
                        capacity=float(self._bin_capacity),
                        sizes=sizes_float,
                        n_total=n,
                        size_rank=rank,
                        remaining_ratio=remaining / max(1, n),
                    )
                )
                idxs.append(j)

            if not feats:
                sol.bins.append([item])
                sol.bin_loads.append(size)
                sol.item_to_bin[item] = len(sol.bins) - 1
                remaining -= 1
                continue

            assert self._model is not None
            feats_arr = np.asarray(feats, dtype=np.float64)

            # Apply the same scaling that was fit during training. The scaler
            # is stored in the model bundle and must be applied before
            # predict_proba to keep the inference distribution consistent with
            # the training distribution.
            if self._scaler is not None:
                feats_arr = self._scaler.transform(feats_arr)

            scores = self._model.predict_proba(feats_arr)[:, 1]
            best_j = idxs[int(np.argmax(scores))]
            sol.bins[best_j].append(item)
            sol.bin_loads[best_j] += size
            sol.item_to_bin[item] = best_j
            remaining -= 1

    def _to_presentable_solution(self, sol: _WorkingSolution) -> BinPackingSolution:
        return BinPackingSolution(
            total_bins_used=len(sol.bins),
            bin_assignments={i: list(items) for i, items in enumerate(sol.bins)},
            final_bin_loads=list(sol.bin_loads),
        )

    @staticmethod
    def _load_model(model_path: str) -> tuple[Any, Any]:
        """Load and validate the model bundle produced by train_repair_model.py.

        The bundle is a dict containing at minimum:
          - "model":           fitted sklearn estimator with predict_proba
          - "scaler":          fitted StandardScaler (applied before predict_proba)
          - "feature_version": int, must match _EXPECTED_FEATURE_VERSION
          - "n_features":      int, sanity-checked against _make_repair_features output

        Returns (model, scaler). Raises if the file is missing, malformed, or
        was produced by an incompatible version of the training script.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with path.open("rb") as handle:
            bundle = pickle.load(handle)

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
        if not hasattr(model, "predict_proba"):
            raise TypeError("Loaded model must expose predict_proba(features).")

        return model, bundle["scaler"]
