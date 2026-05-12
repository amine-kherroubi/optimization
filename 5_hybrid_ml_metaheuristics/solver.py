from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    """Hybrid ML + ALNS solver for the 1D Bin Packing Problem."""

    __slots__ = ("_item_sizes", "_bin_capacity", "_final_solution", "_model", "_rng")

    def __init__(self, item_sizes: list[int], bin_capacity: int):
        if any(size > bin_capacity for size in item_sizes):
            raise ValueError("No single item size can exceed the bin capacity.")
        self._item_sizes = list(item_sizes)
        self._bin_capacity = int(bin_capacity)
        self._final_solution: BinPackingSolution | None = None
        self._model = None
        self._rng = np.random.default_rng(42)

    def solve(self, method: str, **params) -> None:
        # Keep method aliases flexible so benchmark runners can pass minor variants
        # like "hybrid-alns", "ALNS", or "alns_hybrid".
        normalized = self._normalize_method(method)
        if normalized not in {"hybrid alns", "alns hybrid", "alns"}:
            raise ValueError(
                "Unsupported method. Available methods: hybrid alns, alns hybrid, alns."
            )

        # Optional learned repair model. If omitted, the solver still runs fully
        # with deterministic BFD repair.
        model_path = params.get("model_path")
        if model_path:
            self._model = self._load_model(model_path)

        # SA defaults calibrated for long ALNS runs.
        max_iterations = int(params.get("max_iterations", 5_000))
        t0 = float(params.get("initial_temperature", 1.0 / math.log(2.0)))
        alpha_cool = float(params.get("alpha_cool", 0.9995))

        # Warm start from FFD (cheap and usually much better than random packing).
        start = self._build_ffd_start_solution()
        best = start.copy()
        current = start.copy()

        n = len(self._item_sizes)
        k_min = max(1, int(0.05 * n))
        k_max = max(k_min + 1, int(0.25 * n))
        bandit = ThompsonSamplingBandit(n_arms=3)

        temperature = t0
        for _ in range(max_iterations):
            arm = bandit.select_arm(self._rng)
            candidate = current.copy()
            k_items = int(self._rng.integers(k_min, k_max + 1))

            if arm == 0:
                displaced = self._destroy_random(candidate)
            elif arm == 1:
                displaced = self._destroy_worst(candidate)
            else:
                displaced = self._destroy_related(candidate, k_items)

            # A destroy operator can return no items in degenerate small cases;
            # in that case candidate==current and the SA acceptance handles it.
            if displaced:
                if self._model is None:
                    self._repair_bfd(candidate, displaced)
                else:
                    self._repair_learned(candidate, displaced)

            delta = candidate.cost() - current.cost()
            # Metropolis acceptance: always accept improving/tie moves, sometimes
            # accept worsening moves depending on current temperature.
            accepted = delta <= 0 or self._rng.random() < math.exp(
                -delta / max(temperature, 1e-12)
            )
            if accepted:
                current = candidate

            improved = False
            if current.cost() < best.cost():
                best = current.copy()
                improved = True

            bandit.update(arm, 1 if (accepted and improved) else 0)
            temperature *= alpha_cool

        self._final_solution = self._to_external_solution(best)

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
            placed = False
            for j, load in enumerate(sol.bin_loads):
                if load + size <= self._bin_capacity:
                    sol.bins[j].append(item)
                    sol.bin_loads[j] += size
                    sol.item_to_bin[item] = j
                    placed = True
                    break
            if not placed:
                sol.bins.append([item])
                sol.bin_loads.append(size)
                sol.item_to_bin[item] = len(sol.bins) - 1
        return sol

    def _destroy_random(self, sol: _WorkingSolution) -> list[int]:
        if len(sol.bins) <= 1:
            return []
        k_bins = int(self._rng.integers(1, max(2, len(sol.bins) // 5 + 1)))
        k_bins = min(k_bins, len(sol.bins) - 1)
        indices = self._rng.choice(len(sol.bins), size=k_bins, replace=False)
        return self._remove_bins(sol, sorted(int(i) for i in indices))

    def _destroy_worst(self, sol: _WorkingSolution) -> list[int]:
        if len(sol.bins) <= 1:
            return []
        k_bins = int(self._rng.integers(1, max(2, len(sol.bins) // 5 + 1)))
        k_bins = min(k_bins, len(sol.bins) - 1)
        order = sorted(
            range(len(sol.bins)),
            key=lambda j: sol.bin_loads[j] + float(self._rng.uniform(0.0, 1e-6)),
        )
        return self._remove_bins(sol, order[:k_bins])

    def _destroy_related(self, sol: _WorkingSolution, k_items: int) -> list[int]:
        all_items = [item for b in sol.bins for item in b]
        if len(all_items) <= 1:
            return []
        seed = int(self._rng.choice(all_items))
        candidates = [i for i in all_items if i != seed]
        candidates.sort(key=lambda i: abs(self._item_sizes[i] - self._item_sizes[seed]))
        to_remove = [seed] + candidates[: max(0, k_items - 1)]

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

    def _repair_bfd(self, sol: _WorkingSolution, displaced: list[int]) -> None:
        for item in sorted(displaced, key=lambda i: -self._item_sizes[i]):
            self._place_best_fit(sol, item)

    def _repair_learned(self, sol: _WorkingSolution, displaced: list[int]) -> None:
        # The learned repair policy scores every feasible (item, bin) pair and
        # chooses the highest-probability placement.
        n = len(self._item_sizes)
        sizes_float = [float(v) for v in self._item_sizes]
        global_order = sorted(range(n), key=lambda i: -self._item_sizes[i])
        rank = {item: idx for idx, item in enumerate(global_order)}

        remaining = len(displaced)
        for item in sorted(displaced, key=lambda i: -self._item_sizes[i]):
            remaining -= 1
            size = self._item_sizes[item]
            feats, idxs = [], []

            for j, load in enumerate(sol.bin_loads):
                capacity_left = self._bin_capacity - load
                if capacity_left + 1e-9 < size:
                    continue
                # Single shared feature builder ensures training and inference use
                # exactly the same feature count/order.
                feats.append(
                    self._make_repair_features(
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
                continue

            scores = self._model.predict_proba(np.asarray(feats, dtype=np.float64))[
                :, 1
            ]
            best_j = idxs[int(np.argmax(scores))]
            sol.bins[best_j].append(item)
            sol.bin_loads[best_j] += size
            sol.item_to_bin[item] = best_j

    def _place_best_fit(self, sol: _WorkingSolution, item: int) -> None:
        size = self._item_sizes[item]
        best_j = -1
        best_slack = float("inf")
        for j, load in enumerate(sol.bin_loads):
            remaining = self._bin_capacity - load
            if remaining + 1e-9 >= size:
                slack = remaining - size
                if slack < best_slack:
                    best_slack = slack
                    best_j = j

        if best_j == -1:
            sol.bins.append([item])
            sol.bin_loads.append(size)
            sol.item_to_bin[item] = len(sol.bins) - 1
        else:
            sol.bins[best_j].append(item)
            sol.bin_loads[best_j] += size
            sol.item_to_bin[item] = best_j

    @staticmethod
    def _make_repair_features(
        *,
        item: int,
        item_size: float,
        bin_items: list[int],
        bin_load: float,
        capacity: float,
        sizes: list[float],
        n_total: int,
        size_rank: dict[int, int],
        remaining_ratio: float,
    ) -> list[float]:
        """Feature contract for learned repair (must match training script)."""
        remaining_capacity = capacity - bin_load
        slack_after = remaining_capacity - item_size
        largest = max((sizes[k] for k in bin_items), default=0.0)
        smallest = min((sizes[k] for k in bin_items), default=0.0)
        return [
            item_size / capacity,
            (item_size / capacity) ** 2,
            item_size / capacity,
            size_rank.get(item, 0) / max(1, n_total),
            remaining_ratio,
            bin_load / capacity,
            remaining_capacity / capacity,
            slack_after / capacity,
            slack_after / capacity,
            bin_load / capacity,
            len(bin_items) / max(1, n_total),
            largest / capacity,
            smallest / capacity,
            (item_size / remaining_capacity) if remaining_capacity > 1e-9 else 1.0,
        ]

    @staticmethod
    def _normalize_method(method: str) -> str:
        normalized = method.strip().lower().replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    def _to_external_solution(self, sol: _WorkingSolution) -> BinPackingSolution:
        return BinPackingSolution(
            total_bins_used=len(sol.bins),
            bin_assignments={i: list(items) for i, items in enumerate(sol.bins)},
            final_bin_loads=list(sol.bin_loads),
        )

    @staticmethod
    def _load_model(model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with path.open("rb") as handle:
            return pickle.load(handle)
