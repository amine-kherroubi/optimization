from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import heapq
from typing import Optional


@dataclass(slots=True)
class Solution:
    bins_used: int
    assignments: dict[int, set[int]]
    loads: list[int]  # loads[b] = load in bin b


@dataclass(slots=True)
class State:
    # Core search state
    bins_used: int
    loads: list[int]  # length == bins_used
    next_item: int  # assign item indices [0..next_item-1] already
    used_volume: int  # sum(loads)

    # Parent pointer for reconstruction (no per-node item_bins copying)
    parent: Optional["State"]
    placed_bin: int  # bin index where item (next_item-1) was placed; -1 for root


class BinPacking:
    __slots__ = (
        "_sizes",
        "_number_of_items",
        "_bin_capacity",
        "_suffix_sum",
        "_solution",
    )

    def __init__(self, sizes: list[int], bin_capacity: int) -> None:
        if any(s > bin_capacity for s in sizes):
            raise ValueError("All item sizes must be <= bin capacity.")

        self._sizes: list[int] = sorted(sizes, reverse=True)
        self._number_of_items: int = len(self._sizes)
        self._bin_capacity: int = bin_capacity
        self._solution: Solution | None = None

        # Precompute suffix sums for O(1) remaining volume
        n = self._number_of_items
        suf = [0] * (n + 1)
        total = 0
        for i in range(n - 1, -1, -1):
            total += self._sizes[i]
            suf[i] = total
        self._suffix_sum = suf

    def solve(self) -> None:
        self._branch_and_bound()

    def get_solution(self) -> Solution:
        if self._solution is None:
            raise RuntimeError("No solution: call solve() first.")
        return self._solution

    # ---------- bounds / helpers ----------

    def _evaluation(self, state: State) -> int:
        """
        Fast LB:
          remaining = suffix_sum[next_item]
          free_space = bins_used*C - used_volume
          extra = max(0, remaining - free_space)
          LB = bins_used + ceil(extra/C)
        All O(1).
        """
        remaining = self._suffix_sum[state.next_item]
        free_space = state.bins_used * self._bin_capacity - state.used_volume
        extra = remaining - free_space
        if extra <= 0:
            return state.bins_used
        # ceil(extra/C) without floats is fine, but ceil(int/int) is OK too
        return state.bins_used + (extra + self._bin_capacity - 1) // self._bin_capacity

    def _is_goal(self, state: State) -> bool:
        return state.next_item == self._number_of_items

    @staticmethod
    def _build_assignments_from_item_bins(
        item_bins: list[int], bins_used: int
    ) -> dict[int, set[int]]:
        a: dict[int, set[int]] = {b: set() for b in range(bins_used)}
        for i, b in enumerate(item_bins):
            a[b].add(i)
        return a

    def _reconstruct_item_bins(self, goal: State) -> list[int]:
        """Walk parent pointers backward and fill item_bins."""
        n = self._number_of_items
        item_bins = [0] * n
        s = goal
        while s.parent is not None:
            item = s.next_item - 1
            item_bins[item] = s.placed_bin
            s = s.parent
        return item_bins

    # ---------- heuristics (upper bound) ----------

    def _best_fit_decreasing_upper_bound(self) -> tuple[int, list[int], list[int]]:
        """
        Best-Fit Decreasing (often tighter than FFD):
        returns (bins_used, item_bins, loads_list)
        """
        C = self._bin_capacity
        loads: list[int] = []
        item_bins: list[int] = [0] * self._number_of_items

        for i, size in enumerate(self._sizes):
            best_bin = -1
            best_rem = C + 1
            for b, load in enumerate(loads):
                rem = C - load
                if size <= rem and rem - size < best_rem:
                    best_rem = rem - size
                    best_bin = b

            if best_bin >= 0:
                loads[best_bin] += size
                item_bins[i] = best_bin
            else:
                item_bins[i] = len(loads)
                loads.append(size)

        return len(loads), item_bins, loads

    # ---------- branching ----------

    def _generate_children(self, state: State) -> list[State]:
        """
        Symmetry breaking:
        - Skip bins with equal loads (interchangeable)
        - Only open a new bin if the item does NOT fit in any existing bin
          (huge branching reduction, still exact)
        """
        children: list[State] = []
        i = state.next_item
        size = self._sizes[i]
        C = self._bin_capacity
        loads = state.loads

        seen_loads: set[int] = set()
        fits_somewhere = False

        # Try existing bins
        for b in range(state.bins_used):
            lb = loads[b]
            if lb in seen_loads:
                continue
            seen_loads.add(lb)

            if lb + size <= C:
                fits_somewhere = True
                new_loads = loads.copy()
                new_loads[b] = lb + size
                children.append(
                    State(
                        bins_used=state.bins_used,
                        loads=new_loads,
                        next_item=i + 1,
                        used_volume=state.used_volume + size,
                        parent=state,
                        placed_bin=b,
                    )
                )

        # Open new bin only if needed (exact + big cut)
        if not fits_somewhere:
            new_loads = loads.copy()
            new_loads.append(size)
            children.append(
                State(
                    bins_used=state.bins_used + 1,
                    loads=new_loads,
                    next_item=i + 1,
                    used_volume=state.used_volume + size,
                    parent=state,
                    placed_bin=state.bins_used,  # new bin index
                )
            )

        return children

    # ---------- main search ----------

    def _branch_and_bound(self) -> None:
        # Upper bound from greedy
        ub_bins, ub_item_bins, ub_loads = self._best_fit_decreasing_upper_bound()
        incumbent_bins_used = ub_bins
        incumbent_goal_state: State | None = (
            None  # if we find an optimal-by-search solution
        )

        # Root
        root = State(
            bins_used=0,
            loads=[],
            next_item=0,
            used_volume=0,
            parent=None,
            placed_bin=-1,
        )

        # Best-first search by lower bound
        heap: list[tuple[int, int, State]] = []
        counter = 0
        heapq.heappush(heap, (self._evaluation(root), counter, root))

        while heap:
            lb, _, s = heapq.heappop(heap)
            if lb >= incumbent_bins_used:
                continue

            if self._is_goal(s):
                # Found a better feasible solution
                incumbent_bins_used = s.bins_used
                incumbent_goal_state = s
                continue

            # Expand
            for child in self._generate_children(s):
                child_lb = self._evaluation(child)
                if child_lb >= incumbent_bins_used:
                    continue
                counter += 1
                heapq.heappush(heap, (child_lb, counter, child))

        # Build final solution:
        # - If search found a better goal state, reconstruct from it
        # - Else return greedy UB
        if incumbent_goal_state is not None:
            item_bins = self._reconstruct_item_bins(incumbent_goal_state)
            bins_used = incumbent_goal_state.bins_used
            loads = incumbent_goal_state.loads
        else:
            item_bins = ub_item_bins
            bins_used = ub_bins
            loads = ub_loads

        assignments = self._build_assignments_from_item_bins(item_bins, bins_used)
        self._solution = Solution(
            bins_used=bins_used, assignments=assignments, loads=loads
        )
