from dataclasses import dataclass
from math import ceil
import heapq


@dataclass(slots=True)
class Solution:
    bins_used: int
    assignments: dict[int, set[int]]
    loads: dict[int, int]


class BinPacking:
    __slots__ = ("_sizes", "_n", "_cap", "_suffix", "_solution")

    def __init__(self, sizes: list[int], bin_capacity: int) -> None:
        if any(s > bin_capacity for s in sizes):
            raise ValueError("All item sizes must be <= bin capacity.")

        self._sizes: list[int] = sorted(sizes, reverse=True)
        self._n: int = len(sizes)
        self._cap: int = bin_capacity

        # --- CHANGE 1: Precompute suffix sums ---
        # Instead of recomputing sum(sizes[next_item:]) on every lower-bound
        # call (O(n) each time), store a prefix array so each lookup is O(1).
        self._suffix: list[int] = [0] * (self._n + 1)
        for i in range(self._n - 1, -1, -1):
            self._suffix[i] = self._suffix[i + 1] + self._sizes[i]

        self._solution: Solution | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, method: str = "b&b") -> None:
        if method == "b&b":
            self._branch_and_bound()
        elif method == "dp":
            self._dynamic_programming()
        else:
            raise ValueError("Unknown solving method.")

    def get_solution(self) -> Solution:
        if self._solution is None:
            raise RuntimeError("No solution available: call 'solve()' first.")
        return self._solution

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lower_bound(self, loads: tuple[int, ...], next_item: int) -> int:
        """Continuous relaxation lower bound, now O(k) instead of O(n)."""
        remaining = self._suffix[next_item]  # O(1) via precomputed array
        free = sum(self._cap - l for l in loads)  # O(k), k = bins open so far
        extra = max(0, remaining - free)
        return len(loads) + ceil(extra / self._cap)

    def _ffd(self) -> tuple[int, tuple[int, ...]]:
        """First-Fit Decreasing heuristic for a warm-start upper bound."""
        loads: list[int] = []
        for size in self._sizes:
            for i in range(len(loads)):
                if loads[i] + size <= self._cap:
                    loads[i] += size
                    break
            else:
                loads.append(size)
        return len(loads), tuple(sorted(loads, reverse=True))

    # ------------------------------------------------------------------
    # Branch-and-bound
    # ------------------------------------------------------------------

    def _branch_and_bound(self) -> None:
        """
        Best-first B&B over a *canonical* state space.

        STATE REPRESENTATION (Change 2)
        --------------------------------
        Previous state: three dicts (bins_used, assignments, loads) + next_item.
        New state:      (loads_tuple: tuple[int,...], next_item: int)

        The loads tuple is kept sorted in *descending* order.  This gives every
        equivalent bin assignment a single canonical form - two partial solutions
        that open the same bins in different order map to the same state.

        This unlocks Change 3 (visited set) and eliminates the deep-copy of
        assignment dicts, the single costliest operation in the original code.

        VISITED SET (Change 3)
        ----------------------
        Because the state is now a hashable tuple, we maintain a visited set.
        Any state popped from the heap that was already processed is skipped
        immediately ("lazy deletion").  This collapses the exponentially many
        permutations of equivalent bin orderings into one representative each,
        drastically shrinking the effective search tree.

        ASSIGNMENTS REMOVED FROM SEARCH (Change 4)
        -------------------------------------------
        Tracking which item goes in which bin during search is unnecessary and
        expensive.  Assignments are reconstructed in a fast post-processing step
        once the optimal bin count is known.

        ADDITIONAL PRUNING (Change 5)
        ------------------------------
        - Opening a new bin is attempted only when len(loads)+1 < incumbent_k;
          once it would merely tie the incumbent there is no point branching.
        - Global LB (ceil(total_volume / capacity)) triggers an early exit when
          FFD is already optimal.
        """
        incumbent_k, incumbent_loads = self._ffd()

        global_lb = ceil(self._suffix[0] / self._cap)
        if global_lb == incumbent_k:
            # FFD already optimal; skip the search entirely.
            self._solution = self._reconstruct(incumbent_k, incumbent_loads)
            return

        counter: int = 0
        initial: tuple[tuple[int, ...], int] = ((), 0)
        heap: list[tuple[int, int, tuple]] = [(global_lb, 0, initial)]
        visited: set[tuple[tuple[int, ...], int]] = set()

        while heap:
            lb, _, (loads, next_item) = heapq.heappop(heap)

            state_key = (loads, next_item)
            if state_key in visited:
                continue
            visited.add(state_key)

            if lb >= incumbent_k:
                continue

            if next_item == self._n:
                incumbent_k = len(loads)
                incumbent_loads = loads
                if incumbent_k == global_lb:
                    break  # Proven optimal
                continue

            size = self._sizes[next_item]
            seen: set[int] = set()
            mutable = list(loads)

            # --- Branch A: place item in an existing bin ---
            for i in range(len(mutable)):
                l = mutable[i]
                if l in seen:
                    # Bins with identical loads are interchangeable; skip.
                    continue
                if l + size <= self._cap:
                    seen.add(l)
                    mutable[i] = l + size
                    # Re-sorting maintains the canonical (descending) order.
                    # For typical bin counts (<<100) this is negligible.
                    new_loads = tuple(sorted(mutable, reverse=True))
                    mutable[i] = l  # Restore for next iteration
                    new_state = (new_loads, next_item + 1)
                    if new_state not in visited:
                        new_lb = self._lower_bound(new_loads, next_item + 1)
                        if new_lb < incumbent_k:
                            counter += 1
                            heapq.heappush(heap, (new_lb, counter, new_state))

            # --- Branch B: open a new bin ---
            # Guard: only branch if the extra bin can still beat the incumbent.
            if len(loads) + 1 < incumbent_k:
                new_loads = tuple(sorted(mutable + [size], reverse=True))
                new_state = (new_loads, next_item + 1)
                if new_state not in visited:
                    new_lb = self._lower_bound(new_loads, next_item + 1)
                    if new_lb < incumbent_k:
                        counter += 1
                        heapq.heappush(heap, (new_lb, counter, new_state))

        self._solution = self._reconstruct(incumbent_k, incumbent_loads)

    # ------------------------------------------------------------------
    # Post-processing: reconstruct item assignments (Change 4 companion)
    # ------------------------------------------------------------------

    def _reconstruct(self, k: int, target_loads: tuple[int, ...]) -> Solution:
        """
        Given the final bin loads of the optimal solution, recover a valid
        item-to-bin assignment via backtracking.

        This is far cheaper than tracking assignments during the search because:
          - it runs exactly once, after the search terminates;
          - the target is fully constrained (k and each bin's total load are
            fixed), so the backtracking tree is tiny.

        Algorithm: treat target_loads as "remaining capacity to fill".  For each
        item (largest first), try assigning it to a bin whose remaining load
        covers its size.  Duplicate remaining-load values are collapsed to avoid
        symmetric branches.
        """
        remaining: list[int] = list(target_loads)
        assignments: dict[int, set[int]] = {b: set() for b in range(k)}

        def backtrack(idx: int) -> bool:
            if idx == self._n:
                return True
            size = self._sizes[idx]
            seen: set[int] = set()
            for b in range(k):
                r = remaining[b]
                if r in seen or r < size:
                    continue
                seen.add(r)
                remaining[b] -= size
                assignments[b].add(idx)
                if backtrack(idx + 1):
                    return True
                remaining[b] += size
                assignments[b].discard(idx)
            return False

        backtrack(0)

        loads = {b: sum(self._sizes[j] for j in s) for b, s in assignments.items()}
        return Solution(k, assignments, loads)

    # ------------------------------------------------------------------

    def _dynamic_programming(self) -> None:
        raise NotImplementedError("Dynamic programming not implemented.")
