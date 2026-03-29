from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BinPackingSolution:
    """Compact representation of a bin packing solution."""

    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[int]


@dataclass(slots=True)
class SearchState:
    """State container for the depth-first search variants."""

    item_index: int
    current_bin_loads: list[int]
    assignments: list[int] = field(default_factory=list)

    @property
    def bins_count(self) -> int:
        return len(self.current_bin_loads)


class BinPackingSolver:
    """Exact solvers for the 1D bin packing problem."""

    __slots__ = ("_item_sizes", "_total_item_count", "_bin_capacity", "_final_solution")

    def __init__(self, item_sizes: list[int], bin_capacity: int):
        if any(size > bin_capacity for size in item_sizes):
            raise ValueError("No single item size can exceed the bin capacity.")

        # Sorting items in descending order is a crucial optimization for
        # both the initial heuristic and the search tree pruning.
        self._item_sizes: list[int] = sorted(item_sizes, reverse=True)
        self._total_item_count: int = len(item_sizes)
        self._bin_capacity: int = bin_capacity
        self._final_solution: BinPackingSolution | None = None

    def _compute_l1_lower_bound(self, remaining_item_indices: list[int]) -> int:
        if not remaining_item_indices:
            return 0

        remaining_volume = sum(self._item_sizes[i] for i in remaining_item_indices)
        return (remaining_volume + self._bin_capacity - 1) // self._bin_capacity

    def _first_fit_decreasing(self) -> BinPackingSolution:
        bin_loads: list[int] = []
        assignments: dict[int, list[int]] = {}

        for item_index, size in enumerate(self._item_sizes):
            placed = False
            for bin_index, load in enumerate(bin_loads):
                if load + size <= self._bin_capacity:
                    bin_loads[bin_index] += size
                    assignments[bin_index].append(item_index)
                    placed = True
                    break

            if not placed:
                new_bin_index = len(bin_loads)
                bin_loads.append(size)
                assignments[new_bin_index] = [item_index]

        return BinPackingSolution(
            total_bins_used=len(bin_loads),
            bin_assignments=assignments,
            final_bin_loads=bin_loads,
        )

    def solve(self, method: str = "backtracking") -> None:
        match method.lower():
            case "backtracking":
                self._backtracking()

            case "branch and bound":
                self._branch_and_bound()

            case "dynamic programming":
                self._dynamic_programming()

            case _:
                raise ValueError(
                    "Unsupported method. "
                    "Available methods: backtracking, branch and bound, dynamic programming."
                )

    def _backtracking(self) -> None:
        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        # Use FFD to get an initial upper bound, then try to improve it.
        best_solution = self._first_fit_decreasing()
        best_bin_count = best_solution.total_bins_used

        stack: list[SearchState] = [SearchState(0, [], [])]

        while stack:
            state = stack.pop()

            # All items placed — check if this solution beats the current best.
            if state.item_index == self._total_item_count:
                if state.bins_count < best_bin_count:
                    best_bin_count = state.bins_count
                    best_solution = self._convert_to_solution(state)
                continue

            current_item_size = self._item_sizes[state.item_index]

            # Try placing the item into existing bins with symmetry breaking:
            # skip bins with the same load as one already tried to avoid duplicate paths.
            seen_loads: set[int] = set()
            for bin_index, load in enumerate(state.current_bin_loads):
                if (
                    load not in seen_loads
                    and load + current_item_size <= self._bin_capacity
                ):
                    seen_loads.add(load)
                    updated_loads = state.current_bin_loads.copy()
                    updated_loads[bin_index] += current_item_size
                    stack.append(
                        SearchState(
                            state.item_index + 1,
                            updated_loads,
                            state.assignments + [bin_index],
                        )
                    )

            # Try opening a new bin only if it can still beat the current best.
            if state.bins_count + 1 < best_bin_count:
                stack.append(
                    SearchState(
                        state.item_index + 1,
                        state.current_bin_loads + [current_item_size],
                        state.assignments + [state.bins_count],
                    )
                )

        self._final_solution = best_solution

    def _branch_and_bound(self) -> None:
        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        # Establish initial upper bound using First-Fit Decreasing.
        initial_heuristic = self._first_fit_decreasing()
        best_bin_count = initial_heuristic.total_bins_used
        self._final_solution = initial_heuristic

        # Establish global lower bound.
        global_lower_bound = self._compute_l1_lower_bound(
            list(range(self._total_item_count))
        )

        # If our heuristic already matched the theoretical minimum, we are done.
        if best_bin_count == global_lower_bound:
            return

        # Depth-First Search with Pruning
        stack: list[SearchState] = [SearchState(0, [], [])]

        while stack:
            state = stack.pop()

            # If we've assigned all items, check if this is a new best solution.
            if state.item_index == self._total_item_count:
                if state.bins_count < best_bin_count:
                    best_bin_count = state.bins_count
                    self._final_solution = self._convert_to_solution(state)
                continue

            current_item_size = self._item_sizes[state.item_index]

            # Calculate L1 bound for items not yet packed.
            remaining_indices = list(range(state.item_index, self._total_item_count))
            lower_bound = state.bins_count + self._compute_l1_lower_bound(
                remaining_indices
            )

            # If the best possible outcome of this branch can't beat our current best, prune it.
            if lower_bound >= best_bin_count:
                continue

            # Try placing the item in a new bin.
            if state.bins_count + 1 < best_bin_count:
                new_loads = state.current_bin_loads + [current_item_size]
                new_assignments = state.assignments + [state.bins_count]
                stack.append(
                    SearchState(state.item_index + 1, new_loads, new_assignments)
                )

            # Try placing the item in existing bins using symmetry breaking.
            # We only try one bin of each unique "load" size to avoid redundant paths.
            seen_loads: set[int] = set()
            for bin_index, load in enumerate(state.current_bin_loads):
                if (
                    load not in seen_loads
                    and load + current_item_size <= self._bin_capacity
                ):
                    seen_loads.add(load)

                    updated_loads = state.current_bin_loads.copy()
                    updated_loads[bin_index] += current_item_size
                    updated_assignments = state.assignments + [bin_index]

                    stack.append(
                        SearchState(
                            state.item_index + 1, updated_loads, updated_assignments
                        )
                    )

    # Bitmask DP requires O(2^n) memory. Beyond this threshold the allocation is
    # infeasible regardless of available RAM.
    _DP_MAX_ITEMS: int = 20

    def _dynamic_programming(self) -> None:
        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        if self._total_item_count > self._DP_MAX_ITEMS:
            raise ValueError(
                f"Dynamic programming requires O(2^n) memory and is only supported "
                f"for n <= {self._DP_MAX_ITEMS} items; got n={self._total_item_count}. "
                f"Use 'backtracking' or 'branch and bound' for larger instances."
            )

        # Bitmask DP: dp[mask] = minimum bins needed to pack the items in that mask.
        # For each mask, every fitting submask is a candidate single bin,
        # so dp[mask] = min(dp[mask ^ submask] + 1) over all valid submasks.
        total_masks = 1 << self._total_item_count
        infinity = self._total_item_count + 1

        # Precompute the total size of every subset using the lowest set bit trick.
        subset_sum: list[int] = [0] * total_masks
        for mask in range(1, total_masks):
            lowest_bit = mask & (-mask)
            item_index = lowest_bit.bit_length() - 1
            subset_sum[mask] = (
                subset_sum[mask ^ lowest_bit] + self._item_sizes[item_index]
            )

        dp: list[int] = [infinity] * total_masks
        dp[0] = 0

        # Track which submask was assigned to the last bin for reconstruction.
        last_bin_mask: list[int] = [0] * total_masks

        for mask in range(1, total_masks):
            submask = mask
            while submask > 0:
                if subset_sum[submask] <= self._bin_capacity:
                    remaining_mask = mask ^ submask
                    candidate_bin_count = dp[remaining_mask] + 1
                    if candidate_bin_count < dp[mask]:
                        dp[mask] = candidate_bin_count
                        last_bin_mask[mask] = submask
                submask = (submask - 1) & mask

        # Reconstruct bin assignments by tracing last_bin_mask back to zero.
        bin_assignments: dict[int, list[int]] = {}
        bin_loads: list[int] = []
        bin_index = 0
        remaining_mask = total_masks - 1

        while remaining_mask > 0:
            current_bin_submask = last_bin_mask[remaining_mask]
            items_in_bin: list[int] = []

            bit_mask = current_bin_submask
            while bit_mask > 0:
                lowest_bit = bit_mask & (-bit_mask)
                items_in_bin.append(lowest_bit.bit_length() - 1)
                bit_mask ^= lowest_bit

            bin_assignments[bin_index] = items_in_bin
            bin_loads.append(subset_sum[current_bin_submask])
            bin_index += 1
            remaining_mask ^= current_bin_submask

        self._final_solution = BinPackingSolution(
            total_bins_used=len(bin_loads),
            bin_assignments=bin_assignments,
            final_bin_loads=bin_loads,
        )

    def get_solution(self) -> BinPackingSolution:
        if self._final_solution is None:
            raise RuntimeError(
                "No solution available: you must call 'solve()' before retrieving the solution."
            )

        return self._final_solution

    def _convert_to_solution(self, state: SearchState) -> BinPackingSolution:
        mapping: dict[int, list[int]] = {i: [] for i in range(state.bins_count)}
        for item_idx, bin_idx in enumerate(state.assignments):
            mapping[bin_idx].append(item_idx)

        return BinPackingSolution(
            total_bins_used=state.bins_count,
            bin_assignments=mapping,
            final_bin_loads=state.current_bin_loads,
        )
