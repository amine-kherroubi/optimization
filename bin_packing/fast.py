from dataclasses import dataclass
from math import ceil
import heapq


@dataclass(slots=True)
class Solution(object):
    bins_used: int  # Number of bins used
    assignments: dict[int, set[int]]  # Mapping from bin index to items packed
    loads: list[int]  # Load of each bin


@dataclass(slots=True)
class State(object):
    # Core search state
    bins_used: int  # Number of open bins
    loads: list[int]  # Load of each open bin
    next_item: int  # Index of the next item to assign
    used_volume: int  # Total volume packed so far

    # Parent pointer for solution reconstruction — avoids copying assignments at every node
    parent: "State | None"
    placed_bin: int  # Bin index where item (next_item - 1) was placed; -1 for root


class BinPacking(object):
    __slots__ = (
        "_sizes",
        "_number_of_items",
        "_bin_capacity",
        "_suffix_sum",
        "_solution",
    )

    def __init__(self, sizes: list[int], bin_capacity: int) -> None:
        # Ensure all items can fit in a single bin
        if any(size > bin_capacity for size in sizes):
            raise ValueError("All item sizes must be <= bin capacity.")

        self._sizes: list[int] = sorted(sizes, reverse=True)  # Decreasing order for BFD
        self._number_of_items: int = len(self._sizes)
        self._bin_capacity: int = bin_capacity
        self._solution: Solution | None = None

        # Precompute suffix sums for O(1) remaining-volume queries
        n: int = self._number_of_items
        suffix_sum: list[int] = [0] * (n + 1)
        total: int = 0
        for i in range(n - 1, -1, -1):
            total += self._sizes[i]
            suffix_sum[i] = total
        self._suffix_sum: list[int] = suffix_sum

    def solve(self) -> None:
        """Run the branch-and-bound solver."""
        self._branch_and_bound()

    def get_solution(self) -> Solution:
        """Retrieve the computed solution."""
        if self._solution is None:
            raise RuntimeError(
                "No solution available: you must call 'solve()' before retrieving the solution."
            )
        return self._solution

    def _evaluation(self, state: State) -> int:
        """Lower bound on the number of bins needed from this state onward.

        Uses suffix sums and aggregated free space for an O(1) computation.
        """
        remaining_volume: int = self._suffix_sum[state.next_item]
        free_space: int = state.bins_used * self._bin_capacity - state.used_volume
        extra_volume: int = remaining_volume - free_space

        if extra_volume <= 0:
            return state.bins_used
        return state.bins_used + ceil(extra_volume / self._bin_capacity)

    def _is_goal(self, state: State) -> bool:
        """Check if all items have been assigned."""
        return state.next_item == self._number_of_items

    def _best_fit_decreasing(self) -> tuple[int, list[int], list[int]]:
        """Build an initial feasible solution using the BFD greedy heuristic.

        Returns (bins_used, item_bins, loads).
        """
        loads: list[int] = []
        item_bins: list[int] = [0] * self._number_of_items

        for item_index, size in enumerate(self._sizes):
            best_bin: int = -1
            best_remaining: int = self._bin_capacity + 1

            # Find the bin with the tightest fit
            for bin_index, load in enumerate(loads):
                remaining: int = self._bin_capacity - load
                if size <= remaining and remaining - size < best_remaining:
                    best_remaining = remaining - size
                    best_bin = bin_index

            if best_bin >= 0:
                loads[best_bin] += size
                item_bins[item_index] = best_bin
            else:
                # Open a new bin
                item_bins[item_index] = len(loads)
                loads.append(size)

        return len(loads), item_bins, loads

    def _generate_children(self, state: State) -> list[State]:
        """Generate successor states by placing the next item in existing or new bins.

        Symmetry breaking:
        - Bins with equal loads are interchangeable and are deduplicated.
        - A new bin is opened only when the item fits nowhere else.
        """
        children: list[State] = []
        current_item: int = state.next_item
        item_size: int = self._sizes[current_item]
        seen_loads: set[int] = set()  # Track visited load values to skip symmetric bins
        fits_somewhere: bool = False

        # Place item in existing bins if it fits
        for bin_index in range(state.bins_used):
            current_load: int = state.loads[bin_index]

            # Skip bins with duplicate loads — they are interchangeable
            if current_load in seen_loads:
                continue
            seen_loads.add(current_load)

            if current_load + item_size <= self._bin_capacity:
                fits_somewhere = True
                new_loads: list[int] = state.loads.copy()
                new_loads[bin_index] = current_load + item_size
                children.append(
                    State(
                        bins_used=state.bins_used,
                        loads=new_loads,
                        next_item=current_item + 1,
                        used_volume=state.used_volume + item_size,
                        parent=state,
                        placed_bin=bin_index,
                    )
                )

        # Open a new bin only when the item fits nowhere else
        if not fits_somewhere:
            new_loads = state.loads.copy()
            new_loads.append(item_size)
            children.append(
                State(
                    bins_used=state.bins_used + 1,
                    loads=new_loads,
                    next_item=current_item + 1,
                    used_volume=state.used_volume + item_size,
                    parent=state,
                    placed_bin=state.bins_used,  # Index of the newly opened bin
                )
            )

        return children

    def _reconstruct_item_bins(self, goal: State) -> list[int]:
        """Walk parent pointers backward to recover per-item bin assignments."""
        item_bins: list[int] = [0] * self._number_of_items
        current: State | None = goal
        while current is not None and current.parent is not None:
            item: int = current.next_item - 1
            item_bins[item] = current.placed_bin
            current = current.parent
        return item_bins

    @staticmethod
    def _build_assignments(item_bins: list[int], bins_used: int) -> dict[int, set[int]]:
        """Build a bin-to-items mapping from a flat item_bins list."""
        assignments: dict[int, set[int]] = {b: set() for b in range(bins_used)}
        for item_index, bin_index in enumerate(item_bins):
            assignments[bin_index].add(item_index)
        return assignments

    def _branch_and_bound(self) -> None:
        """Branch-and-bound method using best-first search."""
        # Warm start via BFD greedy heuristic
        incumbent_bins_used: int
        incumbent_item_bins: list[int]
        incumbent_loads: list[int]
        incumbent_bins_used, incumbent_item_bins, incumbent_loads = (
            self._best_fit_decreasing()
        )
        incumbent_goal_state: State | None = (
            None  # Set only if search improves on the greedy UB
        )

        # Min-heap ordered by lower bound; counter breaks ties without comparing States
        counter: int = 0
        root: State = State(
            bins_used=0,
            loads=[],
            next_item=0,
            used_volume=0,
            parent=None,
            placed_bin=-1,
        )
        frontier: list[tuple[int, int, State]] = []
        heapq.heappush(frontier, (self._evaluation(root), counter, root))

        while frontier:
            lower_bound, _, current_state = heapq.heappop(frontier)

            # Prune if this branch cannot improve the incumbent
            if lower_bound >= incumbent_bins_used:
                continue

            if self._is_goal(current_state):
                incumbent_bins_used = current_state.bins_used
                incumbent_goal_state = current_state
                continue

            for child in self._generate_children(current_state):
                child_lower_bound: int = self._evaluation(child)
                # Prune states that cannot improve the incumbent
                if child_lower_bound >= incumbent_bins_used:
                    continue
                counter += 1
                heapq.heappush(frontier, (child_lower_bound, counter, child))

        # Reconstruct from search if it improved on the greedy upper bound; otherwise use greedy
        if incumbent_goal_state is not None:
            item_bins: list[int] = self._reconstruct_item_bins(incumbent_goal_state)
            bins_used: int = incumbent_goal_state.bins_used
            loads: list[int] = incumbent_goal_state.loads
        else:
            item_bins = incumbent_item_bins
            bins_used = incumbent_bins_used
            loads = incumbent_loads

        self._solution = Solution(
            bins_used=bins_used,
            assignments=self._build_assignments(item_bins, bins_used),
            loads=loads,
        )
