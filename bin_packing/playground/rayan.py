from dataclasses import dataclass
from math import ceil
import heapq


@dataclass(slots=True)
class Solution(object):
    bins_used: int  # Number of bins used
    assignments: dict[int, set[int]]  # Mapping from bin index to items packed
    loads: dict[int, int]  # Load of each bin


@dataclass(slots=True)
class State(Solution):
    next_item: int  # Index of the next item to assign


class BinPacking(object):
    __slots__ = ("_sizes", "_number_of_items", "_bin_capacity", "_solution")

    def __init__(self, sizes: list[int], bin_capacity: int) -> None:
        if any(size > bin_capacity for size in sizes):
            raise ValueError(
                "All item sizes must be less than or equal to bin capacity."
            )

        "Sorting items in decreasing order (FFD heuristic) for better early solutions and pruning"
        self._sizes: list[int] = sorted(sizes, reverse=True)
        self._number_of_items: int = len(sizes)
        self._bin_capacity: int = bin_capacity
        self._solution: Solution | None = None

    def solve(self, method: str = "b&b") -> None:
        """Select solving method."""
        if method == "b&b":
            self._branch_and_bound()
        elif method == "dp":
            self._dynamic_programming()
        else:
            raise ValueError("Unknown solving method.")

    def get_solution(self) -> Solution:
        """Retrieve computed solution."""
        if self._solution is None:
            raise RuntimeError(
                "No solution available: you must call 'solve()' before retrieving the solution."
            )
        return self._solution

    """
    Lower bound evaluation
    extra_volume = max(0, remaining_volume - free_space)
    LB = bins_used + ceil(extra_volume / bin_capacity)
    """

    def _evaluation(self, state: State) -> int:
        remaining_volume: int = sum(
            self._sizes[i] for i in range(state.next_item, self._number_of_items)
        )

        free_space: int = sum(
            self._bin_capacity - state.loads.get(b, 0) for b in range(state.bins_used)
        )

        extra_volume: int = max(0, remaining_volume - free_space)

        return state.bins_used + ceil(extra_volume / self._bin_capacity)

    def _is_goal(self, state: State) -> bool:
        """Check if all items have been assigned (inchangé)."""
        return state.next_item == self._number_of_items

    """
    Group bins by load and generate one state per unique load
    Bins with same load are interchangeable → avoid duplicates.
    """

    def _generate_new_states(self, state: State) -> list[State]:
        """
        Generate successor states by placing next item in existing or new bins.
        Symmetry breaking: skip bins with duplicate load values.
        """
        new_states: list[State] = []
        current_item = state.next_item
        item_size = self._sizes[current_item]

        seen_loads: set[int] = set()  # already processed loads

        for bin_index in range(state.bins_used):
            current_load = state.loads.get(bin_index, 0)

            # Skip symmetric bins
            if current_load in seen_loads:
                continue

            if item_size <= self._bin_capacity - current_load:
                seen_loads.add(current_load)

                new_assignments = {k: set(v) for k, v in state.assignments.items()}
                new_assignments[bin_index].add(current_item)
                new_loads = state.loads.copy()
                new_loads[bin_index] = current_load + item_size

                new_states.append(
                    State(state.bins_used, new_assignments, new_loads, current_item + 1)
                )

        # Open new bin (empty bins are symmetric)
        new_bins_used = state.bins_used + 1
        new_assignments = {k: set(v) for k, v in state.assignments.items()}
        new_assignments[new_bins_used - 1] = {current_item}
        new_loads = state.loads.copy()
        new_loads[new_bins_used - 1] = item_size

        new_states.append(
            State(new_bins_used, new_assignments, new_loads, current_item + 1)
        )

        return new_states

    """Gives realistic initial solution → early pruning"""

    def _first_fit_decreasing(self) -> State:
        """
        Greedy FFD heuristic to build an initial feasible solution.
        Items are already sorted in decreasing order (done in __init__).
        """
        loads: dict[int, int] = {}
        assignments: dict[int, set[int]] = {}
        bins_used = 0

        for item_index, size in enumerate(self._sizes):
            placed = False
            for b in range(bins_used):
                if loads[b] + size <= self._bin_capacity:
                    loads[b] += size
                    assignments[b].add(item_index)
                    placed = True
                    break
            if not placed:
                assignments[bins_used] = {item_index}
                loads[bins_used] = size
                bins_used += 1

        return State(bins_used, assignments, loads, self._number_of_items)

    """
    Best-First Search using min-heap
    priority queue ordered by lower bound
    Explore most promising states first → reach optimum earlier
    """

    def _branch_and_bound(self) -> None:
        # Incumbent initial via FFD
        incumbent: State = self._first_fit_decreasing()

        # Min-heap frontier: (lower_bound, counter, state)
        # counter is used to avoid comparing two State objects directly.
        counter = 0
        initial_state = State(0, {}, {}, 0)
        frontier: list[tuple[int, int, State]] = []
        heapq.heappush(
            frontier, (self._evaluation(initial_state), counter, initial_state)
        )

        while frontier:
            lb, _, current_state = heapq.heappop(frontier)

            # Prune if bound already worse than incumbent
            if lb >= incumbent.bins_used:
                continue

            if self._is_goal(current_state):
                # Update incumbent if we found a better solution
                if self._evaluation(current_state) < self._evaluation(incumbent):
                    incumbent = current_state
                continue

            for state in self._generate_new_states(current_state):
                lb_state = self._evaluation(state)
                #  Prune states that cannot improve incumbent
                if lb_state >= incumbent.bins_used:
                    continue
                counter += 1
                heapq.heappush(frontier, (lb_state, counter, state))

        self._solution = Solution(
            bins_used=incumbent.bins_used,
            assignments=incumbent.assignments,
            loads=incumbent.loads,
        )

    def _dynamic_programming(self) -> None:
        """Dynamic programming method."""
        raise NotImplementedError("Dynamic programming not implemented.")
