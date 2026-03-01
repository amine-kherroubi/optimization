from dataclasses import dataclass
from math import ceil


@dataclass(slots=True)
class Solution(object):
    bins_used: int
    assignments: dict[int, set[int]]
    loads: dict[int, int]


@dataclass(slots=True)
class State(Solution):
    next_item: int


class BinPacking(object):
    __slots__ = ("_sizes", "_number_of_items", "_bin_capacity", "_solution")

    def __init__(self, sizes: list[int], bin_capacity: int) -> None:
        if any(size > bin_capacity for size in sizes):
            raise ValueError(
                "All item sizes must be less than or equal to bin capacity."
            )

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

    def _evaluation(self, state: State) -> int:
        """Lower bound on the number of bins needed from this state onward."""
        remaining_volume: int = sum(
            self._sizes[i] for i in range(state.next_item, self._number_of_items)
        )
        free_space: int = sum(
            self._bin_capacity - state.loads.get(b, 0) for b in range(state.bins_used)
        )
        extra_volume: int = max(0, remaining_volume - free_space)
        return state.bins_used + ceil(extra_volume / self._bin_capacity)

    def _is_goal(self, state: State) -> bool:
        """Check if all items have been assigned."""
        return state.next_item == self._number_of_items

    def _generate_new_states(self, state: State) -> list[State]:
        """Generate successor states by placing the next item in existing or new bins."""
        new_states: list[State] = []
        current_item: int = state.next_item
        item_size: int = self._sizes[current_item]
        seen_loads: set[int] = set()

        for bin_index in range(state.bins_used):
            current_load: int = state.loads.get(bin_index, 0)
            if current_load in seen_loads:
                continue
            if item_size <= self._bin_capacity - current_load:
                seen_loads.add(current_load)
                new_assignments: dict[int, set[int]] = {
                    key: set(value) for key, value in state.assignments.items()
                }
                new_assignments[bin_index].add(current_item)
                new_loads: dict[int, int] = state.loads.copy()
                new_loads[bin_index] = current_load + item_size
                new_states.append(
                    State(state.bins_used, new_assignments, new_loads, current_item + 1)
                )

        new_bins_used: int = state.bins_used + 1
        new_assignments = {key: set(value) for key, value in state.assignments.items()}
        new_assignments[new_bins_used - 1] = {current_item}
        new_loads = state.loads.copy()
        new_loads[new_bins_used - 1] = item_size
        new_states.append(
            State(new_bins_used, new_assignments, new_loads, current_item + 1)
        )

        return new_states

    def _first_fit_decreasing(self) -> State:
        """Build an initial feasible solution using the FFD greedy heuristic."""
        loads: dict[int, int] = {}
        assignments: dict[int, set[int]] = {}
        bins_used: int = 0

        for item_index, size in enumerate(self._sizes):
            placed: bool = False
            for bin_index in range(bins_used):
                if loads[bin_index] + size <= self._bin_capacity:
                    loads[bin_index] += size
                    assignments[bin_index].add(item_index)
                    placed = True
                    break
            if not placed:
                assignments[bins_used] = {item_index}
                loads[bins_used] = size
                bins_used += 1

        return State(bins_used, assignments, loads, self._number_of_items)

    def _branch_and_bound(self) -> None:
        """Branch-and-bound using depth-first search via an explicit stack."""
        incumbent: State = self._first_fit_decreasing()

        # A list used as a LIFO stack — heapq is replaced by a plain list.
        # heapq.heappush → stack.append
        # heapq.heappop  → stack.pop   (retrieves the last-appended element)
        initial_state: State = State(0, {}, {}, 0)
        stack: list[State] = [initial_state]

        while stack:
            current_state: State = stack.pop()  # LIFO: deepens the search immediately

            # Prune if this branch cannot improve the incumbent.
            # With DFS the lower bound must be recomputed here, because states
            # are no longer stored alongside their bound as heap tuples.
            if self._evaluation(current_state) >= incumbent.bins_used:
                continue

            if self._is_goal(current_state):
                incumbent = current_state
                continue

            for state in self._generate_new_states(current_state):
                if self._evaluation(state) < incumbent.bins_used:
                    stack.append(state)

        self._solution = Solution(
            bins_used=incumbent.bins_used,
            assignments=incumbent.assignments,
            loads=incumbent.loads,
        )

    def _dynamic_programming(self) -> None:
        raise NotImplementedError("Dynamic programming not implemented.")
