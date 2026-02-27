from dataclasses import dataclass


@dataclass(slots=True)
class Solution:
    bins_used: int  # Number of bins used in this state
    assignments: dict[int, set[int]]  # Mapping from bin index to items packed
    loads: dict[int, int]  # Current load of each bin


@dataclass(slots=True)
class State(Solution):
    next_item: int  # Index of the next item to assign


class BinPacking:
    __slots__ = ("_sizes", "_number_of_items", "_bin_capacity", "_solution")

    def __init__(self, sizes: list[int], bin_capacity: int) -> None:
        # Ensure all items can fit in a single bin
        if any(size > bin_capacity for size in sizes):
            raise ValueError("All item sizes must be <= bin capacity.")

        self._sizes = sizes
        self._number_of_items: int = len(sizes)
        self._bin_capacity: int = bin_capacity
        self._solution: Solution | None = None

    def solve(self, method: str = "bb") -> None:
        """Select solving method."""
        if method == "bb":
            self._branch_and_bound()
        elif method == "dp":
            self._dynamic_programming()
        else:
            raise ValueError("Unknown solving method.")

    def get_solution(self) -> Solution | None:
        """Retrieve computed solution."""
        return self._solution

    def _evaluation(self, state: State) -> int:
        """Objective function: number of bins used."""
        return state.bins_used

    def _is_goal(self, state: State) -> bool:
        """Check if all items have been assigned exactly once."""
        encountered = [False] * self._number_of_items
        item_count = 0
        for items in state.assignments.values():
            for item in items:
                if encountered[item]:
                    return False
                encountered[item] = True
                item_count += 1
        return item_count == self._number_of_items

    def _generate_new_states(self, state: State) -> list[State]:
        """Generate successor states by placing next item in existing or new bins."""
        new_states: list[State] = []
        current_item = state.next_item

        # Place item in existing bins if it fits
        for bin_index in range(state.bins_used):
            if self._sizes[current_item] <= self._bin_capacity - state.loads.get(
                bin_index, 0
            ):
                # Deep copy assignments
                new_assignments = {
                    key: set(value) for key, value in state.assignments.items()
                }
                new_assignments[bin_index].add(current_item)
                new_loads = state.loads.copy()
                new_loads[bin_index] = (
                    new_loads.get(bin_index, 0) + self._sizes[current_item]
                )
                new_states.append(
                    State(state.bins_used, new_assignments, new_loads, current_item + 1)
                )

        # Place item in a new bin
        new_bins_used = state.bins_used + 1
        new_assignments = {k: set(v) for k, v in state.assignments.items()}
        new_assignments[new_bins_used - 1] = {current_item}
        new_loads = state.loads.copy()
        new_loads[new_bins_used - 1] = self._sizes[current_item]
        new_states.append(
            State(new_bins_used, new_assignments, new_loads, current_item + 1)
        )

        return new_states

    def _branch_and_bound(self) -> None:
        """Branch-and-bound search to minimize number of bins."""
        frontier: list[State] = [State(0, {}, {}, 0)]
        incumbent: State = State(self._number_of_items + 1, {}, {}, 0)

        while frontier:
            current_state = frontier.pop()
            if self._is_goal(current_state) and self._evaluation(
                current_state
            ) < self._evaluation(incumbent):
                incumbent = current_state
            else:
                for state in self._generate_new_states(current_state):
                    # Prune states that cannot improve incumbent
                    if state.bins_used >= incumbent.bins_used:
                        continue
                    frontier.append(state)

        self._solution = Solution(
            bins_used=incumbent.bins_used,
            assignments=incumbent.assignments,
            loads=incumbent.loads,
        )

    def _dynamic_programming(self) -> None:
        """Dynamic programming method placeholder."""
        ...
