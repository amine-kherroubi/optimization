from math import inf
from attr import dataclass

# Mazal ma kemmeltch


# Dataclass mli7a for memory efficiency
@dataclass(slots=True)
class State:
    bins_used: int  # Number of bins used in this state
    assignments: dict[int, set[int]]  # Mapping from bin index to items packed
    loads: dict[int, int]  # Current load of each bin


class BinPacking(object):
    # Hadi tan tdir optimization ta3 memoire
    # (tsupprimi le dictionnaire associé lel class psk ma ra7ch nzido new attributes f runtime)
    __slots__ = ("_sizes", "_number_of_items", "_bin_capacity", "_solution")

    def __init__(self, sizes: list[int], bin_capacity: int) -> None:
        # Validate that all items fit within a single bin
        if len([size for size in sizes if size > bin_capacity]) > 0:
            raise ValueError(
                "All item sizes must be less than or equal to the bin capacity."
            )

        self._sizes = sizes  # List of item sizes
        self._number_of_items: int = len(sizes)  # Total number of items
        self._bin_capacity: int = bin_capacity  # Maximum capacity of each bin
        self._solution: State | None = None  # Stores best solution after solving

    # Hadi strategy design pattern
    def solve(self, method: str = "bb") -> None:
        # Select solving method
        match method:
            case "bb":
                self._solution = self._branch_and_bound()
            case "dp":
                self._solution = self._dynamic_programming()
            case _:
                raise ValueError("Wrong method.")

    def _evaluation(self, state: State) -> int:
        # Objective function: number of bins used
        return state.bins_used

    def _is_goal(self, state: State) -> bool:
        # Check the current state is a goal state (all items have been assigned exactly once)
        encountered: list[bool] = [False for _ in range(self._number_of_items)]
        item_count: int = 0

        for _, items in state.assignments.items():
            for item in items:
                if encountered[item] == True:
                    return False  # Item assigned more than once
                encountered[item] = True
                item_count += 1

        return item_count == self._number_of_items  # All items assigned

    def _generate_new_states(self, state: State, item: int) -> list[State]:
        # Generate possible states by placing `item` in existing bins or a new bin
        new_states: list[State] = []

        # Try placing in existing bins
        for bin in range(0, state.bins_used - 1):
            if self._sizes[item] <= self._bin_capacity - state.loads[bin]:
                new_assignments: dict[int, set[int]] = state.assignments
                new_assignments[bin].add(item)
                new_loads: dict[int, int] = state.loads
                new_loads[bin] += self._sizes[item]
                new_states.append(
                    State(
                        state.bins_used,
                        new_assignments,
                        new_loads,
                    )
                )

        # Place in a new bin
        new_bins_used = state.bins_used + 1
        new_assignments: dict[int, set[int]] = state.assignments
        new_assignments[new_bins_used].add(item)
        new_loads: dict[int, int] = state.loads
        new_loads[new_bins_used] += self._sizes[item]
        new_states.append(State(new_bins_used, new_assignments, new_loads))

        return new_states

    # mazal ma dertch pruning
    def _branch_and_bound(self) -> State:
        # Branch-and-bound search to minimize number of bins
        current_item: int = 0
        frontier: list[State] = [State(0, {}, {})]  # Initial empty state
        incumbent: State = State(int(inf), {}, {})  # Best solution so far

        while len(frontier) > 0:
            current_state: State = frontier.pop()  # Take a state from frontier

            if self._is_goal(current_state) and self._evaluation(
                current_state
            ) < self._evaluation(incumbent):
                incumbent = current_state  # Update best solution if better

            else:
                # Generate new states by placing the current item
                new_states: list[State] = self._generate_new_states(
                    current_state, current_item
                )
                frontier.extend(new_states)  # Add new states to explore

        return incumbent  # Return best solution found

    def _dynamic_programming(self) -> State: ...
