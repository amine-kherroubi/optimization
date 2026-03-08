from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BinPackingSolution:
    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[int]


class BinPackingSolver:
    __slots__ = ("_item_sizes", "_bin_capacity", "_final_solution")

    def __init__(self, item_sizes: list[int], bin_capacity: int):
        if any(size > bin_capacity for size in item_sizes):
            raise ValueError("No single item size can exceed the bin capacity.")

        # Items are sorted in descending order for better packing quality.
        self._item_sizes: list[int] = sorted(item_sizes, reverse=True)
        self._bin_capacity: int = bin_capacity
        self._final_solution: BinPackingSolution | None = None

    def solve(self, method: str) -> None:
        match method.lower():
            case "next fit":
                self._next_fit()

            case "first fit":
                self._first_fit()

            case "best fit":
                self._best_fit()

            case _:
                raise ValueError(
                    "Unsupported method. Available methods: next fit, first fit, best fit."
                )

    def _next_fit(self) -> None:
        # Only the current open bin is considered for placement.
        bin_loads: list[int] = []
        assignments: dict[int, list[int]] = {}

        current_bin_index = -1

        for item_index, size in enumerate(self._item_sizes):
            if current_bin_index == -1:
                bin_loads.append(size)
                assignments[0] = [item_index]
                current_bin_index = 0
                continue

            if bin_loads[current_bin_index] + size <= self._bin_capacity:
                bin_loads[current_bin_index] += size
                assignments[current_bin_index].append(item_index)
            else:
                current_bin_index += 1
                bin_loads.append(size)
                assignments[current_bin_index] = [item_index]

        self._final_solution = BinPackingSolution(
            total_bins_used=len(bin_loads),
            bin_assignments=assignments,
            final_bin_loads=bin_loads,
        )

    def _first_fit(self) -> None:
        # Place each item in the first bin that can accommodate it.
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

        self._final_solution = BinPackingSolution(
            total_bins_used=len(bin_loads),
            bin_assignments=assignments,
            final_bin_loads=bin_loads,
        )

    def _best_fit(self) -> None:
        # Place the item into the bin that leaves the smallest remaining capacity.
        bin_loads: list[int] = []
        assignments: dict[int, list[int]] = {}

        for item_index, size in enumerate(self._item_sizes):
            best_bin_index: int | None = None
            smallest_remaining_capacity = self._bin_capacity + 1

            for bin_index, load in enumerate(bin_loads):
                new_load = load + size

                if new_load <= self._bin_capacity:
                    remaining_capacity = self._bin_capacity - new_load

                    if remaining_capacity < smallest_remaining_capacity:
                        smallest_remaining_capacity = remaining_capacity
                        best_bin_index = bin_index

            if best_bin_index is not None:
                bin_loads[best_bin_index] += size
                assignments[best_bin_index].append(item_index)
            else:
                new_bin_index = len(bin_loads)
                bin_loads.append(size)
                assignments[new_bin_index] = [item_index]

        self._final_solution = BinPackingSolution(
            total_bins_used=len(bin_loads),
            bin_assignments=assignments,
            final_bin_loads=bin_loads,
        )

    def get_solution(self) -> BinPackingSolution:
        if self._final_solution is None:
            raise RuntimeError(
                "No solution available: you must call 'solve()' before retrieving the solution."
            )

        return self._final_solution
