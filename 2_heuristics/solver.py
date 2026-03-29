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

            case "first fit decreasing":
                self._first_fit_decreasing()

            case "relocation":
                self._relocation()

            case "swap":
                self._swap()

            case _:
                raise ValueError(
                    "Unsupported method. Available methods: next fit, first fit, "
                    "best fit, first fit decreasing, relocation, swap."
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

    def _first_fit_decreasing(self) -> None:
        # Items are already sorted in descending order by the constructor.
        # Running first fit on a descending-sorted list is first fit decreasing.
        self._first_fit()

    def _apply_relocation(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> tuple[list[int], list[list[int]]]:
        # Repeatedly attempt to empty the least loaded bin by moving all its items elsewhere.
        # If every item in the least loaded bin can be relocated, that bin is eliminated.
        # This continues until no bin can be emptied.
        improved = True
        while improved:
            improved = False

            source_order = sorted(
                range(len(assignments)),
                key=lambda index: bin_loads[index],
            )

            for source_bin_index in source_order:
                if not assignments[source_bin_index]:
                    continue

                # Work on a temporary copy so the original state is preserved on failure.
                temporary_loads = list(bin_loads)
                temporary_assignments = [list(items) for items in assignments]
                all_items_placed = True

                for item_index in list(temporary_assignments[source_bin_index]):
                    item_size = self._item_sizes[item_index]
                    item_placed = False

                    for target_bin_index in range(len(temporary_loads)):
                        if target_bin_index == source_bin_index:
                            continue
                        if not temporary_assignments[target_bin_index]:
                            continue
                        if temporary_loads[target_bin_index] + item_size <= self._bin_capacity:
                            temporary_loads[target_bin_index] += item_size
                            temporary_assignments[target_bin_index].append(item_index)
                            temporary_loads[source_bin_index] -= item_size
                            temporary_assignments[source_bin_index].remove(item_index)
                            item_placed = True
                            break

                    if not item_placed:
                        all_items_placed = False
                        break

                if all_items_placed:
                    bin_loads = temporary_loads
                    assignments = temporary_assignments
                    improved = True
                    break

        return bin_loads, assignments

    def _relocation(self) -> None:
        # Start from the first fit decreasing solution, then apply relocation.
        self._first_fit_decreasing()

        bin_loads: list[int] = list(self._final_solution.final_bin_loads)
        assignments: list[list[int]] = [
            list(items) for items in self._final_solution.bin_assignments.values()
        ]

        bin_loads, assignments = self._apply_relocation(bin_loads, assignments)

        self._final_solution = self._build_solution(bin_loads, assignments)

    def _swap(self) -> None:
        # Start from the relocation solution, then attempt pairwise item swaps.
        # A swap is accepted if, after swapping two items between two bins and
        # re-running relocation, the total number of bins strictly decreases.
        self._relocation()

        bin_loads: list[int] = list(self._final_solution.final_bin_loads)
        assignments: list[list[int]] = [
            list(items) for items in self._final_solution.bin_assignments.values()
        ]

        improved = True
        while improved:
            improved = False
            current_bin_count = sum(1 for items in assignments if items)

            for bin_a_index in range(len(assignments)):
                if not assignments[bin_a_index]:
                    continue

                for bin_b_index in range(bin_a_index + 1, len(assignments)):
                    if not assignments[bin_b_index]:
                        continue

                    for item_a_index in list(assignments[bin_a_index]):
                        for item_b_index in list(assignments[bin_b_index]):
                            size_a = self._item_sizes[item_a_index]
                            size_b = self._item_sizes[item_b_index]
                            new_load_a = bin_loads[bin_a_index] - size_a + size_b
                            new_load_b = bin_loads[bin_b_index] - size_b + size_a

                            # The swap must keep both bins within capacity.
                            if new_load_a > self._bin_capacity or new_load_b > self._bin_capacity:
                                continue

                            # Tentatively apply the swap on a copy of the current state.
                            candidate_loads = list(bin_loads)
                            candidate_assignments = [list(items) for items in assignments]
                            candidate_loads[bin_a_index] = new_load_a
                            candidate_loads[bin_b_index] = new_load_b
                            candidate_assignments[bin_a_index].remove(item_a_index)
                            candidate_assignments[bin_a_index].append(item_b_index)
                            candidate_assignments[bin_b_index].remove(item_b_index)
                            candidate_assignments[bin_b_index].append(item_a_index)

                            # Run relocation on the candidate state to check for improvement.
                            candidate_loads, candidate_assignments = self._apply_relocation(
                                candidate_loads, candidate_assignments
                            )
                            candidate_bin_count = sum(1 for items in candidate_assignments if items)

                            if candidate_bin_count < current_bin_count:
                                bin_loads = candidate_loads
                                assignments = candidate_assignments
                                improved = True
                                break

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        self._final_solution = self._build_solution(bin_loads, assignments)

    def _build_solution(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> BinPackingSolution:
        # Remove empty bins and rebuild with contiguous indices.
        non_empty_bins = [
            (load, items)
            for load, items in zip(bin_loads, assignments)
            if items
        ]
        compacted_loads = [load for load, _ in non_empty_bins]
        compacted_assignments = {
            index: items for index, (_, items) in enumerate(non_empty_bins)
        }
        return BinPackingSolution(
            total_bins_used=len(compacted_loads),
            bin_assignments=compacted_assignments,
            final_bin_loads=compacted_loads,
        )

    def get_solution(self) -> BinPackingSolution:
        if self._final_solution is None:
            raise RuntimeError(
                "No solution available: you must call 'solve()' before retrieving the solution."
            )

        return self._final_solution
