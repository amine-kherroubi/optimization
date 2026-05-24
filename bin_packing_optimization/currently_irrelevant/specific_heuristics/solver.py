from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BinPackingSolution:
    """Compact representation of a bin packing solution."""

    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[int]


class BinPackingSolver:
    """Heuristic solvers for the 1D bin packing problem."""

    __slots__ = (
        "_item_sizes",
        "_bin_capacity",
        "_final_solution",
        "_items_in_input_order",
        "_items_in_descending",
    )

    def __init__(self, item_sizes: list[int], bin_capacity: int):
        if any(size > bin_capacity for size in item_sizes):
            raise ValueError("No single item size can exceed the bin capacity.")

        self._item_sizes: list[int] = list(item_sizes)
        self._bin_capacity: int = bin_capacity
        self._final_solution: BinPackingSolution | None = None

        # Preserve the input order and a descending-by-size order for the
        # "decreasing" heuristics. Original indices are retained for readability
        # when inspecting solutions.
        self._items_in_input_order: list[tuple[int, int]] = list(
            enumerate(self._item_sizes)
        )
        self._items_in_descending: list[tuple[int, int]] = sorted(
            self._items_in_input_order,
            key=lambda pair: (-pair[1], pair[0]),
        )

    def solve(self, method: str | None = "first fit") -> None:
        if method is None:
            method = "first fit"
        normalized_method = self._normalize_method(method)

        match normalized_method:
            case "next fit" | "nf":
                self._final_solution = self._next_fit(self._items_in_input_order)

            case "next fit decreasing" | "nfd":
                self._final_solution = self._next_fit(self._items_in_descending)

            case "first fit" | "ff":
                self._final_solution = self._first_fit(self._items_in_input_order)

            case "first fit decreasing" | "ffd":
                self._final_solution = self._first_fit(self._items_in_descending)

            case "best fit" | "bf":
                self._final_solution = self._best_fit(self._items_in_input_order)

            case "best fit decreasing" | "bfd":
                self._final_solution = self._best_fit(self._items_in_descending)

            case "worst fit" | "wf":
                self._final_solution = self._worst_fit(self._items_in_input_order)

            case "worst fit decreasing" | "wfd":
                self._final_solution = self._worst_fit(self._items_in_descending)

            case "relocation":
                self._relocation()

            case "swap":
                self._swap()

            case _:
                raise ValueError(
                    "Unsupported method. Available methods: next fit (nf), first fit "
                    "(ff), best fit (bf), worst fit (wf), next fit decreasing (nfd), "
                    "first fit decreasing (ffd), best fit decreasing (bfd), worst fit "
                    "decreasing (wfd), relocation, swap."
                )

    def _normalize_method(self, method: str) -> str:
        normalized = method.strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    def _next_fit(self, items: list[tuple[int, int]]) -> BinPackingSolution:
        """Next Fit: keep a single open bin and open a new one when necessary."""
        bin_loads: list[int] = []
        assignments: list[list[int]] = []

        current_bin_index = -1

        for item_index, size in items:
            if current_bin_index == -1 or (
                bin_loads[current_bin_index] + size > self._bin_capacity
            ):
                current_bin_index += 1
                bin_loads.append(size)
                assignments.append([item_index])
                continue

            bin_loads[current_bin_index] += size
            assignments[current_bin_index].append(item_index)

        return self._build_solution(bin_loads, assignments)

    def _first_fit(self, items: list[tuple[int, int]]) -> BinPackingSolution:
        """First Fit: place each item in the first bin that can hold it."""
        bin_loads: list[int] = []
        assignments: list[list[int]] = []

        for item_index, size in items:
            target_bin_index: int | None = None

            # Scan bins in order and pick the first one that fits.
            for bin_index, load in enumerate(bin_loads):
                if load + size <= self._bin_capacity:
                    target_bin_index = bin_index
                    break

            if target_bin_index is None:
                bin_loads.append(size)
                assignments.append([item_index])
            else:
                bin_loads[target_bin_index] += size
                assignments[target_bin_index].append(item_index)

        return self._build_solution(bin_loads, assignments)

    def _best_fit(self, items: list[tuple[int, int]]) -> BinPackingSolution:
        """Best Fit: choose the bin with the least remaining capacity after placement."""
        bin_loads: list[int] = []
        assignments: list[list[int]] = []

        for item_index, size in items:
            best_bin_index: int | None = None
            smallest_remaining_capacity = self._bin_capacity + 1

            for bin_index, load in enumerate(bin_loads):
                new_load = load + size
                if new_load > self._bin_capacity:
                    continue

                remaining_capacity = self._bin_capacity - new_load
                if remaining_capacity < smallest_remaining_capacity:
                    smallest_remaining_capacity = remaining_capacity
                    best_bin_index = bin_index

            if best_bin_index is None:
                bin_loads.append(size)
                assignments.append([item_index])
            else:
                bin_loads[best_bin_index] += size
                assignments[best_bin_index].append(item_index)

        return self._build_solution(bin_loads, assignments)

    def _worst_fit(self, items: list[tuple[int, int]]) -> BinPackingSolution:
        """Worst Fit: choose the bin with the most remaining capacity after placement."""
        bin_loads: list[int] = []
        assignments: list[list[int]] = []

        for item_index, size in items:
            worst_bin_index: int | None = None
            largest_remaining_capacity = -1

            for bin_index, load in enumerate(bin_loads):
                new_load = load + size
                if new_load > self._bin_capacity:
                    continue

                remaining_capacity = self._bin_capacity - new_load
                if remaining_capacity > largest_remaining_capacity:
                    largest_remaining_capacity = remaining_capacity
                    worst_bin_index = bin_index

            if worst_bin_index is None:
                bin_loads.append(size)
                assignments.append([item_index])
            else:
                bin_loads[worst_bin_index] += size
                assignments[worst_bin_index].append(item_index)

        return self._build_solution(bin_loads, assignments)

    def _apply_relocation(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> tuple[list[int], list[list[int]]]:
        """Attempt to empty lightly loaded bins by relocating their items elsewhere."""
        improved = True
        while improved:
            improved = False

            # Attempt bins from lightest to heaviest to maximize the chance of removal.
            source_order = sorted(
                range(len(assignments)),
                key=lambda index: bin_loads[index],
            )

            for source_bin_index in source_order:
                if not assignments[source_bin_index]:
                    continue

                # Work on a temporary copy so the state can be reverted if relocation fails.
                temporary_loads = list(bin_loads)
                temporary_assignments = [list(items) for items in assignments]
                all_items_placed = True

                for item_index in list(temporary_assignments[source_bin_index]):
                    item_size = self._item_sizes[item_index]
                    item_placed = False

                    # Move each item to the first bin that can accept it.
                    for target_bin_index in range(len(temporary_loads)):
                        if target_bin_index == source_bin_index:
                            continue
                        if not temporary_assignments[target_bin_index]:
                            continue
                        if (
                            temporary_loads[target_bin_index] + item_size
                            <= self._bin_capacity
                        ):
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
        """Improve a First-Fit Decreasing solution by repeatedly relocating items from light bins."""
        base_solution = self._first_fit(self._items_in_descending)
        bin_loads = list(base_solution.final_bin_loads)
        assignments = [
            list(base_solution.bin_assignments[i])
            for i in range(base_solution.total_bins_used)
        ]

        bin_loads, assignments = self._apply_relocation(bin_loads, assignments)
        self._final_solution = self._build_solution(bin_loads, assignments)

    def _swap(self) -> None:
        """Try pairwise swaps plus relocation to reduce the number of bins."""
        self._relocation()

        assert self._final_solution is not None
        bin_loads = list(self._final_solution.final_bin_loads)
        assignments = [
            list(self._final_solution.bin_assignments[i])
            for i in range(self._final_solution.total_bins_used)
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
                            if (
                                new_load_a > self._bin_capacity
                                or new_load_b > self._bin_capacity
                            ):
                                continue

                            # Apply the swap to a temporary state to test for improvement.
                            candidate_loads = list(bin_loads)
                            candidate_assignments = [
                                list(items) for items in assignments
                            ]
                            candidate_loads[bin_a_index] = new_load_a
                            candidate_loads[bin_b_index] = new_load_b
                            candidate_assignments[bin_a_index].remove(item_a_index)
                            candidate_assignments[bin_a_index].append(item_b_index)
                            candidate_assignments[bin_b_index].remove(item_b_index)
                            candidate_assignments[bin_b_index].append(item_a_index)

                            # Relocation can empty a bin after a beneficial swap.
                            candidate_loads, candidate_assignments = (
                                self._apply_relocation(
                                    candidate_loads, candidate_assignments
                                )
                            )
                            candidate_bin_count = sum(
                                1 for items in candidate_assignments if items
                            )

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
        """Remove empty bins and rebuild the solution with contiguous indices."""
        non_empty_bins = [
            (load, items) for load, items in zip(bin_loads, assignments) if items
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
