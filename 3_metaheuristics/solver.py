from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(slots=True)
class BinPackingSolution:
    """Compact representation of a bin packing solution."""

    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[int]


class BinPackingSolver:
    """Metaheuristic solvers for the 1D bin packing problem."""

    __slots__ = (
        "_item_sizes",
        "_bin_capacity",
        "_final_solution",
        "_items_in_descending",
    )

    def __init__(self, item_sizes: list[int], bin_capacity: int):
        if any(size > bin_capacity for size in item_sizes):
            raise ValueError("No single item size can exceed the bin capacity.")

        self._item_sizes = list(item_sizes)
        self._bin_capacity = bin_capacity
        self._final_solution: BinPackingSolution | None = None
        self._items_in_descending = sorted(
            enumerate(self._item_sizes), key=lambda pair: (-pair[1], pair[0])
        )

    def solve(self, method: str) -> None:
        normalized_method = self._normalize_method(method)

        match normalized_method:
            case "simulated annealing" | "sa":
                self._simulated_annealing()

            case "simulated annealing reheating" | "sa reheating":
                self._simulated_annealing(reheating=True)

            case "simulated annealing adaptive" | "sa adaptive":
                self._simulated_annealing(adaptive=True)

            case _:
                raise ValueError(
                    "Unsupported method. Available methods: "
                    "simulated annealing (sa), simulated annealing reheating "
                    "(sa reheating), simulated annealing adaptive (sa adaptive)."
                )

    def _simulate_initial_solution(self) -> tuple[list[int], list[list[int]]]:
        bin_loads: list[int] = []
        assignments: list[list[int]] = []

        for item_index, size in self._items_in_descending:
            target_bin_index: int | None = None
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

        return bin_loads, assignments

    def _simulated_annealing(
        self,
        reheating: bool = False,
        adaptive: bool = False,
    ) -> None:
        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        current_loads, current_assignments = self._simulate_initial_solution()
        current_cost = self._score(current_loads)
        best_loads = list(current_loads)
        best_assignments = [list(items) for items in current_assignments]
        best_cost = current_cost

        temperature = self._initial_temperature(current_loads)
        cooling_rate = 0.995
        min_temperature = 1e-4
        stagnation = 0
        acceptance_history: list[bool] = []

        for iteration in range(1, 25_001):
            candidate_loads, candidate_assignments = self._generate_neighbor(
                current_loads, current_assignments
            )
            candidate_cost = self._score(candidate_loads)

            if self._should_accept(candidate_cost, current_cost, temperature):
                current_loads = candidate_loads
                current_assignments = candidate_assignments
                current_cost = candidate_cost
                acceptance_history.append(True)
            else:
                acceptance_history.append(False)

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_loads = list(candidate_loads)
                best_assignments = [list(items) for items in candidate_assignments]
                stagnation = 0
            else:
                stagnation += 1

            if adaptive and iteration % 200 == 0:
                acceptance_ratio = sum(acceptance_history[-200:]) / 200
                temperature *= 1.0 - (0.1 * (acceptance_ratio - 0.3))
                temperature = max(temperature, min_temperature)

            if reheating and stagnation >= 600:
                temperature = max(
                    temperature * 0.45, self._initial_temperature(best_loads)
                )
                stagnation = 0

            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature

        self._final_solution = self._build_solution(best_loads, best_assignments)

    def _generate_neighbor(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> tuple[list[int], list[list[int]]]:
        candidate_loads = list(bin_loads)
        candidate_assignments = [list(items) for items in assignments]

        if len(candidate_assignments) < 2 or random.random() < 0.6:
            return self._relocate_item(candidate_loads, candidate_assignments)

        return self._swap_items(candidate_loads, candidate_assignments)

    def _relocate_item(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> tuple[list[int], list[list[int]]]:
        source_bin = random.randrange(len(assignments))
        while not assignments[source_bin]:
            source_bin = random.randrange(len(assignments))

        item_index = random.choice(assignments[source_bin])
        item_size = self._item_sizes[item_index]

        target_bins = [i for i in range(len(assignments)) if i != source_bin]
        random.shuffle(target_bins)

        for target_bin in target_bins:
            if bin_loads[target_bin] + item_size <= self._bin_capacity:
                assignments[source_bin].remove(item_index)
                assignments[target_bin].append(item_index)
                bin_loads[source_bin] -= item_size
                bin_loads[target_bin] += item_size
                break
        else:
            assignments[source_bin].remove(item_index)
            assignments.append([item_index])
            bin_loads[source_bin] -= item_size
            bin_loads.append(item_size)

        return self._clean_bins(bin_loads, assignments)

    def _swap_items(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> tuple[list[int], list[list[int]]]:
        non_empty = [i for i, items in enumerate(assignments) if items]
        if len(non_empty) < 2:
            return self._relocate_item(bin_loads, assignments)

        bin_a, bin_b = random.sample(non_empty, 2)
        item_a = random.choice(assignments[bin_a])
        item_b = random.choice(assignments[bin_b])

        size_a = self._item_sizes[item_a]
        size_b = self._item_sizes[item_b]

        if (
            bin_loads[bin_a] - size_a + size_b <= self._bin_capacity
            and bin_loads[bin_b] - size_b + size_a <= self._bin_capacity
        ):
            assignments[bin_a].remove(item_a)
            assignments[bin_b].remove(item_b)
            assignments[bin_a].append(item_b)
            assignments[bin_b].append(item_a)
            bin_loads[bin_a] += size_b - size_a
            bin_loads[bin_b] += size_a - size_b

        return self._clean_bins(bin_loads, assignments)

    def _clean_bins(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> tuple[list[int], list[list[int]]]:
        compact_loads: list[int] = []
        compact_assignments: list[list[int]] = []

        for load, items in zip(bin_loads, assignments):
            if items:
                compact_loads.append(load)
                compact_assignments.append(items)

        return compact_loads, compact_assignments

    def _score(self, bin_loads: list[int]) -> int:
        return len([load for load in bin_loads if load > 0]) * 10_000 + sum(
            (self._bin_capacity - load) ** 2 for load in bin_loads if load > 0
        )

    def _initial_temperature(self, bin_loads: list[int]) -> float:
        average_load = sum(bin_loads) / len(bin_loads)
        return max(average_load, 1.0) * 5.0

    def _should_accept(
        self,
        candidate_cost: int,
        current_cost: int,
        temperature: float,
    ) -> bool:
        if candidate_cost <= current_cost:
            return True

        return random.random() < math.exp(
            -(candidate_cost - current_cost) / max(temperature, 1e-8)
        )

    def _normalize_method(self, method: str) -> str:
        normalized = method.strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    def _build_solution(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> BinPackingSolution:
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
