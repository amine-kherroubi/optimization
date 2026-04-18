from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(slots=True)
class _TabuMove:
    """Move descriptor used by tabu-search neighborhoods."""

    kind: str
    item_a: int
    source_a: int
    target_a: int
    item_b: int | None = None
    source_b: int | None = None
    target_b: int | None = None


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

            case "tabu search" | "ts":
                self._tabu_search()

            case "reactive tabu search" | "rts":
                self._tabu_search(reactive=True)

            case "tabu search lns" | "ts lns" | "lnts":
                self._tabu_search(reactive=True, lns=True)

            case "tabu search diversified" | "ts diversified" | "hybrid tabu":
                self._tabu_search(reactive=True, lns=True, diversification=True)

            case _:
                raise ValueError(
                    "Unsupported method. Available methods: "
                    "simulated annealing (sa), simulated annealing reheating "
                    "(sa reheating), simulated annealing adaptive (sa adaptive), "
                    "tabu search (ts), reactive tabu search (rts), tabu search lns "
                    "(ts lns, lnts), tabu search diversified (ts diversified, hybrid tabu)."
                )

# =============================
# simulated annealing section

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

# end of simulated annealing section
# ======================================


# ======================================
# tabu search section 
    def _tabu_search(
        self,
        reactive: bool = False,
        lns: bool = False,
        diversification: bool = False,
    ) -> None:
        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        item_count = len(self._item_sizes)

        current_loads, current_assignments = self._simulate_initial_solution()
        current_base_cost = self._score(current_loads)
        lower_bound = math.ceil(sum(self._item_sizes) / self._bin_capacity)

        best_loads = list(current_loads)
        best_assignments = [list(items) for items in current_assignments]
        best_base_cost = current_base_cost

        base_tenure = max(4, int(math.sqrt(item_count) * 0.8))
        min_tenure = 2
        max_tenure = max(10, base_tenure * 4)
        tabu_tenure = base_tenure

        if reactive or lns or diversification:
            max_iterations = min(12_000, max(2_200, item_count * 32))
            max_candidates = min(88, max(22, item_count // 3))
            patience = max(520, item_count * 9)
        else:
            max_iterations = min(8_000, max(1_500, item_count * 20))
            max_candidates = min(64, max(16, item_count // 4))
            patience = max(360, item_count * 7)

        no_improve = 0

        tabu_until: dict[tuple[int, int], int] = {}
        seen_signatures: dict[int, int] = {}
        move_frequency: dict[tuple[int, int], int] = {}

        lns_period = 450
        lns_stagnation_trigger = 260
        long_stagnation_trigger = 500
        diversification_weight = 0.0

        for iteration in range(1, max_iterations + 1):
            if diversification:
                diversification_weight = (
                    0.0
                    if no_improve < long_stagnation_trigger
                    else min(20.0, 0.03 * (no_improve - long_stagnation_trigger + 1))
                )

            move, candidate_loads, candidate_assignments, candidate_cost = (
                self._select_best_tabu_neighbor(
                    current_loads=current_loads,
                    current_assignments=current_assignments,
                    tabu_until=tabu_until,
                    move_frequency=move_frequency,
                    iteration=iteration,
                    tabu_tenure=tabu_tenure,
                    best_global_cost=best_base_cost,
                    diversification_weight=diversification_weight,
                    max_candidates=max_candidates,
                )
            )

            current_loads = candidate_loads
            current_assignments = candidate_assignments
            current_base_cost = candidate_cost
            self._update_tabu_memory(tabu_until, move, iteration, tabu_tenure)
            if diversification:
                self._update_frequency_memory(move_frequency, move)

            improved = False
            if current_base_cost < best_base_cost:
                best_base_cost = current_base_cost
                best_loads = list(current_loads)
                best_assignments = [list(items) for items in current_assignments]
                no_improve = 0
                improved = True
            else:
                no_improve += 1

            cycle_detected = False
            if reactive and iteration % 40 == 0:
                signature = self._state_signature(current_assignments)
                previous = seen_signatures.get(signature)
                cycle_detected = previous is not None and iteration - previous <= 250
                seen_signatures[signature] = iteration
                if len(seen_signatures) > 20_000:
                    stale_before = iteration - 2_000
                    seen_signatures = {
                        key: value
                        for key, value in seen_signatures.items()
                        if value >= stale_before
                    }

            if reactive:
                tabu_tenure = self._reactive_tabu_tenure(
                    current_tenure=tabu_tenure,
                    base_tenure=base_tenure,
                    min_tenure=min_tenure,
                    max_tenure=max_tenure,
                    stagnation=no_improve,
                    cycle_detected=cycle_detected,
                    improved=improved,
                )

            if lns and (iteration % lns_period == 0 or no_improve >= lns_stagnation_trigger):
                lns_loads, lns_assignments = self._lns_destroy_repair(
                    current_loads, current_assignments
                )
                lns_loads, lns_assignments = self._tabu_refine(
                    lns_loads,
                    lns_assignments,
                    refinement_iterations=max(12, item_count // 6),
                    max_candidates=max(10, max_candidates // 3),
                )

                lns_cost = self._score(lns_loads)
                if lns_cost <= current_base_cost:
                    current_loads = lns_loads
                    current_assignments = lns_assignments
                    current_base_cost = lns_cost

                if lns_cost < best_base_cost:
                    best_base_cost = lns_cost
                    best_loads = list(lns_loads)
                    best_assignments = [list(items) for items in lns_assignments]
                    no_improve = 0

            if len(best_loads) <= lower_bound:
                break
            if no_improve >= patience:
                break

        self._final_solution = self._build_solution(best_loads, best_assignments)

    def _select_best_tabu_neighbor(
        self,
        current_loads: list[int],
        current_assignments: list[list[int]],
        tabu_until: dict[tuple[int, int], int],
        move_frequency: dict[tuple[int, int], int],
        iteration: int,
        tabu_tenure: int,
        best_global_cost: int,
        diversification_weight: float,
        max_candidates: int,
    ) -> tuple[_TabuMove, list[int], list[list[int]], int]:
        del tabu_tenure  # Tenure is applied when updating memory, not while evaluating.

        moves = self._sample_tabu_moves(current_assignments, max_candidates)
        if not moves:
            fallback_move = _TabuMove("relocate", 0, 0, 0)
            return fallback_move, list(current_loads), [list(b) for b in current_assignments], self._score(current_loads)

        best_eval: float | None = None
        best_choice: tuple[_TabuMove, list[int], list[list[int]], int] | None = None
        best_inadmissible: tuple[_TabuMove, list[int], list[list[int]], int] | None = None
        best_inadmissible_eval: float | None = None

        for move in moves:
            candidate_loads, candidate_assignments = self._apply_move(
                current_loads, current_assignments, move
            )
            base_cost = self._score(candidate_loads)

            tabu = self._is_tabu(move, tabu_until, iteration)
            aspiration = base_cost < best_global_cost
            admissible = (not tabu) or aspiration

            penalty = 0.0
            if diversification_weight > 0.0:
                penalty = diversification_weight * self._move_frequency_penalty(
                    move, move_frequency
                )
            eval_cost = base_cost + penalty

            if admissible and (best_eval is None or eval_cost < best_eval):
                best_eval = eval_cost
                best_choice = (move, candidate_loads, candidate_assignments, base_cost)

            if best_inadmissible_eval is None or eval_cost < best_inadmissible_eval:
                best_inadmissible_eval = eval_cost
                best_inadmissible = (move, candidate_loads, candidate_assignments, base_cost)

        if best_choice is not None:
            return best_choice
        if best_inadmissible is not None:
            return best_inadmissible

        fallback_move = moves[0]
        fallback_loads, fallback_assignments = self._apply_move(
            current_loads, current_assignments, fallback_move
        )
        return fallback_move, fallback_loads, fallback_assignments, self._score(
            fallback_loads
        )

    def _sample_tabu_moves(
        self,
        assignments: list[list[int]],
        max_candidates: int,
    ) -> list[_TabuMove]:
        non_empty_bins = [idx for idx, items in enumerate(assignments) if items]
        if not non_empty_bins:
            return []

        sampled: list[_TabuMove] = []
        used_signatures: set[tuple[int, ...]] = set()

        relocate_budget = max(1, int(max_candidates * 0.75))
        swap_budget = max_candidates - relocate_budget
        relocate_attempts = 0
        max_relocate_attempts = max(20, relocate_budget * 8)

        while len(sampled) < relocate_budget and relocate_attempts < max_relocate_attempts:
            relocate_attempts += 1
            source_bin = random.choice(non_empty_bins)
            item = random.choice(assignments[source_bin])

            can_open_new_bin = len(assignments[source_bin]) > 1
            existing_targets = [i for i in range(len(assignments)) if i != source_bin]

            if can_open_new_bin and random.random() < 0.1:
                target_bin = len(assignments)
            elif existing_targets:
                target_bin = random.choice(existing_targets)
            elif can_open_new_bin:
                target_bin = len(assignments)
            else:
                continue

            signature = (0, item, source_bin, target_bin)
            if signature in used_signatures:
                continue
            used_signatures.add(signature)
            sampled.append(_TabuMove("relocate", item, source_bin, target_bin))

        if len(non_empty_bins) >= 2:
            swap_attempts = 0
            max_swap_attempts = max(20, swap_budget * 8)
            while (
                len(sampled) < relocate_budget + swap_budget
                and swap_attempts < max_swap_attempts
            ):
                swap_attempts += 1
                bin_a, bin_b = random.sample(non_empty_bins, 2)
                item_a = random.choice(assignments[bin_a])
                item_b = random.choice(assignments[bin_b])

                signature = (1, item_a, bin_a, item_b, bin_b)
                if signature in used_signatures:
                    continue
                used_signatures.add(signature)
                sampled.append(
                    _TabuMove(
                        "swap",
                        item_a=item_a,
                        source_a=bin_a,
                        target_a=bin_b,
                        item_b=item_b,
                        source_b=bin_b,
                        target_b=bin_a,
                    )
                )

        return sampled

    def _apply_move(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
        move: _TabuMove,
    ) -> tuple[list[int], list[list[int]]]:
        loads = list(bin_loads)
        bins = [list(items) for items in assignments]

        if move.kind == "relocate":
            source = move.source_a
            target = move.target_a
            item = move.item_a
            size = self._item_sizes[item]

            if source >= len(bins) or item not in bins[source]:
                return self._clean_bins(loads, bins)

            if target == len(bins):
                if len(bins[source]) <= 1:
                    return self._clean_bins(loads, bins)
                bins[source].remove(item)
                loads[source] -= size
                bins.append([item])
                loads.append(size)
                return self._clean_bins(loads, bins)

            if target >= len(bins) or target == source:
                return self._clean_bins(loads, bins)

            if loads[target] + size <= self._bin_capacity:
                bins[source].remove(item)
                bins[target].append(item)
                loads[source] -= size
                loads[target] += size

            return self._clean_bins(loads, bins)

        if move.kind == "swap":
            if move.item_b is None or move.source_b is None:
                return self._clean_bins(loads, bins)

            source_a = move.source_a
            source_b = move.source_b
            item_a = move.item_a
            item_b = move.item_b

            if source_a >= len(bins) or source_b >= len(bins):
                return self._clean_bins(loads, bins)
            if item_a not in bins[source_a] or item_b not in bins[source_b]:
                return self._clean_bins(loads, bins)

            size_a = self._item_sizes[item_a]
            size_b = self._item_sizes[item_b]

            if (
                loads[source_a] - size_a + size_b <= self._bin_capacity
                and loads[source_b] - size_b + size_a <= self._bin_capacity
            ):
                bins[source_a].remove(item_a)
                bins[source_b].remove(item_b)
                bins[source_a].append(item_b)
                bins[source_b].append(item_a)
                loads[source_a] += size_b - size_a
                loads[source_b] += size_a - size_b

            return self._clean_bins(loads, bins)

        return self._clean_bins(loads, bins)

    def _is_tabu(
        self,
        move: _TabuMove,
        tabu_until: dict[tuple[int, int], int],
        iteration: int,
    ) -> bool:
        if move.kind == "relocate":
            return tabu_until.get((move.item_a, move.target_a), -1) >= iteration

        if move.kind == "swap" and move.item_b is not None and move.target_b is not None:
            first = tabu_until.get((move.item_a, move.target_a), -1) >= iteration
            second = tabu_until.get((move.item_b, move.target_b), -1) >= iteration
            return first or second

        return False

    def _update_tabu_memory(
        self,
        tabu_until: dict[tuple[int, int], int],
        move: _TabuMove,
        iteration: int,
        tabu_tenure: int,
    ) -> None:
        expiry = iteration + tabu_tenure
        tabu_until[(move.item_a, move.source_a)] = expiry

        if move.kind == "swap" and move.item_b is not None and move.source_b is not None:
            tabu_until[(move.item_b, move.source_b)] = expiry

        if len(tabu_until) > 100_000:
            stale_before = iteration - tabu_tenure
            stale_keys = [key for key, value in tabu_until.items() if value < stale_before]
            for key in stale_keys:
                del tabu_until[key]

    def _update_frequency_memory(
        self,
        move_frequency: dict[tuple[int, int], int],
        move: _TabuMove,
    ) -> None:
        key_a = (move.item_a, move.target_a)
        move_frequency[key_a] = move_frequency.get(key_a, 0) + 1

        if move.kind == "swap" and move.item_b is not None and move.target_b is not None:
            key_b = (move.item_b, move.target_b)
            move_frequency[key_b] = move_frequency.get(key_b, 0) + 1

    def _move_frequency_penalty(
        self,
        move: _TabuMove,
        move_frequency: dict[tuple[int, int], int],
    ) -> int:
        penalty = move_frequency.get((move.item_a, move.target_a), 0)
        if move.kind == "swap" and move.item_b is not None and move.target_b is not None:
            penalty += move_frequency.get((move.item_b, move.target_b), 0)
        return penalty

    def _state_signature(self, assignments: list[list[int]]) -> int:
        item_to_bin = [0] * len(self._item_sizes)
        for bin_index, items in enumerate(assignments):
            for item in items:
                item_to_bin[item] = bin_index
        return hash(tuple(item_to_bin))

    def _reactive_tabu_tenure(
        self,
        current_tenure: int,
        base_tenure: int,
        min_tenure: int,
        max_tenure: int,
        stagnation: int,
        cycle_detected: bool,
        improved: bool,
    ) -> int:
        tenure = current_tenure

        if cycle_detected or stagnation >= 300:
            tenure = min(max_tenure, tenure + max(1, tenure // 10))
        elif improved:
            tenure = max(base_tenure, tenure - 1)
        elif stagnation <= 60:
            tenure = max(min_tenure, tenure - 1)

        return tenure

    def _lns_destroy_repair(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
    ) -> tuple[list[int], list[list[int]]]:
        loads = list(bin_loads)
        bins = [list(items) for items in assignments]

        removed_items = self._lns_destroy(loads, bins)
        if removed_items:
            self._lns_repair(loads, bins, removed_items)

        return self._clean_bins(loads, bins)

    def _lns_destroy(self, bin_loads: list[int], assignments: list[list[int]]) -> list[int]:
        item_count = len(self._item_sizes)
        remove_count = max(2, int(0.12 * item_count))
        remove_count = min(remove_count, item_count)

        removed: list[int] = []

        bin_order = list(range(len(assignments)))
        bin_order.sort(key=lambda idx: (self._bin_capacity - bin_loads[idx]), reverse=True)

        for bin_index in bin_order:
            if len(removed) >= remove_count:
                break
            random.shuffle(assignments[bin_index])
            while assignments[bin_index] and len(removed) < remove_count:
                item = assignments[bin_index].pop()
                removed.append(item)
                bin_loads[bin_index] -= self._item_sizes[item]

        return removed

    def _lns_repair(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
        removed_items: list[int],
    ) -> None:
        removed_items.sort(key=lambda item: self._item_sizes[item], reverse=True)

        for item in removed_items:
            size = self._item_sizes[item]
            best_bin: int | None = None
            best_residual: int | None = None

            for bin_index, load in enumerate(bin_loads):
                if load + size > self._bin_capacity:
                    continue
                residual = self._bin_capacity - (load + size)
                if best_residual is None or residual < best_residual:
                    best_residual = residual
                    best_bin = bin_index

            if best_bin is None:
                bin_loads.append(size)
                assignments.append([item])
            else:
                bin_loads[best_bin] += size
                assignments[best_bin].append(item)

    def _tabu_refine(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
        refinement_iterations: int,
        max_candidates: int,
    ) -> tuple[list[int], list[list[int]]]:
        current_loads = list(bin_loads)
        current_assignments = [list(items) for items in assignments]
        best_loads = list(current_loads)
        best_assignments = [list(items) for items in current_assignments]
        best_cost = self._score(current_loads)

        tabu_tenure = 4
        tabu_until: dict[tuple[int, int], int] = {}

        for iteration in range(1, refinement_iterations + 1):
            move, cand_loads, cand_assignments, cand_cost = self._select_best_tabu_neighbor(
                current_loads=current_loads,
                current_assignments=current_assignments,
                tabu_until=tabu_until,
                move_frequency={},
                iteration=iteration,
                tabu_tenure=tabu_tenure,
                best_global_cost=best_cost,
                diversification_weight=0.0,
                max_candidates=max_candidates,
            )

            current_loads = cand_loads
            current_assignments = cand_assignments
            self._update_tabu_memory(tabu_until, move, iteration, tabu_tenure)

            if cand_cost < best_cost:
                best_cost = cand_cost
                best_loads = list(cand_loads)
                best_assignments = [list(items) for items in cand_assignments]

        return best_loads, best_assignments


# end of tabu search section
# ===================================================
