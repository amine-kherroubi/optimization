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
    """Population-based solvers for the 1D bin packing problem."""

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

    def solve(self, method: str, **params) -> None:
        normalized_method = self._normalize_method(method)

        match normalized_method:
            case "genetic algorithm" | "ga":
                if params:
                    raise ValueError("Parameter overrides are only supported for ACO.")
                self._genetic_algorithm()

            case "genetic algorithm memetic" | "ga memetic":
                if params:
                    raise ValueError("Parameter overrides are only supported for ACO.")
                self._genetic_algorithm(memetic=True)

            case "genetic algorithm island" | "ga island":
                if params:
                    raise ValueError("Parameter overrides are only supported for ACO.")
                self._genetic_algorithm_island()

            case "ant colony optimization" | "aco":
                self._ant_colony_optimization(**params)

            case _:
                raise ValueError(
                    "Unsupported method. Available methods: genetic algorithm (ga), "
                    "genetic algorithm memetic (ga memetic), genetic algorithm island "
                    "(ga island), ant colony optimization (aco)."
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

    @staticmethod
    def _normalize_method(method: str) -> str:
        normalized = method.strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    def _random_individual(self) -> tuple[list[int], list[list[int]]]:
        order = list(range(len(self._item_sizes)))
        random.shuffle(order)
        bin_loads: list[int] = []
        assignments: list[list[int]] = []

        for item_index in order:
            size = self._item_sizes[item_index]
            placed = False
            for bin_index, load in enumerate(bin_loads):
                if load + size <= self._bin_capacity:
                    bin_loads[bin_index] += size
                    assignments[bin_index].append(item_index)
                    placed = True
                    break
            if not placed:
                bin_loads.append(size)
                assignments.append([item_index])

        return bin_loads, assignments

    def _tournament_select(
        self,
        population: list[tuple[list[int], list[list[int]]]],
        scores: list[int],
        k: int,
    ) -> tuple[list[int], list[list[int]]]:
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best = min(candidates, key=lambda i: scores[i])
        individual = population[best]
        return list(individual[0]), [list(b) for b in individual[1]]

    def _ubx_crossover(
        self,
        parent_a: tuple[list[int], list[list[int]]],
        parent_b: tuple[list[int], list[list[int]]],
    ) -> tuple[list[int], list[list[int]]]:
        total_items = len(self._item_sizes)
        placed = [False] * total_items
        child_loads: list[int] = []
        child_assignments: list[list[int]] = []

        all_bins = list(parent_a[1]) + list(parent_b[1])
        random.shuffle(all_bins)

        for bin_items in all_bins:
            if all(not placed[item] for item in bin_items):
                load = sum(self._item_sizes[item] for item in bin_items)
                if load <= self._bin_capacity:
                    child_loads.append(load)
                    child_assignments.append(list(bin_items))
                    for item in bin_items:
                        placed[item] = True

        unplaced = sorted(
            [
                (item, self._item_sizes[item])
                for item in range(total_items)
                if not placed[item]
            ],
            key=lambda x: -x[1],
        )

        for item_index, size in unplaced:
            target: int | None = None
            for bin_index, load in enumerate(child_loads):
                if load + size <= self._bin_capacity:
                    target = bin_index
                    break
            if target is None:
                child_loads.append(size)
                child_assignments.append([item_index])
            else:
                child_loads[target] += size
                child_assignments[target].append(item_index)

        return child_loads, child_assignments

    def _ga_local_search(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
        steps: int,
    ) -> tuple[list[int], list[list[int]]]:
        current_loads = list(bin_loads)
        current_assignments = [list(b) for b in assignments]
        current_score = self._score(current_loads)

        for _ in range(steps):
            neighbor_loads, neighbor_assignments = self._generate_neighbor(
                current_loads, current_assignments
            )
            neighbor_score = self._score(neighbor_loads)
            if neighbor_score < current_score:
                current_loads = neighbor_loads
                current_assignments = neighbor_assignments
                current_score = neighbor_score

        return current_loads, current_assignments

    def _genetic_algorithm(
        self,
        *,
        population_size: int = 60,
        generations: int = 400,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.20,
        elite_count: int = 4,
        tournament_k: int = 4,
        memetic: bool = False,
        local_search_steps: int = 80,
    ) -> None:
        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        population: list[tuple[list[int], list[list[int]]]] = [
            self._simulate_initial_solution()
        ]
        while len(population) < population_size:
            population.append(self._random_individual())

        scores = [self._score(individual[0]) for individual in population]
        best_idx = min(range(population_size), key=lambda i: scores[i])
        best_loads = list(population[best_idx][0])
        best_assignments = [list(b) for b in population[best_idx][1]]
        best_score = scores[best_idx]

        for _ in range(generations):
            ranked = sorted(zip(scores, population), key=lambda x: x[0])
            new_population: list[tuple[list[int], list[list[int]]]] = []
            new_scores: list[int] = []

            for score, individual in ranked[:elite_count]:
                new_population.append(
                    (list(individual[0]), [list(b) for b in individual[1]])
                )
                new_scores.append(score)

            while len(new_population) < population_size:
                parent_a = self._tournament_select(population, scores, tournament_k)
                parent_b = self._tournament_select(population, scores, tournament_k)

                if random.random() < crossover_rate:
                    child_loads, child_assignments = self._ubx_crossover(
                        parent_a, parent_b
                    )
                else:
                    parent = (
                        parent_a
                        if self._score(parent_a[0]) <= self._score(parent_b[0])
                        else parent_b
                    )
                    child_loads, child_assignments = list(parent[0]), [
                        list(b) for b in parent[1]
                    ]

                if random.random() < mutation_rate:
                    child_loads, child_assignments = self._generate_neighbor(
                        child_loads,
                        child_assignments,
                    )

                if memetic:
                    child_loads, child_assignments = self._ga_local_search(
                        child_loads,
                        child_assignments,
                        steps=local_search_steps,
                    )

                child_score = self._score(child_loads)
                new_population.append((child_loads, child_assignments))
                new_scores.append(child_score)

                if child_score < best_score:
                    best_score = child_score
                    best_loads = list(child_loads)
                    best_assignments = [list(b) for b in child_assignments]

            population = new_population
            scores = new_scores

        self._final_solution = self._build_solution(best_loads, best_assignments)

    def _genetic_algorithm_island(
        self,
        *,
        num_islands: int = 4,
        island_size: int = 20,
        generations: int = 400,
        migration_interval: int = 40,
        migration_rate: int = 2,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.25,
        elite_count: int = 2,
        tournament_k: int = 3,
    ) -> None:
        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        islands: list[list[tuple[list[int], list[list[int]]]]] = []
        island_scores: list[list[int]] = []

        for island_index in range(num_islands):
            population: list[tuple[list[int], list[list[int]]]] = []
            if island_index == 0:
                population.append(self._simulate_initial_solution())
            while len(population) < island_size:
                population.append(self._random_individual())
            islands.append(population)
            island_scores.append(
                [self._score(individual[0]) for individual in population]
            )

        best_score = math.inf
        best_loads: list[int] = []
        best_assignments: list[list[int]] = []

        for population, scores in zip(islands, island_scores):
            local_best = min(range(island_size), key=lambda i: scores[i])
            if scores[local_best] < best_score:
                best_score = scores[local_best]
                best_loads = list(population[local_best][0])
                best_assignments = [list(b) for b in population[local_best][1]]

        for generation in range(1, generations + 1):
            for island_index in range(num_islands):
                population = islands[island_index]
                scores = island_scores[island_index]
                ranked = sorted(zip(scores, population), key=lambda x: x[0])
                new_population: list[tuple[list[int], list[list[int]]]] = []
                new_scores: list[int] = []

                for score, individual in ranked[:elite_count]:
                    new_population.append(
                        (list(individual[0]), [list(b) for b in individual[1]])
                    )
                    new_scores.append(score)

                while len(new_population) < island_size:
                    parent_a = self._tournament_select(population, scores, tournament_k)
                    parent_b = self._tournament_select(population, scores, tournament_k)

                    if random.random() < crossover_rate:
                        child_loads, child_assignments = self._ubx_crossover(
                            parent_a, parent_b
                        )
                    else:
                        parent = (
                            parent_a
                            if self._score(parent_a[0]) <= self._score(parent_b[0])
                            else parent_b
                        )
                        child_loads, child_assignments = list(parent[0]), [
                            list(b) for b in parent[1]
                        ]

                    if random.random() < mutation_rate:
                        child_loads, child_assignments = self._generate_neighbor(
                            child_loads,
                            child_assignments,
                        )

                    child_score = self._score(child_loads)
                    new_population.append((child_loads, child_assignments))
                    new_scores.append(child_score)

                    if child_score < best_score:
                        best_score = child_score
                        best_loads = list(child_loads)
                        best_assignments = [list(b) for b in child_assignments]

                islands[island_index] = new_population
                island_scores[island_index] = new_scores

            if generation % migration_interval == 0:
                emigrants: list[list[tuple[list[int], list[list[int]]]]] = []
                for island_index in range(num_islands):
                    population = islands[island_index]
                    scores = island_scores[island_index]
                    ranked_idx = sorted(range(island_size), key=lambda i: scores[i])
                    emigrants.append(
                        [
                            (
                                list(population[i][0]),
                                [list(b) for b in population[i][1]],
                            )
                            for i in ranked_idx[:migration_rate]
                        ]
                    )

                for island_index in range(num_islands):
                    target = (island_index + 1) % num_islands
                    population = islands[target]
                    scores = island_scores[target]
                    worst_idx = sorted(
                        range(island_size),
                        key=lambda i: scores[i],
                        reverse=True,
                    )[:migration_rate]
                    for rank, slot in enumerate(worst_idx):
                        incoming = emigrants[island_index][rank]
                        population[slot] = incoming
                        scores[slot] = self._score(incoming[0])

        self._final_solution = self._build_solution(best_loads, best_assignments)

    def _ant_colony_optimization(
        self,
        *,
        num_ants: int = 30,
        generations: int = 300,
        alpha: float = 1.0,  # pheromone influence exponent
        beta: float = 3.0,  # heuristic influence exponent
        evaporation_rate: float = 0.2,
        pheromone_init: float = 1.0,
        pheromone_min: float = 0.01,
        pheromone_max: float = 6.0,
        elite_ants: int = 3,
        local_search_steps: int = 60,
        deposit_strength: float = 1000.0,
    ) -> None:
        self._validate_aco_parameters(
            num_ants=num_ants,
            generations=generations,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            pheromone_init=pheromone_init,
            pheromone_min=pheromone_min,
            pheromone_max=pheromone_max,
            elite_ants=elite_ants,
            local_search_steps=local_search_steps,
            deposit_strength=deposit_strength,
        )

        if not self._item_sizes:
            self._final_solution = BinPackingSolution(0, {}, [])
            return

        n = len(self._item_sizes)

        # τ[i][j]: how desirable it is to pack item j right after item i.
        # Uniform start means no ordering is initially preferred over another.
        pheromone: list[list[float]] = [[pheromone_init] * n for _ in range(n)]

        # η[j] = size[j] / capacity — larger items score higher, nudging ants
        # toward big-items-first orderings (the FFD intuition).
        heuristic: list[float] = [s / self._bin_capacity for s in self._item_sizes]

        # Seed global best from FFD so pheromone updates are meaningful from
        # generation 1 rather than wasted on purely random early solutions.
        seed_loads, seed_assignments = self._simulate_initial_solution()
        best_score: float = self._score(seed_loads)
        best_loads: list[int] = list(seed_loads)
        best_assignments: list[list[int]] = [list(b) for b in seed_assignments]
        best_sequence: list[int] = self._aco_assignments_to_sequence(seed_assignments)

        for _ in range(generations):
            # Each ant independently builds one complete solution this generation.
            generation_results: list[
                tuple[float, list[int], list[list[int]], list[int]]
            ] = []

            for _ in range(num_ants):
                sequence = self._aco_build_sequence(pheromone, heuristic, alpha, beta)
                loads, assignments = self._aco_first_fit(sequence)
                generation_results.append(
                    (self._score(loads), loads, assignments, sequence)
                )

            generation_results.sort(key=lambda x: x[0])

            # Polish the iteration-best ant with local search before depositing
            # pheromone — sharper solutions produce sharper reinforcement signals.
            best_iter_score, best_iter_loads, best_iter_assignments, best_iter_seq = (
                generation_results[0]
            )
            refined_loads, refined_assignments = self._aco_local_search(
                best_iter_loads, best_iter_assignments, local_search_steps
            )
            if (refined_score := self._score(refined_loads)) < best_iter_score:
                best_iter_seq = self._aco_assignments_to_sequence(refined_assignments)
                generation_results[0] = (
                    refined_score,
                    refined_loads,
                    refined_assignments,
                    best_iter_seq,
                )

            if generation_results[0][0] < best_score:
                best_score, best_loads, best_assignments, best_sequence = (
                    generation_results[0][0],
                    list(generation_results[0][1]),
                    [list(b) for b in generation_results[0][2]],
                    list(generation_results[0][3]),
                )

            # Evaporate all edges: τ[i][j] *= (1 − ρ), clamped to τ_min.
            # Old information fades so the colony can redirect toward better paths.
            for i in range(n):
                for j in range(n):
                    pheromone[i][j] = max(
                        pheromone[i][j] * (1.0 - evaporation_rate), pheromone_min
                    )

            # Global-best deposit: reinforce the all-time best ordering every
            # generation. Deposit amount scales with solution quality and keeps
            # reinforcement visible despite the large objective value.
            for k in range(len(best_sequence) - 1):
                i, j = best_sequence[k], best_sequence[k + 1]
                pheromone[i][j] = min(
                    pheromone[i][j] + deposit_strength / best_score,
                    pheromone_max,
                )

            # Iteration-best deposit: a few runners-up also reinforce their edges,
            # injecting diversity so the colony doesn't converge prematurely.
            for score, _, _, sequence in generation_results[1 : elite_ants + 1]:
                for k in range(len(sequence) - 1):
                    i, j = sequence[k], sequence[k + 1]
                    pheromone[i][j] = min(
                        pheromone[i][j] + deposit_strength / score,
                        pheromone_max,
                    )

        self._final_solution = self._build_solution(best_loads, best_assignments)

    @staticmethod
    def _validate_aco_parameters(
        *,
        num_ants: int,
        generations: int,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        pheromone_init: float,
        pheromone_min: float,
        pheromone_max: float,
        elite_ants: int,
        local_search_steps: int,
        deposit_strength: float,
    ) -> None:
        if num_ants <= 0:
            raise ValueError("num_ants must be positive.")
        if generations < 0:
            raise ValueError("generations must be non-negative.")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")
        if beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if not 0.0 <= evaporation_rate < 1.0:
            raise ValueError("evaporation_rate must be in [0, 1).")
        if pheromone_min <= 0.0:
            raise ValueError("pheromone_min must be positive.")
        if pheromone_max < pheromone_min:
            raise ValueError("pheromone_max must be at least pheromone_min.")
        if not pheromone_min <= pheromone_init <= pheromone_max:
            raise ValueError("pheromone_init must be within the pheromone bounds.")
        if not 0 <= elite_ants < num_ants:
            raise ValueError("elite_ants must be between 0 and num_ants - 1.")
        if local_search_steps < 0:
            raise ValueError("local_search_steps must be non-negative.")
        if deposit_strength <= 0.0:
            raise ValueError("deposit_strength must be positive.")

    def _aco_build_sequence(
        self,
        pheromone: list[list[float]],
        heuristic: list[float],
        alpha: float,
        beta: float,
    ) -> list[int]:
        unvisited = list(range(len(self._item_sizes)))

        # Random start keeps different ants exploring different orderings.
        current = random.choice(unvisited)
        unvisited.remove(current)
        sequence = [current]

        while unvisited:
            # Combined attractiveness: τ[current][j]^α × η[j]^β
            weights = [
                (pheromone[current][j] ** alpha) * (heuristic[j] ** beta)
                for j in unvisited
            ]

            total = sum(weights)
            if total == 0.0:
                chosen = random.choice(unvisited)
            else:
                # Roulette-wheel selection: items with higher attractiveness
                # cover a proportionally larger arc of [0, total].
                threshold = random.random() * total
                cumulative = 0.0
                chosen = unvisited[-1]
                for item, weight in zip(unvisited, weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        chosen = item
                        break

            sequence.append(chosen)
            unvisited.remove(chosen)
            current = chosen

        return sequence

    def _aco_first_fit(
        self,
        sequence: list[int],
    ) -> tuple[list[int], list[list[int]]]:
        # Pack items in the ant's chosen order using First-Fit:
        # place each item into the first bin that has room, or open a new one.
        bin_loads: list[int] = []
        assignments: list[list[int]] = []

        for item_index in sequence:
            size = self._item_sizes[item_index]
            placed = False
            for bin_index, load in enumerate(bin_loads):
                if load + size <= self._bin_capacity:
                    bin_loads[bin_index] += size
                    assignments[bin_index].append(item_index)
                    placed = True
                    break
            if not placed:
                bin_loads.append(size)
                assignments.append([item_index])

        return bin_loads, assignments

    def _aco_assignments_to_sequence(
        self,
        assignments: list[list[int]],
    ) -> list[int]:
        # Flatten bins into a sequence, largest item first within each bin,
        # so the sequence can be used as a pheromone deposit path.
        sequence: list[int] = []
        for bin_items in assignments:
            sequence.extend(sorted(bin_items, key=lambda i: -self._item_sizes[i]))
        return sequence

    def _aco_local_search(
        self,
        bin_loads: list[int],
        assignments: list[list[int]],
        steps: int,
    ) -> tuple[list[int], list[list[int]]]:
        current_loads = list(bin_loads)
        current_assignments = [list(b) for b in assignments]
        current_score = self._score(current_loads)

        for _ in range(steps):
            neighbor_loads, neighbor_assignments = self._generate_neighbor(
                current_loads, current_assignments
            )
            neighbor_score = self._score(neighbor_loads)
            if neighbor_score < current_score:
                current_loads = neighbor_loads
                current_assignments = neighbor_assignments
                current_score = neighbor_score

        return current_loads, current_assignments

    def get_solution(self) -> BinPackingSolution:
        if self._final_solution is None:
            raise RuntimeError(
                "No solution available: you must call 'solve()' before retrieving the solution."
            )

        return self._final_solution
