from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from bpp.core.interfaces import SolverStrategy
from bpp.core.models import ProblemInstance, Solution


def _load_solver(module_path: str) -> type:
    module = import_module(module_path)
    return module.BinPackingSolver


@dataclass(frozen=True, slots=True)
class _Adapter(SolverStrategy):
    _category: str
    _methods: tuple[str, ...]
    _solver_module: str

    @property
    def category(self) -> str:
        return self._category

    @property
    def methods(self) -> tuple[str, ...]:
        return self._methods

    def solve(self, problem: ProblemInstance, method: str, **params: object) -> Solution:
        solver_cls = _load_solver(self._solver_module)
        solver = solver_cls(problem.item_sizes, problem.bin_capacity)
        solver.solve(method, **params)
        result = solver.get_solution()
        return Solution(result.total_bins_used, result.bin_assignments, result.final_bin_loads)


def build_default_strategies() -> list[SolverStrategy]:
    return [
        _Adapter("exact_methods", ("backtracking", "branch and bound", "dynamic programming"), "1_exact_methods.solver"),
        _Adapter("specific_heuristics", ("next fit", "first fit", "best fit", "worst fit", "next fit decreasing", "first fit decreasing", "best fit decreasing", "worst fit decreasing", "relocation", "swap"), "2_specific_heuristics.solver"),
        _Adapter("trajectory_based_metaheuristics", ("simulated annealing", "sa", "tabu search", "ts", "iterated local search", "ils"), "3_trajectory_based_metaheuristics.solver"),
        _Adapter("population_based_metaheuristics", ("genetic algorithm", "ga", "particle swarm optimization", "pso", "ant colony optimization", "aco"), "4_population_based_metaheuristics.solver"),
        _Adapter("hybrid_ml_metaheuristics", ("hybrid alns", "alns"), "5_hybrid_ml_metaheuristics.solver"),
    ]
