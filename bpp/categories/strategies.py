from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import inspect
import sys

from bpp.core.interfaces import SolverStrategy
from bpp.core.models import ProblemInstance, Solution

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class _Adapter(SolverStrategy):
    _category: str
    _methods: tuple[str, ...]
    _solver_file: str
    _aliases: dict[str, str] = field(default_factory=dict)

    @property
    def category(self) -> str:
        return self._category

    @property
    def methods(self) -> tuple[str, ...]:
        return self._methods

    def solve(
        self, problem: ProblemInstance, method: str, **params: object
    ) -> Solution:
        normalized = _normalize(method)
        selected = self._aliases.get(normalized, normalized)
        if selected not in {_normalize(m) for m in self._methods}:
            allowed = ", ".join(self._methods)
            raise ValueError(
                f"Unsupported method '{method}' for '{self._category}'. Allowed: {allowed}"
            )

        solver_cls = _load_solver(self._solver_file)
        solver = solver_cls(problem.item_sizes, problem.bin_capacity)
        solve_signature = inspect.signature(solver.solve)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in solve_signature.parameters.values()
        )
        filtered_params = params if accepts_var_kwargs else {}
        solver.solve(selected, **filtered_params)
        result = solver.get_solution()
        return Solution(
            result.total_bins_used, result.bin_assignments, result.final_bin_loads
        )


def _normalize(method: str) -> str:
    return " ".join(method.strip().lower().replace("_", " ").replace("-", " ").split())


_MODULE_CACHE: dict[str, ModuleType] = {}


def _load_solver(relative_solver_path: str) -> type:
    if relative_solver_path in _MODULE_CACHE:
        return _MODULE_CACHE[relative_solver_path].BinPackingSolver

    file_path = _PROJECT_ROOT / relative_solver_path
    module_name = (
        f"bpp_solver_{relative_solver_path.replace('/', '_').replace('.py', '')}"
    )
    spec = spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load solver module from {file_path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[relative_solver_path] = module
    return module.BinPackingSolver


def build_default_strategies() -> list[SolverStrategy]:
    return [
        _Adapter(
            "exact_methods",
            ("backtracking", "branch and bound", "dynamic programming"),
            "1_exact_methods/solver.py",
        ),
        _Adapter(
            "specific_heuristics",
            (
                "next fit",
                "first fit",
                "best fit",
                "worst fit",
                "next fit decreasing",
                "first fit decreasing",
                "best fit decreasing",
                "worst fit decreasing",
                "relocation",
                "swap",
            ),
            "2_specific_heuristics/solver.py",
            {
                "nf": "next fit",
                "ff": "first fit",
                "bf": "best fit",
                "wf": "worst fit",
                "nfd": "next fit decreasing",
                "ffd": "first fit decreasing",
                "bfd": "best fit decreasing",
                "wfd": "worst fit decreasing",
            },
        ),
        _Adapter(
            "trajectory_based_metaheuristics",
            (
                "simulated annealing",
                "sa",
                "tabu search",
                "ts",
                "iterated local search",
                "ils",
            ),
            "3_trajectory_based_metaheuristics/solver.py",
        ),
        _Adapter(
            "population_based_metaheuristics",
            (
                "genetic algorithm",
                "ga",
                "particle swarm optimization",
                "pso",
                "ant colony optimization",
                "aco",
            ),
            "4_population_based_metaheuristics/solver.py",
        ),
        _Adapter(
            "hybrid_ml_metaheuristics",
            ("hybrid alns", "alns"),
            "5_hybrid_ml_metaheuristics/solver.py",
        ),
    ]
