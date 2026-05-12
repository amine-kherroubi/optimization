from __future__ import annotations

"""Category adapters that bridge the unified API to legacy solver implementations."""

from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
import inspect
from pathlib import Path
import sys
from types import ModuleType

from bpp.core.interfaces import SolverStrategy
from bpp.core.models import ProblemInstance, Solution

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODULE_CACHE: dict[str, ModuleType] = {}


def _normalize(token: str) -> str:
    return " ".join(token.strip().lower().replace("_", " ").replace("-", " ").split())


def _load_solver(relative_solver_path: str) -> type:
    if relative_solver_path in _MODULE_CACHE:
        return _MODULE_CACHE[relative_solver_path].BinPackingSolver
    file_path = _PROJECT_ROOT / relative_solver_path
    module_name = f"bpp_solver_{relative_solver_path.replace('/', '_').replace('.py', '')}"
    spec = spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load solver module from {file_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[relative_solver_path] = module
    return module.BinPackingSolver


@dataclass(frozen=True, slots=True)
class CategoryAdapter(SolverStrategy):
    """Reusable adapter for one category of solvers."""

    _category: str
    _methods: tuple[str, ...]
    _solver_file: str
    _aliases: dict[str, str] = field(default_factory=dict)
    _allowed_params: frozenset[str] = frozenset()

    @property
    def category(self) -> str:
        return self._category

    @property
    def methods(self) -> tuple[str, ...]:
        return self._methods

    def solve(self, problem: ProblemInstance, method: str, **params: object) -> Solution:
        selected = self._resolve_method(method)
        filtered_params = self._resolve_params(params)

        solver_cls = _load_solver(self._solver_file)
        solver = solver_cls(problem.item_sizes, problem.bin_capacity)
        solver.solve(selected, **filtered_params)
        result = solver.get_solution()
        return Solution(result.total_bins_used, result.bin_assignments, result.final_bin_loads)

    def _resolve_method(self, method: str) -> str:
        normalized = _normalize(method)
        selected = self._aliases.get(normalized, normalized)
        normalized_methods = {_normalize(candidate): candidate for candidate in self._methods}
        if selected not in normalized_methods:
            allowed = ", ".join(self._methods)
            raise ValueError(f"Unsupported method '{method}' for '{self._category}'. Allowed: {allowed}")
        return normalized_methods[selected]

    def _resolve_params(self, params: dict[str, object]) -> dict[str, object]:
        if not params:
            return {}

        # Validate caller-provided kwargs first (best DX: clear typo/unsupported hints).
        if self._allowed_params:
            unknown = sorted(set(params) - set(self._allowed_params))
            if unknown:
                allowed = ", ".join(sorted(self._allowed_params))
                raise ValueError(f"Unsupported params for '{self._category}': {unknown}. Allowed: {allowed}")

        solver_cls = _load_solver(self._solver_file)
        solve_signature = inspect.signature(solver_cls.solve)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in solve_signature.parameters.values())
        return dict(params) if accepts_var_kwargs else {}
