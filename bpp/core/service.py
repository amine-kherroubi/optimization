from __future__ import annotations

"""Application service coordinating strategy lookup and solve execution."""

from .models import ProblemInstance, SolveRequest, Solution
from .registry import StrategyRegistry


class BinPackingService:
    """Thin orchestrator: route request to the correct category strategy."""

    def __init__(self, registry: StrategyRegistry):
        self._registry = registry

    def solve(self, problem: ProblemInstance, request: SolveRequest) -> Solution:
        strategy = self._registry.get_by_category(request.category)
        params = request.params or {}
        return strategy.solve(problem, request.method, **params)
