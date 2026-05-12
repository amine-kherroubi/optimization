from __future__ import annotations

from .models import ProblemInstance, SolveRequest, Solution
from .registry import StrategyRegistry


class BinPackingService:
    def __init__(self, registry: StrategyRegistry):
        self._registry = registry

    def solve(self, problem: ProblemInstance, request: SolveRequest) -> Solution:
        strategy = self._registry.get_by_category(request.category)
        params = request.params or {}
        return strategy.solve(problem, request.method, **params)
