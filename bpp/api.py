from __future__ import annotations

from bpp.categories.strategies import build_default_strategies
from bpp.core.models import ProblemInstance, SolveRequest, Solution
from bpp.core.registry import StrategyRegistry
from bpp.core.service import BinPackingService


def create_default_service() -> BinPackingService:
    registry = StrategyRegistry()
    for strategy in build_default_strategies():
        registry.register(strategy)
    return BinPackingService(registry)


def solve(
    item_sizes: list[int],
    bin_capacity: int,
    category: str,
    method: str,
    **params: object,
) -> Solution:
    service = create_default_service()
    problem = ProblemInstance(item_sizes=item_sizes, bin_capacity=bin_capacity)
    request = SolveRequest(category=category, method=method, params=params)
    return service.solve(problem, request)
