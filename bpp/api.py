from __future__ import annotations

"""High-level API facade for solving 1D bin packing instances."""

from bpp.categories.strategies import build_default_strategies
from bpp.core.models import ProblemInstance, SolveRequest, Solution
from bpp.core.registry import StrategyRegistry
from bpp.core.service import BinPackingService


def create_default_service() -> BinPackingService:
    """Build the default service with all five registered categories."""
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
    """Solve one instance through a unified category+method API.

    Params are validated by the selected category adapter.
    """
    service = create_default_service()
    problem = ProblemInstance(item_sizes=item_sizes, bin_capacity=bin_capacity)
    request = SolveRequest(category=category, method=method, params=params)
    return service.solve(problem, request)
