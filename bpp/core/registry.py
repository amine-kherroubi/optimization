from __future__ import annotations

from dataclasses import dataclass

from .interfaces import SolverStrategy


@dataclass(slots=True)
class StrategyRegistry:
    _by_category: dict[str, SolverStrategy]

    def __init__(self) -> None:
        self._by_category = {}

    def register(self, strategy: SolverStrategy) -> None:
        key = strategy.category.strip().lower()
        if key in self._by_category:
            raise ValueError(f"Category already registered: {strategy.category}")
        self._by_category[key] = strategy

    def get_by_category(self, category: str) -> SolverStrategy:
        key = category.strip().lower()
        if key not in self._by_category:
            available = ", ".join(sorted(self._by_category))
            raise ValueError(f"Unknown category '{category}'. Available: {available}")
        return self._by_category[key]

    def all_methods(self) -> dict[str, tuple[str, ...]]:
        return {k: v.methods for k, v in self._by_category.items()}
