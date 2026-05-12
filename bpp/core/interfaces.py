from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ProblemInstance, Solution


class SolverStrategy(ABC):
    """Abstraction for any 1D-BPP algorithm implementation."""

    @property
    @abstractmethod
    def category(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def methods(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def solve(
        self, problem: ProblemInstance, method: str, **params: object
    ) -> Solution:
        raise NotImplementedError
