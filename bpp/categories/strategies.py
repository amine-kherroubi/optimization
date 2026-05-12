"""Central strategy assembly for all five educational categories."""

from __future__ import annotations

from bpp.core.interfaces import SolverStrategy

from .exact import EXACT_STRATEGY
from .heuristics import HEURISTICS_STRATEGY
from .hybrid import HYBRID_STRATEGY
from .population import POPULATION_STRATEGY
from .trajectory import TRAJECTORY_STRATEGY


def build_default_strategies() -> list[SolverStrategy]:
    return [
        EXACT_STRATEGY,
        HEURISTICS_STRATEGY,
        TRAJECTORY_STRATEGY,
        POPULATION_STRATEGY,
        HYBRID_STRATEGY,
    ]
