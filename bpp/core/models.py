from __future__ import annotations

"""Domain models used across the API, registry, and service layers."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProblemInstance:
    """Input instance for 1D-BPP."""

    item_sizes: list[int]
    bin_capacity: int


@dataclass(frozen=True, slots=True)
class Solution:
    """Normalized solution shape returned by every solver family."""

    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[int]


@dataclass(frozen=True, slots=True)
class SolveRequest:
    """Request contract from API layer to service layer."""

    category: str
    method: str
    params: dict[str, object] | None = None
