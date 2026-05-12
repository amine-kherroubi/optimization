from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProblemInstance:
    item_sizes: list[int]
    bin_capacity: int


@dataclass(frozen=True, slots=True)
class Solution:
    total_bins_used: int
    bin_assignments: dict[int, list[int]]
    final_bin_loads: list[int]


@dataclass(frozen=True, slots=True)
class SolveRequest:
    category: str
    method: str
    params: dict[str, object] | None = None
