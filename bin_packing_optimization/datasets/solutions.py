from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InstanceSolution:
    """Best-known solution metadata for one benchmark instance."""

    name: str
    best_lb: int
    best_ub: int
    status: str
    selected: bool


_DATASET_TO_FILE: dict[str, str] = {
    "ani-ai-ai": "AI.csv",
    "ani-ai-ani": "ANI.csv",
    "falkenauer-t": "Falkenauer.csv",
    "falkenauer-u": "Falkenauer.csv",
    "gi": "GI.csv",
    "hard28": "Hard28.csv",
    "randomly-generated": "Randomly_Generated.csv",
    "scholl-1": "Scholl.csv",
    "scholl-2": "Scholl.csv",
    "scholl-3": "Scholl.csv",
    "schwerin-1": "Schwerin.csv",
    "schwerin-2": "Schwerin.csv",
    "wäscher": "Wäscher.csv",
}


@lru_cache(maxsize=None)
def _dataset_solutions(dataset_key: str) -> dict[str, InstanceSolution]:
    solutions_dir = Path(__file__).resolve().parent / "__________solutions"
    filename = _DATASET_TO_FILE.get(dataset_key)
    if filename is None:
        return {}

    filepath = solutions_dir / filename
    if not filepath.exists():
        return {}

    table: dict[str, InstanceSolution] = {}
    with filepath.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["Name"].strip()
            table[name] = InstanceSolution(
                name=name,
                best_lb=int(row["Best LB"]),
                best_ub=int(row["Best UB"]),
                status=row["Status"].strip(),
                selected=row.get("Selected", "0").strip() == "1",
            )
    return table


def get_instance_solution(dataset_key: str, instance_name: str) -> InstanceSolution | None:
    """Return best-known solution information for a given dataset instance."""
    candidate_names = {instance_name, f"{instance_name}.txt"}
    solutions = _dataset_solutions(dataset_key)
    for candidate in candidate_names:
        if candidate in solutions:
            return solutions[candidate]
    return None
