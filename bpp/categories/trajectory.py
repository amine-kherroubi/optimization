from .base import CategoryAdapter

TRAJECTORY_STRATEGY = CategoryAdapter(
    "trajectory_based_metaheuristics",
    ("simulated annealing", "sa", "tabu search", "ts", "iterated local search", "ils"),
    "3_trajectory_based_metaheuristics/solver.py",
)
