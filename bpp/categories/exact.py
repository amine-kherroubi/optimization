from .base import CategoryAdapter

EXACT_STRATEGY = CategoryAdapter(
    "exact_methods",
    ("backtracking", "branch and bound", "dynamic programming"),
    "1_exact_methods/solver.py",
)
