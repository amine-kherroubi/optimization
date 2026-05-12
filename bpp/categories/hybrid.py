from .base import CategoryAdapter

HYBRID_STRATEGY = CategoryAdapter(
    "hybrid_ml_metaheuristics",
    ("hybrid alns", "alns"),
    "5_hybrid_ml_metaheuristics/solver.py",
    _allowed_params=frozenset({"model_path", "max_iterations", "initial_temperature", "alpha_cool"}),
)
