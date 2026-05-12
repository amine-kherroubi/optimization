from .base import CategoryAdapter

POPULATION_STRATEGY = CategoryAdapter(
    "population_based_metaheuristics",
    ("genetic algorithm", "ga", "particle swarm optimization", "pso", "ant colony optimization", "aco"),
    "4_population_based_metaheuristics/solver.py",
)
