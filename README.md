# Bin Packing Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Project-0A66C2?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A research-oriented Python repository for benchmarking exact, heuristic, and hybrid algorithms on the 1D Bin Packing Problem.

## Project overview

The package under `bin_packing_optimization` provides solver implementations, dataset parsers, and a reusable benchmarking harness. The repository is intended for reproducible experimentation rather than production deployment.

## Repository layout

```text
.
├── bin_packing_optimization/
│   ├── datasets/
│   ├── exact_methods/
│   ├── specific_heuristics/
│   ├── trajectory_based_metaheuristics/
│   ├── population_based_metaheuristics/
│   ├── learning_guided_metaheuristics/
│   │   └── hybrid_alns/
│   └── utilities/
├── pyproject.toml
├── requirements.txt
└── results/
```

## Solver families

| Family                          | Description                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------- |
| Exact methods                   | Optimal search strategies such as branch-and-bound and dynamic programming.      |
| Specific heuristics             | Fast constructive heuristics and local improvement procedures.                   |
| Trajectory-based metaheuristics | Single-solution methods such as simulated annealing and tabu search.             |
| Population-based metaheuristics | Population-driven methods such as genetic algorithms and ACO.                    |
| Learning-guided metaheuristics  | Hybrid ALNS pipelines that combine search heuristics with learned repair models. |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
import importlib

from bin_packing_optimization.utilities.benchmarking import create_benchmark

solver_module = importlib.import_module("bin_packing_optimization.exact_methods.solver")

benchmark = create_benchmark(
    dataset_key="falkenauer-t",
    solver_module=solver_module,
    time_limit=None,
)
benchmark.run(method="branch and bound")
benchmark.save_results_to_csv()
```

Benchmark outputs are written under the repository-wide `results/` tree, with CSV files grouped by dataset key and timestamp. Graphs are emitted alongside the CSV output when graph generation utilities are used.
