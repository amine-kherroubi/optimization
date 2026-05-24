# Bin Packing Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Project-0A66C2?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A research repository for benchmarking algorithm families on the 1D Bin Packing Problem.

## Repository Structure

```text
.
├── bin_packing_optimization/
│   ├── datasets/
│   ├── exact_methods/
│   ├── specific_heuristics/
│   ├── trajectory_based_metaheuristics/
│   ├── population_based_metaheuristics/
│   ├── hybrid_learning_metaheuristics/
│   │   └── hybrid_alns/
│   │       ├── models/
│   │       └── repair_model_training/
│   └── utilities/
│       ├── benchmarking.py
│       ├── graphing.py
│       └── statistics.py
└── results/
```

## Solver Families

| Family                          | Path                               |
| ------------------------------- | ---------------------------------- |
| Exact methods                   | `exact_methods/`                   |
| Specific heuristics             | `specific_heuristics/`             |
| Trajectory-based metaheuristics | `trajectory_based_metaheuristics/` |
| Population-based metaheuristics | `population_based_metaheuristics/` |
| Hybrid learning metaheuristics  | `hybrid_learning_metaheuristics/`  |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
