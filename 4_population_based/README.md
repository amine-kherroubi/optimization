# Population-Based Solvers

This folder contains population-based methods for the 1D bin packing problem:

- `genetic algorithm` / `ga`
- `genetic algorithm memetic` / `ga memetic`
- `genetic algorithm island` / `ga island`
- `ant colony optimization` / `aco`

## Ant Colony Optimization Tuning

ACO parameters can be passed directly through `BinPackingSolver.solve()`:

```python
solver.solve(
    "aco",
    num_ants=20,
    generations=100,
    alpha=1.0,
    beta=4.0,
    evaporation_rate=0.2,
    elite_ants=3,
    local_search_steps=40,
    deposit_strength=1000.0,
)
```

The tuning helper starts from `ACO_DEFAULTS`, tries one parameter grid at a
time, keeps the best value, and writes one CSV row per tested value.

By default, it launches the full tuning suite: one global run and one run per
benchmark family.

```shell
venv/bin/python 4_population_based/tune_aco.py
```

Useful options:

```shell
venv/bin/python 4_population_based/tune_aco.py \
  --datasets falkenauer-t falkenauer-u scholl-1 scholl-2 scholl-3 \
  --max-items 250 \
  --max-per-size 1 \
  --seeds 0,1,2 \
  --rounds 1
```

To run one quick tuning job only, add `--single`:

```shell
venv/bin/python 4_population_based/tune_aco.py \
  --single \
  --datasets falkenauer-t \
  --seeds 0 \
  --parameters beta
```

Candidates are ranked by dataset-balanced average lower-bound gap, then by the
worst dataset gap, exact hits, and average runtime.

### Main ACO Parameters

| Parameter | Meaning |
| --- | --- |
| `num_ants` | Number of candidate solutions built per generation. |
| `generations` | Number of pheromone update cycles. |
| `alpha` | Strength of pheromone influence. |
| `beta` | Strength of item-size heuristic influence. |
| `evaporation_rate` | Fraction of pheromone removed each generation. |
| `elite_ants` | Number of iteration-best runners-up that also deposit pheromone. |
| `local_search_steps` | Improvement attempts applied to the best ant per generation. |
| `deposit_strength` | Multiplier for pheromone deposited by good solutions. |

Start with a small subset and one seed, then increase seeds and sampled
instances once the parameter ranges look reasonable.

The full suite writes separate CSV files:

```text
results/4_population_based/aco_tuning/aco_tuning_global.csv
results/4_population_based/aco_tuning/aco_tuning_falkenauer_t.csv
results/4_population_based/aco_tuning/aco_tuning_falkenauer_u.csv
results/4_population_based/aco_tuning/aco_tuning_scholl_1.csv
results/4_population_based/aco_tuning/aco_tuning_scholl_2.csv
results/4_population_based/aco_tuning/aco_tuning_scholl_3.csv
```

For a quick test run, tune only one or two parameters:

```shell
venv/bin/python 4_population_based/tune_aco.py \
  --seeds 0 \
  --parameters beta deposit_strength
```

## Notebook

The main notebook is:

```text
4_population_based/notebook.ipynb
```

It reads `results/4_population_based/aco_tuning/aco_tuning_*.csv`, reconstructs the
selected ACO parameter values, and then compares GA variants with the tuned ACO
configuration. If needed, it creates:

```text
results/4_population_based/aco_tuning/population_comparison.csv
```

The notebook builds tables, plots, and workflow diagrams. Its figures are saved
under:

```text
results/4_population_based/aco_tuning/figures/
```
