"""Pre-train the LinUCB bandit on a held-out training slice.

After training, the persisted state in ``models/bandit_state_v1.pkl`` is loaded
at execution time (via ``method_args["bandit_state"]``) so that the bandit
inherits its learned policy across instances instead of starting from scratch on
every solve.
"""

from __future__ import annotations

import os
import time
import warnings

warnings.filterwarnings("ignore")

from bin_packing_optimization.datasets.registry import DATASET_REGISTRY
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.models import (
    load_repair_model,
    save_bandit_state,
)
import bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver as hybrid

# Training slice — kept disjoint from the test slices used in the article so
# the bandit doesn't "see" its evaluation set. ``num_items`` chosen on the
# smaller side to keep one-instance training time reasonable.
TRAINING_SLICE = [
    # (dataset_key, num_items_filter, max_instances)
    ("scholl-2", 100, 60),     # 60 medium-size scholl instances
    ("scholl-2", 200, 30),     # 30 larger ones for variety
    ("falkenauer-t", 120, 30), # triplets, different structure
]

BASE_ARGS = {
    "max_iterations": 5000,
    "initial_temperature": 1.4426950408889634,
    "alpha_cool": 0.9995,
    "use_offline_model": True,
    "use_online_rl": True,
}

OUTPUT = os.path.join(os.path.dirname(__file__), "models", "bandit_state_v1.pkl")


def _load_instances(dataset_key, num_items, max_instances):
    cfg = DATASET_REGISTRY[dataset_key]
    instances = []
    for path in cfg.directory.glob(cfg.glob):
        try:
            inst = cfg.parser(path, cfg.key)
        except (ValueError, IndexError):
            continue
        if num_items is not None and inst.num_items != num_items:
            continue
        instances.append(inst)
    instances.sort(key=lambda i: (i.num_items, i.name))
    return instances[:max_instances]


def main():
    repair_model = load_repair_model("repair_model_v2.pkl")
    state = None  # cold start
    total_instances = 0
    t_start = time.perf_counter()

    for dataset_key, num_items, max_instances in TRAINING_SLICE:
        instances = _load_instances(dataset_key, num_items, max_instances)
        print(
            f"\n=== {dataset_key} (num_items={num_items}, {len(instances)} inst) ==="
        )
        for inst in instances:
            args = {**BASE_ARGS, "model_bundle": repair_model}
            if state is not None:
                args["bandit_state"] = state

            t0 = time.perf_counter()
            solver = hybrid.BinPackingSolver(inst.sizes, inst.bin_capacity)
            solver.solve(**args)
            state = solver.get_bandit_state()
            total_instances += 1

            elapsed = time.perf_counter() - t0
            print(
                f"  [{total_instances:>3}] {inst.name:<22} "
                f"bins={solver.get_solution().total_bins_used} "
                f"calls={state['calls']:>5}  t={elapsed:>6.2f}s"
            )

    wall = time.perf_counter() - t_start
    save_bandit_state(state, OUTPUT)
    print(f"\nSaved {OUTPUT}")
    print(f"  trained on {total_instances} instances, total wall {wall:.1f}s")
    print(f"  final bandit calls = {state['calls']}")
    print(f"  arm_counts (random/worst/related) = {state['arm_counts']}")


if __name__ == "__main__":
    main()
