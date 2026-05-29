"""Single-session reproducer for Table II (ablation) and Table IV (comparison).

Runs every configuration and baseline on the same Scholl-2 50-item x5 slice in one
process on the documented hardware, so the hybrid's numbers are identical across
both tables and all runtimes are directly comparable.
"""
import statistics as st
import warnings

from bin_packing_optimization.utilities.benchmarking import create_benchmark
from bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.models import (
    load_repair_model,
)
import bin_packing_optimization.currently_irrelevant.specific_heuristics.solver as heur
import bin_packing_optimization.currently_irrelevant.trajectory_based_metaheuristics.solver as traj
import bin_packing_optimization.currently_irrelevant.population_based_metaheuristics.solver as pop
import bin_packing_optimization.hybrid_learning_metaheuristics.hybrid_alns.hybrid_alns_solver as hybrid

warnings.filterwarnings("ignore")

DATASET, NUM_ITEMS, MAX = "scholl-2", 50, 5
BASE = {"max_iterations": 300, "initial_temperature": 1.4426950408889634, "alpha_cool": 0.9995}


def run(module, method, args=None):
    b = create_benchmark(DATASET, module)
    b.run(method=method, method_args=args, num_items=NUM_ITEMS, max_instances=MAX)
    res = b.get_results()
    bins = st.mean(r.bins_used for r in res)
    gap = st.mean(r.bins_used - r.lower_bound for r in res)
    fill = st.mean(100.0 * r.total_weight / (r.bins_used * r.bin_capacity) for r in res)
    t = st.mean(r.elapsed_time for r in res)
    return bins, gap, fill, t


def model():
    return load_repair_model("repair_model_v2.pkl")


def main():
    abl = [
        ("No learning", run(hybrid, "no-learning", {**BASE, "use_offline_model": False, "use_online_rl": False})),
        ("Online only", run(hybrid, "online-only", {**BASE, "use_offline_model": False, "use_online_rl": True})),
        ("Offline only", run(hybrid, "offline-only", {**BASE, "use_offline_model": True, "use_online_rl": False, "model_bundle": model()})),
        ("Both combined", run(hybrid, "both-combined", {**BASE, "use_offline_model": True, "use_online_rl": True, "model_bundle": model()})),
    ]
    cmp = [
        ("FFD", run(heur, "first fit decreasing")),
        ("BFD", run(heur, "best fit decreasing")),
        ("Simulated annealing", run(traj, "simulated annealing")),
        ("Tabu search", run(traj, "tabu search")),
        ("Genetic algorithm", run(pop, "genetic algorithm")),
        ("Ant colony", run(pop, "ant colony optimization")),
        ("Dual-learning ALNS", run(hybrid, "combined", {**BASE, "use_offline_model": True, "use_online_rl": True, "model_bundle": model()})),
    ]

    print("\n\n===== TABLE II (ablation) =====")
    print(f"{'Config':<16}{'AvgBins':>8}{'AvgGap':>8}{'Fill%':>8}{'Time(s)':>9}")
    for lab, (bn, gp, fl, tm) in abl:
        print(f"{lab:<16}{bn:>8.1f}{gp:>8.2f}{fl:>8.2f}{tm:>9.3f}")

    print("\n===== TABLE IV (comparison) =====")
    print(f"{'Method':<22}{'AvgGap':>8}{'Time(s)':>9}")
    for lab, (bn, gp, fl, tm) in cmp:
        print(f"{lab:<22}{gp:>8.2f}{tm:>9.3f}")


if __name__ == "__main__":
    main()
