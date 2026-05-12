from __future__ import annotations

from bpp import solve


def run() -> None:
    item_sizes = [4, 8, 1, 4, 2, 1]
    bin_capacity = 10

    solution = solve(
        item_sizes=item_sizes,
        bin_capacity=bin_capacity,
        category="specific_heuristics",
        method="first fit decreasing",
    )

    print(f"Bins used: {solution.total_bins_used}")
    print(f"Bin loads: {solution.final_bin_loads}")


if __name__ == "__main__":
    run()
