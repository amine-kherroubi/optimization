from __future__ import annotations

from bpp import solve


def run() -> None:
    solution = solve(
        item_sizes=[4, 8, 1, 4, 2, 1],
        bin_capacity=10,
        category="specific_heuristics",
        method="first fit decreasing",
    )
    print(f"Bins used: {solution.total_bins_used}")


if __name__ == "__main__":
    run()
