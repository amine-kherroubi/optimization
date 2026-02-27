from amine import BinPacking

bp = BinPacking([5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12], 20)
bp.solve("b&b")
solution = bp.get_solution()

print("Bins used:", solution.bins_used)
for bin_index, contents in solution.assignments.items():
    print(f"Bin {bin_index}: {sorted(contents)} (load = {solution.loads[bin_index]})")
