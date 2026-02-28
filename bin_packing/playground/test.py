from rayan import BinPacking

sizes: list[int] = [
    5,
    5,
    6,
    6,
    7,
    7,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
    12,
    1,
    8,
    6,
    12,
    19,
    20,
    7,
    7,
    3,
    3,
    3,
]
bin_capacity: int = 20

bp = BinPacking(sizes, 20)
bp.solve("b&b")
solution = bp.get_solution()

print("Sorted items:", sizes)
print("Bins used:", solution.bins_used)
for bin_index, contents in solution.assignments.items():
    print(f"Bin {bin_index}: {sorted(contents)} (load = {solution.loads[bin_index]})")
