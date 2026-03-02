# Optimization Labs Repository

## Bin Packing Problem

### Commands to run the Falkenauer benchmark

All commands must be run from the **project root**.

```shell
# All instances, T variant (default)
python3 -m bin_packing.benchmark

# Only 120-item instances, U variant
python3 -m bin_packing.benchmark --variant U --num-items 120

# All instances up to 249 items, T variant
python3 -m bin_packing.benchmark --variant T --max-items 249

# Run with a strict time limit per instance (e.g., 10 seconds)
python3 -m bin_packing.benchmark --time-limit 10

# Skip graph generation
python3 -m bin_packing.benchmark --no-graphs
```

#### Options

| Flag           | Values   | Default | Description                                  |
| -------------- | -------- | ------- | -------------------------------------------- |
| `--variant`    | `T`, `U` | `T`     | Dataset variant: T (triplets) or U (uniform) |
| `--num-items`  | integer  | —       | Run only instances with **exactly** N items  |
| `--max-items`  | integer  | —       | Run only instances with **at most** N items  |
| `--time-limit` | float    | —       | Maximum solve time (in seconds) per instance |
| `--no-graphs`  | flag     | off     | Skip graph generation after the run          |

`--num-items` and `--max-items` are mutually exclusive.

#### Output

Results are printed as a table with columns: `Instance`, `Items`, `Capacity`, `LB` (theoretical lower bound), `Bins`, `Gap` (bins − LB), `Time (s)`, `Method`, and `State` (`Done` or `T.O.` for timeouts). A summary block reports total time, average bins, the number of provably optimal solutions (gap = 0), and the number of timeouts (if any).

Unless `--no-graphs` is passed, four PNG graphs are saved to `results/`, each suffixed with the variant letter:

| File                        | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `fig1_time_vs_n_<V>.png`    | Solve time vs number of items, colored by time                 |
| `fig2_bins_vs_lb_<V>.png`   | Bins used vs lower bound, with gap highlighted in red          |
| `fig3_fill_rate_<V>.png`    | Average bin fill rate per instance on a red-yellow-green scale |
| `fig4_time_by_size_<V>.png` | Box plot of solve times grouped by instance size               |
