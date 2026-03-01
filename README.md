# Optimization Labs Repository

## Bin Packing Problem

### Commands to run the Falkenauer benchmark

All commands must be run from the **project root**.

```shell
# All instances, T variant (default)
python3 -m bin_packing.falkenauer_benchmark

# Only 120-item instances, U variant
python3 -m bin_packing.falkenauer_benchmark --variant U --num-items 120

# All instances up to 249 items, T variant
python3 -m bin_packing.falkenauer_benchmark --variant T --max-items 249
```

#### Options

| Flag          | Values   | Default | Description                                  |
| ------------- | -------- | ------- | -------------------------------------------- |
| `--variant`   | `T`, `U` | `T`     | Dataset variant: T (random) or U (uniform)   |
| `--num-items` | integer  | —       | Run only instances with **exactly** N items  |
| `--max-items` | integer  | —       | Run only instances with **at most** N items  |

`--num-items` and `--max-items` are mutually exclusive.
