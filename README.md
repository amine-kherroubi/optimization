# Optimization Labs Repository

## Commands to run the Falkenauer benchmark

```shell
# All instances, T variant (default)
python3 -m bin_packing.falkenauer_benchmark.py

# Only 120-item instances, U variant
python3 -m bin_packing.falkenauer_benchmark.py --variant U --num-items 120

# All instances up to 249 items, T variant
python3 -m bin_packing.falkenauer_benchmark.py --variant T --max-items 249
```
