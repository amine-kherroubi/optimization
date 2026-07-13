"""Generate the ablation figure (Fig. 2) from the aligned 5000-iter i9 run."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["No\nlearning", "Online\nonly", "Offline\nonly", "Both\ncombined"]
avg_gap = [0.05, 0.05, 0.05, 0.05]
avg_time = [3.136, 3.208, 8.954, 8.950]

x = np.arange(len(methods))
w = 0.38

fig, ax = plt.subplots(figsize=(3.4, 2.3))
b1 = ax.bar(
    x - w / 2,
    avg_gap,
    w,
    label="Avg gap (bins)",
    color="#2b6cb0",
    edgecolor="black",
    linewidth=0.5,
)
b2 = ax.bar(
    x + w / 2,
    avg_time,
    w,
    label="Avg time (s)",
    color="#cbd5e0",
    edgecolor="black",
    linewidth=0.5,
)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel("Value", fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, 10.0)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )

fig.tight_layout(pad=0.4)
fig.savefig("ablation.png", dpi=300)
print("wrote ablation.png")
