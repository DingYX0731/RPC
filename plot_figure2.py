"""
Plot AIME InternLM2-Math-Plus-7B: Accuracy (%) vs #Sampling n for PPL, SC, PC, RPC.
"""
import matplotlib.pyplot as plt
import numpy as np

# X: sampling sizes (display labels); plot at equal spacing
n_labels = np.array([1, 2, 4, 8, 16, 32, 40, 48, 64, 80, 96, 112, 128])
x_pos = np.arange(len(n_labels))  # 0, 1, 2, ... equal spacing

# Approximate data from the figure (a) OlympiadBench
ppl = np.array([5.06, 5.66, 5.93, 6.14, 6.21, 6.31, 6.21, 6.37, 6.48, 6.41, 6.28, 6.11, 5.96])
sc = np.array([5.06, 5.19, 6.23, 7.49, 8.51, 8.89, 9.04, 9.05, 9.24, 9.33, 9.31, 9.39, 9.40])
pc = np.array([5.06, 5.66, 6.60, 7.80, 8.76, 9.04, 9.22, 9.20, 9.40, 9.42, 9.43, 9.56, 9.45])
rpc = np.array([5.06, 5.66, 6.55, 7.86, 8.71, 9.08, 9.37, 9.50, 9.53, 9.69, 9.87, 9.69, 9.74])

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel("#Sampling n")
ax.set_ylabel("Accuracy (%)")
ax.set_title("AIME InternLM2-Math-Plus-7B", fontsize=12)
ax.set_xlim(-0.5, len(n_labels) - 0.5)
# ylim from data (was 41.5–52.5 for MATH; MathOdyssey data is ~7–11)
y_min = min(ppl.min(), sc.min(), pc.min(), rpc.min())
y_max = max(ppl.max(), sc.max(), pc.max(), rpc.max())
ax.set_ylim(y_min - 1, y_max + 1)
ax.set_xticks(x_pos)
ax.set_xticklabels(n_labels)
ax.grid(True, color="gray", alpha=0.4, linestyle="-")

ax.plot(x_pos, ppl, color="C0", marker="^", markersize=6, label="PPL", linewidth=2)
ax.plot(x_pos, sc, color="C2", marker="o", markersize=6, label="SC", linewidth=2)
ax.plot(x_pos, pc, color="C1", marker="D", markersize=5, label="PC", linewidth=2)
ax.plot(x_pos, rpc, color="C3", marker="s", markersize=5, label="RPC", linewidth=2)

ax.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig("figure/figure2_AIME.png", dpi=150, bbox_inches="tight")
plt.savefig("figure/figure2_AIME.pdf", bbox_inches="tight")
print("Saved figure/figure2_AIME.png and figure/figure2_AIME.pdf")
plt.show()

    