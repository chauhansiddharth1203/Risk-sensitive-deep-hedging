"""Trade-off plot for the regime-gate threshold sweep."""
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results/report", exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10})

# (threshold, window) -> (mean, std)
data = {
    "2024 OOT":  {2.0: (-0.322, 0.207), 2.5: (+0.555, 0.165),
                  3.0: (+1.112, 0.103), 3.5: (+1.165, 0.058)},
    "2020 COVID": {2.0: (-3.091, 0.113), 2.5: (-2.077, 0.273),
                   3.0: (-1.178, 0.411), 3.5: (-1.961, 1.481)},
    "2017 calm":  {2.0: (-2.670, 0.348), 2.5: (-1.387, 0.245),
                   3.0: (-0.349, 0.265), 3.5: (+0.100, 0.198)},
    "2008 GFC":   {2.0: (-2.440, 0.148), 2.5: (-2.181, 0.133),
                   3.0: (-2.351, 0.468), 3.5: (-3.848, 1.836)},
}
# Plain-Adam reference numbers
plain_adam = {
    "2024 OOT":  (+1.160, 0.060),
    "2020 COVID": (-8.050, 4.960),
    "2017 calm":  (+0.480, 0.140),
    "2008 GFC":   (-25.000, 4.000),  # approx -- placeholder for the bar
}

thresholds = [2.0, 2.5, 3.0, 3.5]
colours = {"2024 OOT": "#2E7D32", "2020 COVID": "#C62828",
           "2017 calm": "#1565C0", "2008 GFC": "#6A1B9A"}

fig, ax = plt.subplots(figsize=(9.5, 5.5))
for win, pts in data.items():
    means = [pts[t][0] for t in thresholds]
    stds  = [pts[t][1] for t in thresholds]
    ax.errorbar(thresholds, means, yerr=stds,
                marker="o", capsize=4, lw=1.6, ms=7,
                label=win, color=colours[win])

# Highlight thr=3.0 as the sweet spot
ax.axvline(3.0, ls="--", color="black", lw=1, alpha=0.5)
ax.text(3.02, ax.get_ylim()[0] + 0.3, "sweet spot",
        fontsize=9, style="italic", color="black")

ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(thresholds)
ax.set_xlabel("Frozen gate threshold (normalised VIX units)")
ax.set_ylabel("CVaR$_{95}$ gap Delta = Deep - BS  (mean +/- seed-std, 5 seeds)")
ax.set_title("Regime-gate threshold sweep -- calm-recovery / crisis-stability trade-off")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(alpha=0.25)
ax.set_ylim(-4.2, 2.0)
plt.tight_layout()
plt.savefig("results/report/gate_threshold_sweep.png",
            dpi=170, bbox_inches="tight")
plt.savefig("results/report/gate_threshold_sweep.pdf",
            bbox_inches="tight")
plt.close()
print("Saved -> results/report/gate_threshold_sweep.{png,pdf}")
