import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------
# Output directory
# -------------------------
os.makedirs("results/plots", exist_ok=True)

# -------------------------
# Bootstrap results (from your run)
# -------------------------
methods = [
    "Variance Hedge",
    "Entropic Hedge",
    "CVaR Hedge"
]

mean_diff = np.array([
    20.98,   # Variance vs Delta
    23.25,   # Entropic vs Delta
    22.98    # CVaR vs Delta
])

ci_lower = np.array([
    19.58,
    21.22,
    20.96
])

ci_upper = np.array([
    22.32,
    25.17,
    24.79
])

# Compute error bars
yerr = np.vstack([
    mean_diff - ci_lower,
    ci_upper - mean_diff
])

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    mean_diff,
    yerr=yerr,
    capsize=8,
    alpha=0.8
)

plt.axhline(0, color="black", linewidth=1, linestyle="--")

plt.ylabel("CVaR Improvement vs Delta Hedge")
plt.title("Bootstrap 95% Confidence Intervals for CVaR Improvement")

plt.tight_layout()
plt.savefig("results/plots/cvar_confidence_intervals.png", dpi=300)
plt.close()

print("CVaR confidence interval plot generated successfully.")
