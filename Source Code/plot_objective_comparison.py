import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------
# Output directory
# -------------------------
os.makedirs("results/plots", exist_ok=True)

# -------------------------
# Load all PnL distributions
# -------------------------
pnl_var = torch.load("results/pnl_variance_deep.pt").cpu().numpy()
pnl_entropic = torch.load("results/pnl_entropic_deep.pt").cpu().numpy()
pnl_cvar = torch.load("results/pnl_deep.pt").cpu().numpy()
pnl_delta = torch.load("results/pnl_variance_delta.pt").cpu().numpy()

# -------------------------
# Plot 1: Overlayed PnL Distributions
# -------------------------
plt.figure(figsize=(9, 6))
plt.hist(pnl_delta, bins=120, density=True, alpha=0.4, label="Delta Hedge")
plt.hist(pnl_var, bins=120, density=True, alpha=0.5, label="Deep Hedge (Variance)")
plt.hist(pnl_entropic, bins=120, density=True, alpha=0.5, label="Deep Hedge (Entropic)")
plt.hist(pnl_cvar, bins=120, density=True, alpha=0.5, label="Deep Hedge (CVaR)")
plt.xlabel("PnL")
plt.ylabel("Density")
plt.legend()
plt.title("PnL Distribution Comparison Across Hedging Objectives")
plt.tight_layout()
plt.savefig("results/plots/pnl_overlay_all_objectives.png", dpi=300)
plt.close()

# -------------------------
# Plot 2: CVaR(95%) Comparison
# -------------------------
q = 0.05

cvar_delta = pnl_delta[pnl_delta <= np.quantile(pnl_delta, q)].mean()
cvar_var = pnl_var[pnl_var <= np.quantile(pnl_var, q)].mean()
cvar_entropic = pnl_entropic[pnl_entropic <= np.quantile(pnl_entropic, q)].mean()
cvar_cvar = pnl_cvar[pnl_cvar <= np.quantile(pnl_cvar, q)].mean()

labels = [
    "Delta",
    "Variance",
    "Entropic",
    "CVaR"
]
values = [
    cvar_delta,
    cvar_var,
    cvar_entropic,
    cvar_cvar
]

plt.figure(figsize=(7, 5))
plt.bar(labels, values)
plt.ylabel("CVaR (95%)")
plt.title("CVaR Comparison Across Hedging Objectives")
plt.tight_layout()
plt.savefig("results/plots/cvar_comparison_all_objectives.png", dpi=300)
plt.close()

# -------------------------
# Plot 3: Variance Comparison
# -------------------------
var_delta = np.var(pnl_delta)
var_var = np.var(pnl_var)
var_entropic = np.var(pnl_entropic)
var_cvar = np.var(pnl_cvar)

labels = [
    "Delta",
    "Variance",
    "Entropic",
    "CVaR"
]
values = [
    var_delta,
    var_var,
    var_entropic,
    var_cvar
]

plt.figure(figsize=(7, 5))
plt.bar(labels, values)
plt.ylabel("PnL Variance")
plt.title("Variance Comparison Across Hedging Objectives")
plt.tight_layout()
plt.savefig("results/plots/variance_comparison_all_objectives.png", dpi=300)
plt.close()

print("Objective comparison plots generated successfully.")
