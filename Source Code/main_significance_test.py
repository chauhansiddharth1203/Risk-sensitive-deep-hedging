import torch
from analysis.bootstrap_cvar import bootstrap_cvar_difference

# -------------------------
# Load PnL distributions
# -------------------------
pnl_cvar = torch.load("results/pnl_deep.pt")
pnl_entropic = torch.load("results/pnl_entropic_deep.pt")
pnl_variance = torch.load("results/pnl_variance_deep.pt")
pnl_delta = torch.load("results/pnl_variance_delta.pt")

alpha = 0.95

print("\n=== Bootstrap CVaR Significance Tests (95%) ===")

# CVaR vs Delta
mean_diff, lo, hi = bootstrap_cvar_difference(
    pnl_cvar, pnl_delta, alpha
)
print("\nCVaR Deep Hedge vs Delta Hedge:")
print(f"Mean CVaR difference: {mean_diff:.2f}")
print(f"95% CI: [{lo:.2f}, {hi:.2f}]")

# Entropic vs Delta
mean_diff, lo, hi = bootstrap_cvar_difference(
    pnl_entropic, pnl_delta, alpha
)
print("\nEntropic Hedge vs Delta Hedge:")
print(f"Mean CVaR difference: {mean_diff:.2f}")
print(f"95% CI: [{lo:.2f}, {hi:.2f}]")

# Variance vs Delta
mean_diff, lo, hi = bootstrap_cvar_difference(
    pnl_variance, pnl_delta, alpha
)
print("\nVariance Hedge vs Delta Hedge:")
print(f"Mean CVaR difference: {mean_diff:.2f}")
print(f"95% CI: [{lo:.2f}, {hi:.2f}]")
