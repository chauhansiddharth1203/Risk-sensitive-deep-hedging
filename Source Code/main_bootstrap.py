"""
main_bootstrap.py
-----------------
Bootstrap 95% confidence intervals for all Week 1 CVaR results.

Confirms that the improvements from adding a variance swap are statistically
significant and not just Monte Carlo noise.

Produces:
  1. Console table: CVaR point estimate + 95% CI for each model vs delta hedge
  2. Plot: results/bootstrap_ci_plot.png -- forest plot of confidence intervals

Run AFTER training all 6 models (main.py, main_variance.py, main_entropic.py,
main_varswap_cvar.py, main_varswap_variance.py, main_varswap_entropic.py)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from policy.network import HedgingPolicy
from policy.network_varswap import HedgingPolicyVarSwap
from market.heston import simulate_heston
from market.heston_with_var_swap import simulate_heston_with_var_swap
from baselines.delta_hedge import delta_hedge_pnl
from analysis.bootstrap_cvar import bootstrap_cvar_difference

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

N_TEST      = 5000
T           = 30
ALPHA       = 0.95
N_BOOTSTRAP = 2000


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return np.sort(pnl)[:k].mean()


def rollout_stock_only(policy, S, payoff_fn):
    policy.eval()
    pnls = []
    with torch.no_grad():
        for i in range(S.shape[0]):
            pnls.append(policy.rollout(S[i], payoff_fn).item())
    return np.array(pnls)


def rollout_varswap(policy, S, VS, payoff_fn):
    policy.eval()
    pnls = []
    with torch.no_grad():
        for i in range(S.shape[0]):
            pnls.append(policy.rollout(S[i], VS[i], payoff_fn).item())
    return np.array(pnls)


# ------------------------------------------------------------------ #
# 1. Simulate shared test paths                                       #
# ------------------------------------------------------------------ #
print("Simulating Heston test paths ...")
S_h,  payoff_h  = simulate_heston(N=N_TEST, T=T, device=device)
S_vs, VS, payoff_vs = simulate_heston_with_var_swap(N=N_TEST, T=T, device=device)

pnl_delta_h  = delta_hedge_pnl(S_h,  payoff_h).cpu().numpy()
pnl_delta_vs = delta_hedge_pnl(S_vs, payoff_vs).cpu().numpy()


# ------------------------------------------------------------------ #
# 2. Load and evaluate all 6 models                                   #
# ------------------------------------------------------------------ #
MODEL_SPECS = [
    # (label,                   type,    path,                                         cost_rate, baseline_pnl)
    ("CVaR (stock only)",     "stock", "results/deep_hedge_var_cvar_annealed.pth",   0.0002,  pnl_delta_h),
    ("Variance (stock only)", "stock", "results/deep_hedge_variance.pth",            0.001,   pnl_delta_h),
    ("Entropic (stock only)", "stock", "results/deep_hedge_entropic.pth",            0.001,   pnl_delta_h),
    ("CVaR (stock+VS)",       "vs",    "results/varswap_cvar.pth",                   0.0002,  pnl_delta_vs),
    ("Variance (stock+VS)",   "vs",    "results/varswap_variance.pth",               0.001,   pnl_delta_vs),
    ("Entropic (stock+VS)",   "vs",    "results/varswap_entropic.pth",               0.001,   pnl_delta_vs),
]

results = {}
for label, mtype, path, cr, baseline in MODEL_SPECS:
    print(f"  Evaluating {label} ...")
    if mtype == "stock":
        pol = HedgingPolicy(state_dim=3, action_dim=1, cost_rate=cr).to(device)
        pol.load_state_dict(torch.load(path, map_location=device))
        pnl = rollout_stock_only(pol, S_h, payoff_h)
    else:
        pol = HedgingPolicyVarSwap(cost_rate=cr).to(device)
        pol.load_state_dict(torch.load(path, map_location=device))
        pnl = rollout_varswap(pol, S_vs, VS, payoff_vs)

    mean_diff, ci_lo, ci_hi = bootstrap_cvar_difference(
        torch.tensor(pnl),
        torch.tensor(baseline),
        alpha=ALPHA,
        n_bootstrap=N_BOOTSTRAP,
    )
    results[label] = {
        "cvar":       cvar_np(pnl),
        "cvar_delta": cvar_np(baseline),
        "mean_diff":  mean_diff,
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
    }


# ------------------------------------------------------------------ #
# 3. Print table                                                      #
# ------------------------------------------------------------------ #
print(f"\n{'Model':<26}  {'CVaR':>7}  {'Delta vs Delta':>10}  {'95% CI':>22}  {'Sig?':>5}")
print("-" * 78)
for label, r in results.items():
    sig = "YES *" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "no"
    print(
        f"{label:<26}  {r['cvar']:>7.2f}  {r['mean_diff']:>+10.2f}  "
        f"[{r['ci_lo']:>+6.2f}, {r['ci_hi']:>+6.2f}]  {sig:>5}"
    )
print("\n* Significant: 95% CI does not cross zero.")


# ------------------------------------------------------------------ #
# 4. Forest plot                                                      #
# ------------------------------------------------------------------ #
labels    = list(results.keys())
diffs     = [results[l]["mean_diff"] for l in labels]
ci_lo     = [results[l]["ci_lo"]    for l in labels]
ci_hi     = [results[l]["ci_hi"]    for l in labels]

colours = ["#2196F3"] * 3 + ["#FF5722"] * 3   # blue = stock only, orange = stock+VS

fig, ax = plt.subplots(figsize=(9, 6))

for i, (label, diff, lo, hi, col) in enumerate(zip(labels, diffs, ci_lo, ci_hi, colours)):
    ax.plot([lo, hi], [i, i], color=col, linewidth=3.0, solid_capstyle="round", alpha=0.7)
    ax.plot(diff, i, "o", color=col, markersize=9, zorder=5)

ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("CVaR improvement over Black-Scholes Delta Hedge  (positive = better)", fontsize=9)
ax.set_title(
    "Bootstrap 95% Confidence Intervals -- Week 1 Results\n"
    "CVaR Improvement vs Black-Scholes Delta Hedge",
    fontsize=10,
)
ax.grid(axis="x", alpha=0.3)

legend_handles = [
    Patch(color="#2196F3", label="Stock only"),
    Patch(color="#FF5722", label="Stock + Variance Swap"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

# Add significance annotation
for i, label in enumerate(labels):
    r = results[label]
    if r["ci_lo"] > 0 or r["ci_hi"] < 0:
        ax.text(max(ci_hi) + 0.3, i, "[OK]", va="center", color="green", fontsize=11)

plt.tight_layout()
plt.savefig("results/bootstrap_ci_plot.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/bootstrap_ci_plot.png")
plt.show()
