"""
main_regime_switching.py
------------------------
Week 3 Experiment B: Regime-switching Heston model.

Tests whether a policy trained in calm markets can still hedge effectively
when the market switches to a stressed regime mid-path.

Calm regime  : kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.70
Stressed regime: kappa=1.5, theta=0.09, sigma_v=0.50, rho=-0.85
Transitions  : P(calm->stressed)=0.05, P(stressed->calm)=0.25 per step

Three strategies compared on 5,000 regime-switching test paths:
  1. Policy trained on calm Heston only   (already trained: varswap_cvar.pth)
  2. Policy trained on regime-switching   (trained here)
  3. Black-Scholes delta hedge            (baseline)

Also breaks down CVaR separately for calm-dominated vs stressed-dominated paths
to show where performance drops.

Produces:
  results/regime_comparison.png        -- CVaR bar chart
  results/regime_calm_vs_stress.png    -- performance broken down by regime
  results/regime_varswap_cvar.pth      -- policy trained on regime-switching paths
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.network_varswap import HedgingPolicyVarSwap
from training.trainer_cvar_regime_varswap import train_cvar_regime_varswap
from market.regime_switching_heston_varswap import simulate_regime_switching_heston_varswap
from baselines.delta_hedge import delta_hedge_pnl

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

COST_RATE = 0.0002
ALPHA     = 0.95
N_TEST    = 5000
T         = 30


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


def rollout_varswap(policy, S, VS, payoff_fn):
    policy.eval()
    pnls = []
    with torch.no_grad():
        for i in range(S.shape[0]):
            pnls.append(policy.rollout(S[i], VS[i], payoff_fn).item())
    return np.array(pnls)


# ------------------------------------------------------------------ #
# 1. Load calm-trained policy (Week 1)                               #
# ------------------------------------------------------------------ #
print("Loading calm-trained policy (Week 1, varswap_cvar.pth) ...")
policy_calm = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
policy_calm.load_state_dict(
    torch.load("results/varswap_cvar.pth", map_location=device)
)
policy_calm.eval()


# ------------------------------------------------------------------ #
# 2. Train policy on regime-switching paths                          #
# ------------------------------------------------------------------ #
print("\nTraining new policy on regime-switching paths ...")
policy_regime = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
opt_regime    = torch.optim.Adam(policy_regime.parameters(), lr=3e-4)
train_cvar_regime_varswap(
    policy_regime, opt_regime, device=device, epochs=300, batch_size=64
)


# ------------------------------------------------------------------ #
# 3. Simulate regime-switching test paths                            #
# ------------------------------------------------------------------ #
print("\nSimulating regime-switching test paths ...")
torch.manual_seed(0)
S_rs, VS_rs, payoff_rs, regimes = simulate_regime_switching_heston_varswap(
    N=N_TEST, T=T, device=device
)

pnl_delta  = delta_hedge_pnl(S_rs, payoff_rs).cpu().numpy()
pnl_calm   = rollout_varswap(policy_calm,   S_rs, VS_rs, payoff_rs)
pnl_regime = rollout_varswap(policy_regime, S_rs, VS_rs, payoff_rs)

cvar_delta  = cvar_np(pnl_delta)
cvar_calm   = cvar_np(pnl_calm)
cvar_regime = cvar_np(pnl_regime)

# Fraction of time each path spent in stressed regime
stress_fraction = regimes.float().mean(dim=1).cpu().numpy()   # (N,)

print("\n======= Regime-Switching Results =======")
print(f"  Avg fraction of time in stressed regime : {stress_fraction.mean():.1%}")
print(f"\n  Delta Hedge (BS)                : {cvar_delta:>8.2f}")
print(f"  Calm-trained policy (stock+VS)  : {cvar_calm:>8.2f}  (Delta {cvar_calm - cvar_delta:>+.2f})")
print(f"  Regime-trained policy (stock+VS): {cvar_regime:>8.2f}  (Delta {cvar_regime - cvar_delta:>+.2f})")


# ------------------------------------------------------------------ #
# 4. Breakdown: calm-dominated vs stressed-dominated paths           #
# ------------------------------------------------------------------ #
# "Mostly calm"   = paths where <20% of steps were stressed
# "Mostly stressed" = paths where >40% of steps were stressed
calm_mask   = stress_fraction < 0.20
stress_mask = stress_fraction > 0.40

print(f"\n  Breakdown (calm-trained policy):")
print(f"  Mostly calm paths   ({calm_mask.sum()} paths):    CVaR = {cvar_np(pnl_calm[calm_mask]):.2f}")
print(f"  Mostly stressed paths ({stress_mask.sum()} paths): CVaR = {cvar_np(pnl_calm[stress_mask]):.2f}")


# ------------------------------------------------------------------ #
# 5. Plot A: Overall CVaR bar chart                                  #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))

labels  = ["Delta Hedge\n(BS baseline)", "Calm-trained policy\n(zero-shot to stress)", "Regime-trained\npolicy"]
vals    = [cvar_delta, cvar_calm, cvar_regime]
colours = ["#607D8B", "#FF9800", "#FF5722"]

bars = ax.bar(labels, vals, color=colours, edgecolor="black", linewidth=0.6, width=0.5)
ax.axhline(cvar_delta, color="#607D8B", linewidth=1.2, linestyle="--", alpha=0.7)

for bar, val in zip(bars, vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2, val - 0.5,
        f"{val:.2f}", ha="center", va="top", fontsize=10,
        fontweight="bold", color="white",
    )

ax.set_ylabel("CVaR at 95% confidence  (higher = better)", fontsize=9)
ax.set_title(
    "Regime-Switching Heston: Can a Policy Trained in Calm Markets Survive a Crisis?\n"
    "~83% calm, ~17% stressed per path on average",
    fontsize=10,
)
ax.set_ylim(min(vals) - 3, max(vals) + 3)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/regime_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/regime_comparison.png")


# ------------------------------------------------------------------ #
# 6. Plot B: CVaR by stress exposure                                 #
# ------------------------------------------------------------------ #
# Bin paths by fraction of time in stressed regime
bins_pct  = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.01]
bin_labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", ">50%"]

calm_cvars   = []
regime_cvars = []
delta_cvars  = []
counts       = []

for lo, hi in zip(bins_pct[:-1], bins_pct[1:]):
    mask = (stress_fraction >= lo) & (stress_fraction < hi)
    n    = mask.sum()
    counts.append(n)
    if n >= 5:
        calm_cvars.append(cvar_np(pnl_calm[mask]))
        regime_cvars.append(cvar_np(pnl_regime[mask]))
        delta_cvars.append(cvar_np(pnl_delta[mask]))
    else:
        calm_cvars.append(np.nan)
        regime_cvars.append(np.nan)
        delta_cvars.append(np.nan)

x    = np.arange(len(bin_labels))
w    = 0.25
fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.bar(x - w, delta_cvars,  width=w, color="#607D8B", edgecolor="black",
        linewidth=0.5, label="Delta Hedge")
ax2.bar(x,     calm_cvars,   width=w, color="#FF9800", edgecolor="black",
        linewidth=0.5, label="Calm-trained policy")
ax2.bar(x + w, regime_cvars, width=w, color="#FF5722", edgecolor="black",
        linewidth=0.5, label="Regime-trained policy")

ax2.set_xticks(x)
ax2.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(bin_labels, counts)], fontsize=8)
ax2.set_xlabel("Fraction of path spent in stressed regime", fontsize=9)
ax2.set_ylabel("CVaR at 95% confidence", fontsize=9)
ax2.set_title(
    "How Does Performance Change as the Market Gets More Stressed?\n"
    "CVaR by Stress Exposure (grouped by % of steps in stressed regime)",
    fontsize=10,
)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/regime_calm_vs_stress.png", dpi=150, bbox_inches="tight")
print("Saved: results/regime_calm_vs_stress.png")

plt.show()
print("\nDone.")
