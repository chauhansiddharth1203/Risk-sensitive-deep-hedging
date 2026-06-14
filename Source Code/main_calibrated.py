"""
main_calibrated.py
------------------
Week 3 Experiment A: SPX-calibrated Heston parameters.

Tests whether the Week 1 finding (variance swap beats delta hedge) holds
when using realistic SPX-calibrated parameters instead of textbook values.

Three strategies compared on 5,000 SPX-calibrated test paths:
  1. Policy trained on textbook Heston, tested on SPX paths  (zero-shot transfer)
  2. Policy trained directly on SPX paths                    (retrained)
  3. Black-Scholes delta hedge                               (baseline)

Textbook params: kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.70, v0=0.04
SPX params:      kappa=1.5, theta=0.05, sigma_v=0.40, rho=-0.80, v0=0.05

Produces:
  results/spx_comparison.png       -- CVaR bar chart
  results/spx_pnl_distribution.png -- PnL distribution overlay
  results/spx_varswap_cvar.pth     -- retrained SPX policy
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.network_varswap import HedgingPolicyVarSwap
from training.trainer_cvar_spx_varswap import train_cvar_spx_varswap
from market.heston_spx_varswap import simulate_heston_spx_varswap
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
# 1. Load Week 1 policy (trained on textbook Heston)                 #
# ------------------------------------------------------------------ #
print("Loading Week 1 policy (trained on textbook Heston params) ...")
policy_textbook = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
policy_textbook.load_state_dict(
    torch.load("results/varswap_cvar.pth", map_location=device)
)
policy_textbook.eval()


# ------------------------------------------------------------------ #
# 2. Retrain on SPX-calibrated paths                                 #
# ------------------------------------------------------------------ #
print("\nTraining new policy on SPX-calibrated paths ...")
policy_spx = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
opt_spx    = torch.optim.Adam(policy_spx.parameters(), lr=3e-4)
train_cvar_spx_varswap(policy_spx, opt_spx, device=device, epochs=300, batch_size=64)


# ------------------------------------------------------------------ #
# 3. Simulate SPX test paths                                         #
# ------------------------------------------------------------------ #
print("\nSimulating SPX test paths ...")
torch.manual_seed(0)
S_spx, VS_spx, payoff_spx = simulate_heston_spx_varswap(N=N_TEST, T=T, device=device)

pnl_delta     = delta_hedge_pnl(S_spx, payoff_spx).cpu().numpy()
pnl_textbook  = rollout_varswap(policy_textbook, S_spx, VS_spx, payoff_spx)
pnl_spx       = rollout_varswap(policy_spx,      S_spx, VS_spx, payoff_spx)

torch.save(torch.tensor(pnl_spx), "results/pnl_spx_varswap.pt")

cvar_delta    = cvar_np(pnl_delta)
cvar_textbook = cvar_np(pnl_textbook)
cvar_spx      = cvar_np(pnl_spx)

print("\n======= SPX-Calibrated Results =======")
print(f"  Delta Hedge (BS)                     : {cvar_delta:>8.2f}")
print(f"  Trained on textbook, tested on SPX   : {cvar_textbook:>8.2f}  (Delta {cvar_textbook - cvar_delta:>+.2f})")
print(f"  Trained on SPX, tested on SPX        : {cvar_spx:>8.2f}  (Delta {cvar_spx - cvar_delta:>+.2f})")


# ------------------------------------------------------------------ #
# 4. Plot A: CVaR bar chart                                          #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))

labels  = ["Delta Hedge\n(BS baseline)", "Textbook policy\n(zero-shot transfer)", "SPX-retrained\npolicy"]
vals    = [cvar_delta, cvar_textbook, cvar_spx]
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
    "SPX-Calibrated Parameters: Does the Finding Hold?\n"
    "kappa=1.5, theta=0.05, sigma_v=0.40, rho=-0.80, v0=0.05",
    fontsize=10,
)
ax.set_ylim(min(vals) - 3, max(vals) + 3)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/spx_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/spx_comparison.png")


# ------------------------------------------------------------------ #
# 5. Plot B: PnL distribution                                        #
# ------------------------------------------------------------------ #
fig2, ax2 = plt.subplots(figsize=(8, 4))
lo = min(pnl_delta.min(), pnl_textbook.min(), pnl_spx.min()) - 2
hi = max(pnl_delta.max(), pnl_textbook.max(), pnl_spx.max()) + 2
bins = np.linspace(lo, hi, 80)

ax2.hist(pnl_delta,    bins=bins, alpha=0.4, color="gray",    label="Delta Hedge")
ax2.hist(pnl_textbook, bins=bins, alpha=0.5, color="#FF9800", label="Textbook policy (zero-shot)")
ax2.hist(pnl_spx,      bins=bins, alpha=0.5, color="#FF5722", label="SPX-retrained policy")
ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_xlabel("Final P&L")
ax2.set_ylabel("Frequency")
ax2.set_title("P&L Distribution under SPX-Calibrated Heston Parameters")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/spx_pnl_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: results/spx_pnl_distribution.png")

plt.show()
print("\nDone.")
