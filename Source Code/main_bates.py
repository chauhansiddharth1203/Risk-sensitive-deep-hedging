"""
main_bates.py
-------------
Week 2 main experiment: Bates model (Heston + Poisson jumps).

Trains two CVaR policies on Bates paths:
  1. Stock only     -- saved to results/bates_cvar.pth
  2. Stock + VS     -- saved to results/bates_varswap_cvar.pth

Then evaluates both on 5000 Bates test paths alongside the BS delta hedge,
and produces a comparison of Heston vs Bates results.

Key question: Does adding the variance swap still beat the delta hedge when
stock prices can jump?

Produces:
  results/bates_comparison.png   -- bar chart: CVaR for all strategies under Bates
  results/bates_vs_heston.png    -- side-by-side Heston vs Bates CVaR improvement
  results/pnl_bates_cvar.pt      -- PnL tensor for stock-only Bates policy
  results/pnl_bates_varswap.pt   -- PnL tensor for stock+VS Bates policy
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from policy.network import HedgingPolicy
from policy.network_varswap import HedgingPolicyVarSwap
from training.trainer_cvar_bates import train_cvar_bates
from training.trainer_cvar_bates_varswap import train_cvar_bates_varswap
from market.bates import simulate_bates
from market.bates_with_var_swap import simulate_bates_with_var_swap
from baselines.delta_hedge import delta_hedge_pnl

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

COST_RATE = 0.0002
ALPHA     = 0.95
N_TEST    = 5000
T         = 30


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


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
# 1. Train stock-only CVaR policy on Bates                            #
# ------------------------------------------------------------------ #
print("=" * 60)
print("Training stock-only CVaR policy on Bates model ...")
print("=" * 60)
policy_stock = HedgingPolicy(state_dim=3, action_dim=1, cost_rate=COST_RATE).to(device)
opt_stock    = torch.optim.Adam(policy_stock.parameters(), lr=3e-4)
train_cvar_bates(policy_stock, opt_stock, device=device, epochs=300, batch_size=64)


# ------------------------------------------------------------------ #
# 2. Train stock+VS CVaR policy on Bates                             #
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("Training stock+VS CVaR policy on Bates model ...")
print("=" * 60)
policy_vs = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
opt_vs    = torch.optim.Adam(policy_vs.parameters(), lr=3e-4)
train_cvar_bates_varswap(policy_vs, opt_vs, device=device, epochs=300, batch_size=64)


# ------------------------------------------------------------------ #
# 3. Evaluate on Bates test paths                                     #
# ------------------------------------------------------------------ #
print("\nSimulating Bates test paths ...")
torch.manual_seed(0)
S_b,  payoff_b  = simulate_bates(N=N_TEST, T=T, device=device)
S_bvs, VS_b, payoff_bvs = simulate_bates_with_var_swap(N=N_TEST, T=T, device=device)

pnl_delta_b  = delta_hedge_pnl(S_b,  payoff_b).cpu().numpy()
pnl_stock_b  = rollout_stock_only(policy_stock, S_b,  payoff_b)
pnl_vs_b     = rollout_varswap(policy_vs,    S_bvs, VS_b, payoff_bvs)

torch.save(torch.tensor(pnl_stock_b), "results/pnl_bates_cvar.pt")
torch.save(torch.tensor(pnl_vs_b),    "results/pnl_bates_varswap.pt")

cvar_delta_b = cvar_np(pnl_delta_b)
cvar_stock_b = cvar_np(pnl_stock_b)
cvar_vs_b    = cvar_np(pnl_vs_b)

print("\n======= Bates Model Results =======")
print(f"  Delta Hedge (BS, Bates paths) : {cvar_delta_b:>8.2f}")
print(f"  Deep Hedge  stock-only        : {cvar_stock_b:>8.2f}  (Delta {cvar_stock_b - cvar_delta_b:>+.2f})")
print(f"  Deep Hedge  stock+VS          : {cvar_vs_b:>8.2f}  (Delta {cvar_vs_b - cvar_delta_b:>+.2f})")


# ------------------------------------------------------------------ #
# 4. Load Heston results for comparison                               #
# ------------------------------------------------------------------ #
# Load pre-saved Heston PnL tensors from Week 1
heston_results_available = all(
    os.path.exists(p)
    for p in ["results/pnl_deep.pt", "results/pnl_varswap_cvar.pt", "results/pnl_delta.pt"]
)

if heston_results_available:
    pnl_stock_h = torch.load("results/pnl_deep.pt").numpy()
    pnl_vs_h    = torch.load("results/pnl_varswap_cvar.pt").numpy()
    pnl_delta_h = torch.load("results/pnl_delta.pt").numpy()

    cvar_delta_h = cvar_np(pnl_delta_h)
    cvar_stock_h = cvar_np(pnl_stock_h)
    cvar_vs_h    = cvar_np(pnl_vs_h)

    print("\n======= Heston Model Results (Week 1) =======")
    print(f"  Delta Hedge (BS, Heston paths) : {cvar_delta_h:>8.2f}")
    print(f"  Deep Hedge  stock-only         : {cvar_stock_h:>8.2f}  (Delta {cvar_stock_h - cvar_delta_h:>+.2f})")
    print(f"  Deep Hedge  stock+VS           : {cvar_vs_h:>8.2f}  (Delta {cvar_vs_h - cvar_delta_h:>+.2f})")
else:
    print("\nNote: Week 1 Heston PnL files not found; skipping Heston vs Bates comparison plot.")


# ------------------------------------------------------------------ #
# 5. Plot A: Bates CVaR comparison bar chart                         #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))

strategies = ["Delta Hedge\n(BS baseline)", "Stock only\n(deep hedge)", "Stock + VS\n(deep hedge)"]
cvar_vals  = [cvar_delta_b, cvar_stock_b, cvar_vs_b]
colours    = ["#607D8B", "#2196F3", "#FF5722"]

bars = ax.bar(strategies, cvar_vals, color=colours, edgecolor="black", linewidth=0.6, width=0.5)
ax.axhline(cvar_delta_b, color="#607D8B", linewidth=1.2, linestyle="--", alpha=0.7)

for bar, val in zip(bars, cvar_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val - 0.6,
        f"{val:.2f}",
        ha="center", va="top", fontsize=10, fontweight="bold", color="white",
    )

ax.set_ylabel("CVaR at 95% confidence  (higher = better)", fontsize=9)
ax.set_title(
    "Bates Model (Heston + Jumps): CVaR Comparison\n"
    f"lambda=1 jump/yr, μ_J=-5%, σ_J=8%   |   5,000 test paths",
    fontsize=10,
)
ax.set_ylim(min(cvar_vals) - 3, max(cvar_vals) + 3)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/bates_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/bates_comparison.png")


# ------------------------------------------------------------------ #
# 6. Plot B: Heston vs Bates side-by-side improvement               #
# ------------------------------------------------------------------ #
if heston_results_available:
    fig2, ax2 = plt.subplots(figsize=(9, 5))

    x = np.array([0, 1])
    w = 0.28

    imp_stock_h = cvar_stock_h - cvar_delta_h
    imp_vs_h    = cvar_vs_h    - cvar_delta_h
    imp_stock_b = cvar_stock_b - cvar_delta_b
    imp_vs_b    = cvar_vs_b    - cvar_delta_b

    ax2.bar(x - w, [imp_stock_h, imp_stock_b], width=w, color="#2196F3",
            edgecolor="black", linewidth=0.5, label="Stock only")
    ax2.bar(x,     [imp_vs_h,    imp_vs_b],    width=w, color="#FF5722",
            edgecolor="black", linewidth=0.5, label="Stock + VS")

    ax2.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax2.set_xticks(x - w / 2)
    ax2.set_xticklabels(["Heston model\n(Week 1)", "Bates model\n(Week 2)"], fontsize=10)
    ax2.set_ylabel("CVaR improvement vs Delta Hedge  (positive = better)", fontsize=9)
    ax2.set_title(
        "Does Adding a Variance Swap Still Help Under Jumps?\n"
        "CVaR Improvement Over Delta Hedge: Heston vs Bates",
        fontsize=10,
    )

    for bars_list, vals in [(ax2.patches[:2], [imp_stock_h, imp_stock_b]),
                             (ax2.patches[2:], [imp_vs_h,    imp_vs_b])]:
        for bar, val in zip(bars_list, vals):
            ypos = bar.get_height() + (0.1 if val >= 0 else -0.4)
            ax2.text(
                bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:+.2f}", ha="center", va="bottom", fontsize=9,
            )

    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/bates_vs_heston.png", dpi=150, bbox_inches="tight")
    print("Saved: results/bates_vs_heston.png")

plt.show()
print("\nDone.")
