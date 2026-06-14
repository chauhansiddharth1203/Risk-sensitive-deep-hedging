"""
plot_bates_comparison.py
------------------------
Regenerate Bates comparison plots using already-saved PnL tensors.
Run this after re-running main.py (to refresh pnl_delta.pt with K=100 fix).

Produces:
  results/bates_comparison.png  -- CVaR bar chart under Bates
  results/bates_vs_heston.png   -- Heston vs Bates side-by-side improvement
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from market.bates import simulate_bates
from market.bates_with_var_swap import simulate_bates_with_var_swap
from baselines.delta_hedge import delta_hedge_pnl

os.makedirs("results", exist_ok=True)
torch.manual_seed(0)

ALPHA  = 0.95
N_TEST = 5000
T      = 30


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ------------------------------------------------------------------ #
# 1. Load Bates PnLs (saved by main_bates.py)                        #
# ------------------------------------------------------------------ #
pnl_stock_b = torch.load("results/pnl_bates_cvar.pt").numpy()
pnl_vs_b    = torch.load("results/pnl_bates_varswap.pt").numpy()

# Re-simulate Bates paths (same seed) to compute delta hedge PnL
print("Simulating Bates test paths for delta hedge ...")
S_b,   payoff_b   = simulate_bates(N=N_TEST, T=T, device="cpu")
pnl_delta_b = delta_hedge_pnl(S_b, payoff_b).numpy()

cvar_delta_b = cvar_np(pnl_delta_b)
cvar_stock_b = cvar_np(pnl_stock_b)
cvar_vs_b    = cvar_np(pnl_vs_b)

print(f"\nBates  | Delta: {cvar_delta_b:.2f}  Stock: {cvar_stock_b:.2f}  VS: {cvar_vs_b:.2f}")


# ------------------------------------------------------------------ #
# 2. Load corrected Heston PnLs (freshly saved by main.py)           #
# ------------------------------------------------------------------ #
pnl_stock_h = torch.load("results/pnl_deep.pt").numpy()
pnl_vs_h    = torch.load("results/pnl_varswap_cvar.pt").numpy()
pnl_delta_h = torch.load("results/pnl_delta.pt").numpy()

cvar_delta_h = cvar_np(pnl_delta_h)
cvar_stock_h = cvar_np(pnl_stock_h)
cvar_vs_h    = cvar_np(pnl_vs_h)

print(f"Heston | Delta: {cvar_delta_h:.2f}  Stock: {cvar_stock_h:.2f}  VS: {cvar_vs_h:.2f}")


# ------------------------------------------------------------------ #
# 3. Plot A: Bates CVaR bar chart                                    #
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
    f"lambda=1 jump/yr, μ_J=-5%, σ_J=8%   |   {N_TEST:,} test paths",
    fontsize=10,
)
ax.set_ylim(min(cvar_vals) - 3, max(cvar_vals) + 3)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/bates_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/bates_comparison.png")


# ------------------------------------------------------------------ #
# 4. Plot B: Heston vs Bates side-by-side improvement                #
# ------------------------------------------------------------------ #
imp_stock_h = cvar_stock_h - cvar_delta_h
imp_vs_h    = cvar_vs_h    - cvar_delta_h
imp_stock_b = cvar_stock_b - cvar_delta_b
imp_vs_b    = cvar_vs_b    - cvar_delta_b

fig2, ax2 = plt.subplots(figsize=(9, 5))

x = np.array([0.0, 1.0])
w = 0.28

b1 = ax2.bar(x - w / 2, [imp_stock_h, imp_stock_b], width=w,
             color="#2196F3", edgecolor="black", linewidth=0.5, label="Stock only")
b2 = ax2.bar(x + w / 2, [imp_vs_h,    imp_vs_b],    width=w,
             color="#FF5722", edgecolor="black", linewidth=0.5, label="Stock + VS")

ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")
ax2.set_xticks(x)
ax2.set_xticklabels(["Heston model\n(stoch. vol only)", "Bates model\n(stoch. vol + jumps)"], fontsize=10)
ax2.set_ylabel("CVaR improvement vs Delta Hedge  (positive = better)", fontsize=9)
ax2.set_title(
    "Does the Variance Swap Still Help Under Jumps?\n"
    "CVaR Improvement Over Delta Hedge: Heston vs Bates",
    fontsize=10,
)

for bar, val in zip(list(b1) + list(b2), [imp_stock_h, imp_stock_b, imp_vs_h, imp_vs_b]):
    yoff = 0.12 if val >= 0 else -0.35
    ax2.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
        f"{val:+.2f}", ha="center", va="bottom", fontsize=9,
    )

ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/bates_vs_heston.png", dpi=150, bbox_inches="tight")
print("Saved: results/bates_vs_heston.png")

plt.show()
print("\nDone.")
