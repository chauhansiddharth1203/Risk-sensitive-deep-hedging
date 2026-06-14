"""
main_heston_benchmark.py
-------------------------
Week 4 Experiment A: Heston delta-vega benchmark.

Compares four strategies on the same 2000 Heston test paths:

  1. BS delta hedge          -- ignores stochastic vol entirely
  2. Heston delta-vega hedge -- uses current v_t analytically (theoretical best)
  3. Stock-only deep hedge   -- learned, no vol instrument
  4. Stock + VS deep hedge   -- learned, with vol instrument

Key question: How close does deep hedging get to the theoretical Heston optimum?

Note: The Heston delta-vega hedge uses the EXACT current variance at each step
(information the BS hedger ignores). It represents the ceiling for a model-aware
analytical hedger that also has access to instantaneous vol.

Produces:
  results/heston_benchmark.png     -- 4-strategy CVaR bar chart
  results/heston_benchmark_pnl.png -- PnL distribution overlay
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.network import HedgingPolicy
from policy.network_varswap import HedgingPolicyVarSwap
from market.heston_with_var_swap import simulate_heston_with_var_swap
from baselines.delta_hedge import delta_hedge_pnl
from baselines.heston_delta_vega_hedge import heston_delta_vega_pnl

os.makedirs("results", exist_ok=True)
device = "cpu"    # must be CPU for numpy interop in heston_delta_vega_pnl
torch.manual_seed(0)

N_TEST    = 500    # fewer paths due to per-path numpy computation in analytical hedge
ALPHA     = 0.95
COST_RATE = 0.0002


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ------------------------------------------------------------------ #
# 1. Simulate test paths                                             #
# ------------------------------------------------------------------ #
print("Simulating Heston test paths ...")
S, VS, payoff_fn = simulate_heston_with_var_swap(N=N_TEST, device=device)


# ------------------------------------------------------------------ #
# 2. BS delta hedge                                                  #
# ------------------------------------------------------------------ #
print("Running BS delta hedge ...")
pnl_bs = delta_hedge_pnl(S, payoff_fn, K=100.0).numpy()


# ------------------------------------------------------------------ #
# 3. Heston delta-vega hedge (analytical)                            #
# ------------------------------------------------------------------ #
print(f"Running Heston delta-vega hedge on {N_TEST} paths (takes ~2 min) ...")
pnl_heston_dv = heston_delta_vega_pnl(S, VS, payoff_fn, cost_rate=COST_RATE).numpy()


# ------------------------------------------------------------------ #
# 4. Load deep hedge policies                                        #
# ------------------------------------------------------------------ #
print("Evaluating deep hedge policies ...")

policy_stock = HedgingPolicy(state_dim=3, action_dim=1, cost_rate=COST_RATE)
policy_stock.load_state_dict(
    torch.load("results/deep_hedge_var_cvar_annealed.pth", map_location="cpu")
)
policy_stock.eval()

policy_vs = HedgingPolicyVarSwap(cost_rate=COST_RATE)
policy_vs.load_state_dict(
    torch.load("results/varswap_cvar.pth", map_location="cpu")
)
policy_vs.eval()

pnl_stock = []
pnl_vs_   = []
with torch.no_grad():
    for i in range(N_TEST):
        pnl_stock.append(policy_stock.rollout(S[i], payoff_fn).item())
        pnl_vs_.append(policy_vs.rollout(S[i], VS[i], payoff_fn).item())

pnl_stock = np.array(pnl_stock)
pnl_vs_   = np.array(pnl_vs_)


# ------------------------------------------------------------------ #
# 5. Results table                                                    #
# ------------------------------------------------------------------ #
labels_all = [
    "BS Delta\nHedge",
    "Heston\nDelta-Vega",
    "Deep Hedge\n(stock only)",
    "Deep Hedge\n(stock + VS)",
]
pnls_all = [pnl_bs, pnl_heston_dv, pnl_stock, pnl_vs_]
cvars    = [cvar_np(p) for p in pnls_all]

print("\n======= Heston Benchmark Results =======")
print(f"{'Strategy':<28} {'CVaR':>8}  {'vs BS Delta':>12}")
print("-" * 52)
for label, cv in zip(labels_all, cvars):
    label_clean = label.replace("\n", " ")
    delta_vs_bs = cv - cvars[0]
    print(f"{label_clean:<28} {cv:>8.2f}  {delta_vs_bs:>+12.2f}")

# Key comparison
gap_to_ceiling = cvars[3] - cvars[1]
print(f"\n  Deep hedge (stock+VS) vs Heston analytical: {gap_to_ceiling:>+.2f}")
pct = abs(gap_to_ceiling) / abs(cvars[1] - cvars[0]) * 100 if cvars[1] != cvars[0] else 0
print(f"  Deep hedge closes {100 - pct:.0f}% of the gap between BS delta and Heston analytical")


# ------------------------------------------------------------------ #
# 6. Plot A: CVaR bar chart                                          #
# ------------------------------------------------------------------ #
colours = ["#607D8B", "#9C27B0", "#2196F3", "#FF5722"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels_all, cvars, color=colours, edgecolor="black", linewidth=0.6, width=0.55)
ax.axhline(cvars[0], color="#607D8B", linewidth=1.2, linestyle="--", alpha=0.6,
           label="BS delta baseline")
ax.axhline(cvars[1], color="#9C27B0", linewidth=1.2, linestyle=":",  alpha=0.6,
           label="Heston analytical ceiling")

for bar, val in zip(bars, cvars):
    ax.text(bar.get_x() + bar.get_width() / 2, val - 0.5,
            f"{val:.2f}", ha="center", va="top", fontsize=10,
            fontweight="bold", color="white")

ax.set_ylabel("CVaR at 95% confidence  (higher = better)", fontsize=9)
ax.set_title(
    "How Close Does Deep Hedging Get to the Theoretical Heston Optimum?\n"
    "4-Strategy Comparison on Heston Paths  (N=500 test paths)",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.set_ylim(min(cvars) - 3, max(cvars) + 3)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/heston_benchmark.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/heston_benchmark.png")


# ------------------------------------------------------------------ #
# 7. Plot B: PnL distribution overlay                                #
# ------------------------------------------------------------------ #
fig2, ax2 = plt.subplots(figsize=(9, 4))
lo   = min(p.min() for p in pnls_all) - 2
hi   = max(p.max() for p in pnls_all) + 2
bins = np.linspace(lo, hi, 70)

for label, pnl, col, alpha_hist in zip(
    labels_all, pnls_all, colours, [0.35, 0.50, 0.45, 0.55]
):
    ax2.hist(pnl, bins=bins, alpha=alpha_hist, color=col,
             label=label.replace("\n", " "))

ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax2.set_xlabel("Final P&L")
ax2.set_ylabel("Frequency")
ax2.set_title("P&L Distribution: All Four Strategies")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/heston_benchmark_pnl.png", dpi=150, bbox_inches="tight")
print("Saved: results/heston_benchmark_pnl.png")

plt.show()
print("\nDone.")
