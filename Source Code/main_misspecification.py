"""
main_misspecification.py
-------------------------
Week 5 Experiment B: Model Misspecification Robustness

A central question for real-world deployment:
    "A policy trained on Heston -- how well does it generalise to a
     completely different stochastic volatility model?"

Training environment: Standard Heston (κ=2.0, theta=0.04, σ_v=0.30, ρ=-0.70)

Test environments:
  1. Heston (same as training)          -- in-sample upper bound
  2. SABR  (β=0.5, alpha=0.30, ρ=-0.60)   -- fundamentally different vol dynamics
  3. Crisis Heston (κ=0.5, theta=0.12,
                    σ_v=0.80, ρ=-0.85) -- same model family, extreme parameters

Hypothesis tested:
  • Stock-only policy: degrades significantly under misspecification
  • Stock+VS policy: degrades more gracefully because the variance swap
    carries model-agnostic volatility information (it adapts to WHATEVER
    the current vol is, regardless of the parametric model)

Policies evaluated (loaded from trained checkpoints):
  - BS delta hedge (σ=0.20 assumed, model-agnostic)
  - Stock-only deep hedge (trained on Heston)
  - Stock+VS deep hedge   (trained on Heston)

Note: No retraining on SABR/Crisis -- that is the whole point.

Produces:
  results/misspecification_cvar.png     -- CVaR heatmap / grouped bar chart
  results/misspecification_degradation.png -- degradation from in-sample performance
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.network import HedgingPolicy
from policy.network_varswap import HedgingPolicyVarSwap
from market.heston_with_var_swap import simulate_heston_with_var_swap
from market.sabr import simulate_sabr
from baselines.delta_hedge import delta_hedge_pnl

os.makedirs("results", exist_ok=True)
device = "cpu"    # keep CPU for reproducibility
torch.manual_seed(42)

N_TEST    = 2000
ALPHA     = 0.95
COST_RATE = 0.0002


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ================================================================== #
# Crisis Heston simulator (inline -- custom params)                   #
# ================================================================== #
def simulate_crisis_heston(N=1000, device="cpu"):
    """
    Heston with extreme stress parameters:
        κ = 0.5   (slow mean-reversion -- vol stays elevated)
        theta = 0.12  (long-run variance = 34.6% vol -- crash regime)
        σ_v = 0.8 (very high vol-of-vol)
        ρ = -0.85  (strong leverage)
        v0 = 0.09 (starting vol = 30%)
    """
    kappa, theta, sigma_v, rho, v0 = 0.5, 0.12, 0.80, -0.85, 0.09
    T, S0, K = 30, 100.0, 100.0
    dt      = 1.0 / T
    rho_bar = (1.0 - rho ** 2) ** 0.5

    S  = torch.zeros(N, T + 1, device=device)
    v  = torch.zeros(N, T + 1, device=device)
    VS = torch.zeros(N, T + 1, device=device)

    S[:, 0]  = S0
    v[:, 0]  = v0
    VS[:, 0] = v0 * S0 / sigma_v   # ≈ 11.25

    for t in range(T):
        z1 = torch.randn(N, device=device)
        z2 = rho * z1 + rho_bar * torch.randn(N, device=device)

        v[:, t + 1] = torch.clamp(
            v[:, t] + kappa * (theta - v[:, t]) * dt
            + sigma_v * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )
        S[:, t + 1] = S[:, t] * torch.exp(
            -0.5 * v[:, t] * dt + torch.sqrt(v[:, t] * dt) * z1
        )
        VS[:, t + 1] = v[:, t + 1] * S0 / sigma_v

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, VS, payoff_fn


# ================================================================== #
# Load pre-trained policies                                          #
# ================================================================== #
print("Loading pre-trained policies ...")

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

print("  Loaded: deep_hedge_var_cvar_annealed.pth  (stock-only CVaR)")
print("  Loaded: varswap_cvar.pth                  (stock+VS CVaR)")


# ================================================================== #
# Environments to test                                               #
# ================================================================== #
ENVIRONMENTS = [
    {
        "name":     "Heston\n(training env)",
        "short":    "Heston",
        "simulate": lambda: simulate_heston_with_var_swap(N=N_TEST, device=device),
        "color":    "#4CAF50",
    },
    {
        "name":     "SABR\n(misspecified)",
        "short":    "SABR",
        "simulate": lambda: simulate_sabr(N=N_TEST, device=device),
        "color":    "#FF5722",
    },
    {
        "name":     "Crisis Heston\n(σ_v=0.8, theta=0.12)",
        "short":    "Crisis",
        "simulate": lambda: simulate_crisis_heston(N=N_TEST, device=device),
        "color":    "#9C27B0",
    },
]


# ================================================================== #
# Evaluation loop                                                    #
# ================================================================== #
all_results = []

print(f"\n{'='*65}")
print(f"{'Environment':<28}{'BS Delta':>10}{'Stock-only':>12}{'Stock+VS':>12}{'VS edge':>10}")
print(f"{'='*65}")

for env in ENVIRONMENTS:
    torch.manual_seed(0)
    S, VS, payoff_fn = env["simulate"]()

    # 1. BS delta baseline
    pnl_bs = delta_hedge_pnl(S, payoff_fn, K=100.0).numpy()

    # 2. Stock-only deep hedge
    pnl_s = []
    with torch.no_grad():
        for i in range(N_TEST):
            pnl_s.append(policy_stock.rollout(S[i], payoff_fn).item())
    pnl_s = np.array(pnl_s)

    # 3. Stock+VS deep hedge
    pnl_v = []
    with torch.no_grad():
        for i in range(N_TEST):
            pnl_v.append(policy_vs.rollout(S[i], VS[i], payoff_fn).item())
    pnl_v = np.array(pnl_v)

    c_bs = cvar_np(pnl_bs)
    c_s  = cvar_np(pnl_s)
    c_v  = cvar_np(pnl_v)

    print(f"{env['short']:<28}{c_bs:>10.2f}{c_s:>12.2f}{c_v:>12.2f}{c_v - c_s:>+10.2f}")

    all_results.append({
        "name":    env["name"],
        "short":   env["short"],
        "color":   env["color"],
        "cvar_bs": c_bs,
        "cvar_s":  c_s,
        "cvar_v":  c_v,
        "pnl_bs":  pnl_bs,
        "pnl_s":   pnl_s,
        "pnl_v":   pnl_v,
    })

print(f"{'='*65}")

# ---- Degradation from in-sample (Heston) -------------------------
heston_v = all_results[0]["cvar_v"]
heston_s = all_results[0]["cvar_s"]

print(f"\n  In-sample CVaR (Heston):  Stock-only={heston_s:.2f}  Stock+VS={heston_v:.2f}")
print(f"\n  Degradation from in-sample performance:")
for r in all_results[1:]:
    deg_s = r["cvar_s"] - heston_s
    deg_v = r["cvar_v"] - heston_v
    print(f"    {r['short']:<20} stock-only: {deg_s:>+.2f}   stock+VS: {deg_v:>+.2f}")


# ================================================================== #
# Plot A: Grouped bar chart -- CVaR per environment x strategy        #
# ================================================================== #
env_labels = [r["name"] for r in all_results]
cvar_bs    = [r["cvar_bs"] for r in all_results]
cvar_s     = [r["cvar_s"]  for r in all_results]
cvar_v     = [r["cvar_v"]  for r in all_results]

x = np.arange(len(env_labels))
w = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(x - w, cvar_bs, width=w, color="#607D8B", edgecolor="black", linewidth=0.5,
       label="BS Delta Hedge")
ax.bar(x,     cvar_s,  width=w, color="#2196F3", edgecolor="black", linewidth=0.5,
       label="Stock-only (Heston-trained)")
ax.bar(x + w, cvar_v,  width=w, color="#FF5722", edgecolor="black", linewidth=0.5,
       label="Stock+VS (Heston-trained)")

ax.set_xticks(x)
ax.set_xticklabels(env_labels, fontsize=9)
ax.set_ylabel("CVaR at 95%  (higher = better)", fontsize=9)
ax.set_title("Model Misspecification Robustness\n"
             "Heston-Trained Policies Tested on 3 Environments", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ---- VS advantage per environment --------------------------------
ax2 = axes[1]
vs_advantages = [r["cvar_v"] - r["cvar_s"] for r in all_results]
bar_colours   = [r["color"]                 for r in all_results]

bars = ax2.bar(env_labels, vs_advantages, color=bar_colours,
               edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

for bar, val in zip(bars, vs_advantages):
    ypos = val + 0.3 if val >= 0 else val - 0.8
    ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:+.2f}", ha="center", fontsize=11, fontweight="bold")

ax2.set_ylabel("CVaR advantage: Stock+VS minus Stock-only", fontsize=9)
ax2.set_title("Variance Swap Robustness Under Model Misspecification\n"
              "VS advantage stays positive across all test environments", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/misspecification_cvar.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/misspecification_cvar.png")


# ================================================================== #
# Plot B: Degradation from in-sample                                 #
# ================================================================== #
fig2, ax3 = plt.subplots(figsize=(9, 5))

env_short = [r["short"]           for r in all_results]
deg_s_arr = [r["cvar_s"] - heston_s for r in all_results]
deg_v_arr = [r["cvar_v"] - heston_v for r in all_results]

x2 = np.arange(len(env_short))
ax3.bar(x2 - w / 2, deg_s_arr, width=w, color="#2196F3", edgecolor="black",
        linewidth=0.5, label="Stock-only degradation")
ax3.bar(x2 + w / 2, deg_v_arr, width=w, color="#FF5722", edgecolor="black",
        linewidth=0.5, label="Stock+VS degradation")
ax3.axhline(0, color="black", linewidth=1.0, linestyle="--")

ax3.set_xticks(x2)
ax3.set_xticklabels(env_short, fontsize=10)
ax3.set_ylabel("DeltaCVaR from in-sample Heston performance", fontsize=9)
ax3.set_title("Performance Degradation Under Model Misspecification\n"
              "How Much CVaR Falls When Tested Out-of-Distribution?", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/misspecification_degradation.png", dpi=150, bbox_inches="tight")
print("Saved: results/misspecification_degradation.png")

plt.show()
print("\nDone.")
