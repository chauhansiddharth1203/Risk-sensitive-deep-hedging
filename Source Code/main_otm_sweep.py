"""
main_otm_sweep.py
-----------------
Week 4 Experiment A: OTM options payoff sweep.

Tests whether the variance swap advantage grows as options move further OTM.

Theory predicts:
  - ITM options (K=85, 90): high delta, low vega/delta ratio -> VS less critical
  - ATM options (K=100):    balanced delta and vega -> VS helpful
  - OTM options (K=105, 110): low delta, high vega/delta ratio -> VS most critical

For each strike, we train a stock+VS CVaR policy AND a stock-only CVaR policy,
then compare both with the BS delta hedge at that strike.

Produces:
  results/otm_sweep_improvement.png  -- CVaR improvement vs strike
  results/otm_sweep_absolute.png     -- absolute CVaR vs strike for all strategies
  results/otm_sweep_results.csv      -- full results table
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.network import HedgingPolicy
from policy.network_varswap import HedgingPolicyVarSwap
from market.heston_with_var_swap import simulate_heston_with_var_swap
from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

STRIKES    = [85, 90, 95, 100, 105, 110]
COST_RATE  = 0.0002
ALPHA      = 0.95
N_TEST     = 5000
EPOCHS     = 200
BATCH_SIZE = 64
T          = 30


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


def train_policy(policy, optimizer, K, mtype, epochs, batch_size, device):
    """Train a policy for a given strike K."""
    policy.train()

    alpha_start, alpha_end = 0.80, 0.95

    for epoch in range(epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / max(epochs - 1, 1)

        if mtype == "vs":
            S, VS, payoff_fn = simulate_heston_with_var_swap(
                N=batch_size, K=K, device=device
            )
            pnls = torch.stack([policy.rollout(S[i], VS[i], payoff_fn)
                                 for i in range(batch_size)])
        else:
            S, payoff_fn = simulate_heston(N=batch_size, K=K, device=device)
            pnls = torch.stack([policy.rollout(S[i], payoff_fn)
                                 for i in range(batch_size)])

        loss = -cvar(pnls, alpha)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    policy.eval()


def evaluate_policy(policy, K, mtype, N, device):
    """Evaluate a trained policy on N test paths."""
    pnls = []
    with torch.no_grad():
        if mtype == "vs":
            S, VS, payoff_fn = simulate_heston_with_var_swap(N=N, K=K, device=device)
            for i in range(N):
                pnls.append(policy.rollout(S[i], VS[i], payoff_fn).item())
        else:
            S, payoff_fn = simulate_heston(N=N, K=K, device=device)
            for i in range(N):
                pnls.append(policy.rollout(S[i], payoff_fn).item())
    return np.array(pnls)


# ------------------------------------------------------------------ #
# Main sweep                                                          #
# ------------------------------------------------------------------ #
results = []
torch.manual_seed(42)

for K in STRIKES:
    print(f"\n{'='*55}")
    print(f"Strike K = {K}  (moneyness = {K/100:.0%})")
    print(f"{'='*55}")

    # --- Train stock-only policy ---
    print("  Training stock-only CVaR policy ...")
    pol_stock = HedgingPolicy(state_dim=3, action_dim=1, cost_rate=COST_RATE).to(device)
    opt_stock = torch.optim.Adam(pol_stock.parameters(), lr=3e-4)
    train_policy(pol_stock, opt_stock, K, "stock", EPOCHS, BATCH_SIZE, device)

    # --- Train stock+VS policy ---
    print("  Training stock+VS CVaR policy ...")
    pol_vs = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    opt_vs = torch.optim.Adam(pol_vs.parameters(), lr=3e-4)
    train_policy(pol_vs, opt_vs, K, "vs", EPOCHS, BATCH_SIZE, device)

    # --- Evaluate on shared test paths ---
    torch.manual_seed(0)
    S_test, VS_test, payoff_test = simulate_heston_with_var_swap(N=N_TEST, K=K, device=device)

    pnl_delta = delta_hedge_pnl(S_test, payoff_test, K=float(K)).cpu().numpy()
    pnl_stock = evaluate_policy(pol_stock, K, "stock", N_TEST, device)
    pnl_vs    = evaluate_policy(pol_vs,    K, "vs",    N_TEST, device)

    c_delta = cvar_np(pnl_delta)
    c_stock = cvar_np(pnl_stock)
    c_vs    = cvar_np(pnl_vs)

    row = {
        "K":              K,
        "moneyness":      K / 100,
        "cvar_delta":     c_delta,
        "cvar_stock":     c_stock,
        "cvar_vs":        c_vs,
        "imp_stock":      c_stock - c_delta,
        "imp_vs":         c_vs    - c_delta,
        "vs_over_stock":  c_vs    - c_stock,
    }
    results.append(row)

    print(f"  Delta:    {c_delta:>8.2f}")
    print(f"  Stock:    {c_stock:>8.2f}  (Delta {row['imp_stock']:>+.2f})")
    print(f"  Stock+VS: {c_vs:>8.2f}  (Delta {row['imp_vs']:>+.2f})")
    print(f"  VS gain over stock-only: {row['vs_over_stock']:>+.2f}")

    # Save models
    torch.save(pol_stock.state_dict(), f"results/otm_stock_K{K}.pth")
    torch.save(pol_vs.state_dict(),    f"results/otm_vs_K{K}.pth")


# ------------------------------------------------------------------ #
# Save CSV                                                            #
# ------------------------------------------------------------------ #
csv_path = "results/otm_sweep_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved: {csv_path}")


# ------------------------------------------------------------------ #
# Plot A: CVaR improvement vs strike                                  #
# ------------------------------------------------------------------ #
Ks         = [r["K"]         for r in results]
imp_stock  = [r["imp_stock"] for r in results]
imp_vs     = [r["imp_vs"]    for r in results]
vs_margin  = [r["vs_over_stock"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(Ks, imp_stock, "o-", color="#2196F3", linewidth=2, markersize=7, label="Stock only")
ax.plot(Ks, imp_vs,    "o-", color="#FF5722", linewidth=2, markersize=7, label="Stock + VS")
ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
ax.fill_between(Ks, imp_vs, 0,
                where=[v > 0 for v in imp_vs],
                alpha=0.12, color="#FF5722", label="VS beats delta hedge")
ax.fill_between(Ks, imp_vs, 0,
                where=[v < 0 for v in imp_vs],
                alpha=0.12, color="#2196F3")
ax.set_xlabel("Strike K  (ATM = 100)", fontsize=10)
ax.set_ylabel("CVaR improvement vs Delta Hedge", fontsize=9)
ax.set_title("CVaR Improvement Across Strikes\n(positive = beats delta hedge)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(Ks)

ax2 = axes[1]
ax2.plot(Ks, vs_margin, "o-", color="#4CAF50", linewidth=2.5, markersize=8)
ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")
ax2.fill_between(Ks, vs_margin, 0,
                 where=[v > 0 for v in vs_margin],
                 alpha=0.15, color="#4CAF50")
ax2.set_xlabel("Strike K  (ATM = 100)", fontsize=10)
ax2.set_ylabel("CVaR gain: Stock+VS over Stock-only", fontsize=9)
ax2.set_title("Marginal Benefit of Adding Variance Swap\nby Option Strike", fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xticks(Ks)

plt.tight_layout()
plt.savefig("results/otm_sweep_improvement.png", dpi=150, bbox_inches="tight")
print("Saved: results/otm_sweep_improvement.png")


# ------------------------------------------------------------------ #
# Plot B: Absolute CVaR across strategies and strikes                #
# ------------------------------------------------------------------ #
fig2, ax3 = plt.subplots(figsize=(9, 5))
cvar_delta_vals = [r["cvar_delta"] for r in results]
cvar_stock_vals = [r["cvar_stock"] for r in results]
cvar_vs_vals    = [r["cvar_vs"]    for r in results]

ax3.plot(Ks, cvar_delta_vals, "s--", color="#607D8B", linewidth=1.5,
         markersize=7, label="Delta Hedge (BS)")
ax3.plot(Ks, cvar_stock_vals, "o-",  color="#2196F3", linewidth=2,
         markersize=7, label="Stock only (deep hedge)")
ax3.plot(Ks, cvar_vs_vals,    "o-",  color="#FF5722", linewidth=2,
         markersize=7, label="Stock + VS (deep hedge)")

ax3.set_xlabel("Strike K", fontsize=10)
ax3.set_ylabel("CVaR at 95% confidence  (higher = better)", fontsize=9)
ax3.set_title("Absolute CVaR Across Strikes -- All Strategies", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.set_xticks(Ks)

plt.tight_layout()
plt.savefig("results/otm_sweep_absolute.png", dpi=150, bbox_inches="tight")
print("Saved: results/otm_sweep_absolute.png")

plt.show()
print("\nDone.")
