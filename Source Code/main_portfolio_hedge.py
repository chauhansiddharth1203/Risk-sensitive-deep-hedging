"""
main_portfolio_hedge.py
------------------------
Week 4 Experiment B: Hedge a portfolio of options.

Tests whether the variance swap advantage grows for more vega-dominated payoffs.

Three option portfolios:
  1. ATM call     : max(S_T - 100, 0)               -- baseline
  2. ATM straddle : max(S_T - 100, 0) + max(100 - S_T, 0) = |S_T - 100|
                    Near-zero delta, pure vega bet
  3. Strangle     : max(S_T - 105, 0) + max(95 - S_T, 0)
                    OTM call + OTM put, even more vol-sensitive

Theory predicts:
  - Straddle/strangle: delta ≈ 0, vega is dominant risk
  - VS should matter most for these payoffs
  - Stock-only deep hedge should fail worst on straddle/strangle

Produces:
  results/portfolio_cvar_comparison.png  -- grouped bar chart
  results/portfolio_pnl_overlay.png      -- PnL distributions
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.network_varswap import HedgingPolicyVarSwap
from market.heston_with_var_swap import simulate_heston_with_var_swap
from baselines.delta_hedge import delta_hedge_pnl, bs_call_delta
from risk.cvar import cvar

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

COST_RATE  = 0.0002
ALPHA      = 0.95
N_TEST     = 5000
EPOCHS     = 600
BATCH_SIZE = 64
T          = 30
S0         = 100.0


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ------------------------------------------------------------------ #
# Define portfolios                                                   #
# ------------------------------------------------------------------ #
PORTFOLIOS = {
    "ATM Call\n(K=100)": {
        "payoff":      lambda S_T: torch.clamp(S_T - 100.0, min=0.0),
        "bs_delta_fn": lambda S, tau: bs_call_delta(S, 100.0, tau, 0.0, 0.2),
        "description": "Single ATM call -- baseline",
    },
    "ATM Straddle\n(K=100)": {
        "payoff":      lambda S_T: (torch.clamp(S_T - 100.0, min=0.0)
                                    + torch.clamp(100.0 - S_T, min=0.0)),
        "bs_delta_fn": lambda S, tau: (2.0 * bs_call_delta(S, 100.0, tau, 0.0, 0.2) - 1.0),
        "description": "Long call + long put -- pure vega",
    },
    "Strangle\n(K=95/105)": {
        "payoff":      lambda S_T: (torch.clamp(S_T - 105.0, min=0.0)
                                    + torch.clamp(95.0 - S_T,  min=0.0)),
        "bs_delta_fn": lambda S, tau: (bs_call_delta(S, 105.0, tau, 0.0, 0.2)
                                        - (1.0 - bs_call_delta(S, 95.0, tau, 0.0, 0.2))),
        "description": "OTM call (K=105) + OTM put (K=95)",
    },
}


# ------------------------------------------------------------------ #
# BS delta hedge for arbitrary portfolio                              #
# ------------------------------------------------------------------ #
@torch.no_grad()
def portfolio_delta_hedge_pnl(S, portfolio_delta_fn, payoff_fn, cost_rate=COST_RATE):
    N, Tt = S.shape[0], S.shape[1] - 1
    pnl       = torch.zeros(N, device=S.device)
    prev_delta = torch.zeros(N, device=S.device)

    for t in range(Tt):
        tau   = max((Tt - t) / Tt, 1e-6)
        delta = portfolio_delta_fn(S[:, t], tau).to(S.device)

        pnl += prev_delta * (S[:, t + 1] - S[:, t])
        pnl -= cost_rate * torch.abs(delta - prev_delta)

        prev_delta = delta

    pnl -= payoff_fn(S[:, -1])
    return pnl


# ------------------------------------------------------------------ #
# Train stock+VS policy for each portfolio                           #
# ------------------------------------------------------------------ #
def train_vs_policy(payoff_fn_train, epochs, batch_size, device):
    from risk.cvar import cvar as cvar_fn

    policy = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=3e-4)
    policy.train()

    alpha_start, alpha_end = 0.80, 0.95

    for epoch in range(epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / max(epochs - 1, 1)

        S_b, VS_b, _ = simulate_heston_with_var_swap(N=batch_size, device=device)

        # Use custom payoff (not the default one from simulator)
        pnl_list = []
        for i in range(batch_size):
            pnl = torch.zeros((), device=device)
            prev_dS = torch.zeros((), device=device)
            prev_dV = torch.zeros((), device=device)

            for t in range(S_b.shape[1] - 1):
                state = torch.stack([
                    S_b[i, t]  / policy.S0,
                    VS_b[i, t] / policy.VS0,
                    torch.tensor(t / (S_b.shape[1] - 1), device=device),
                    prev_dS,
                    prev_dV,
                ])
                action = policy.net(state)
                dS = torch.tanh(action[0]) * 5.0
                dV = torch.tanh(action[1]) * 5.0

                pnl += prev_dS * (S_b[i, t+1] - S_b[i, t])
                pnl += prev_dV * (VS_b[i, t+1] - VS_b[i, t])
                pnl -= (COST_RATE * torch.abs(dS - prev_dS) * (S_b[i, t] / policy.S0)
                      + COST_RATE * torch.abs(dV - prev_dV) * (VS_b[i, t] / policy.VS0))

                prev_dS = dS
                prev_dV = dV

            pnl -= payoff_fn_train(S_b[i, -1])
            pnl_list.append(pnl)

        pnls = torch.stack(pnl_list)
        loss = -cvar_fn(pnls, alpha)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

    policy.eval()
    return policy


# ------------------------------------------------------------------ #
# Main loop over portfolios                                          #
# ------------------------------------------------------------------ #
results = {}

# Fixed test paths (same for all portfolios)
torch.manual_seed(0)
S_test, VS_test, _ = simulate_heston_with_var_swap(N=N_TEST, device=device)

for name, spec in PORTFOLIOS.items():
    clean_name = name.replace("\n", " ")
    print(f"\n{'='*55}")
    print(f"Portfolio: {clean_name}")
    print(f"  {spec['description']}")
    print(f"{'='*55}")

    payoff_test = spec["payoff"]
    delta_fn    = spec["bs_delta_fn"]

    # BS portfolio delta hedge
    pnl_delta = portfolio_delta_hedge_pnl(
        S_test, delta_fn, payoff_test
    ).cpu().numpy()

    # Train and evaluate stock+VS deep hedge
    print(f"  Training stock+VS CVaR policy ({EPOCHS} epochs) ...")
    policy_vs = train_vs_policy(payoff_test, EPOCHS, BATCH_SIZE, device)

    pnl_vs = []
    with torch.no_grad():
        for i in range(N_TEST):
            pnl = torch.zeros((), device=device)
            prev_dS = torch.zeros((), device=device)
            prev_dV = torch.zeros((), device=device)
            for t in range(S_test.shape[1] - 1):
                state = torch.stack([
                    S_test[i, t]  / policy_vs.S0,
                    VS_test[i, t] / policy_vs.VS0,
                    torch.tensor(t / (S_test.shape[1] - 1), device=device),
                    prev_dS, prev_dV,
                ])
                action = policy_vs.net(state)
                dS = torch.tanh(action[0]) * 5.0
                dV = torch.tanh(action[1]) * 5.0
                pnl += prev_dS * (S_test[i, t+1] - S_test[i, t])
                pnl += prev_dV * (VS_test[i, t+1] - VS_test[i, t])
                pnl -= (COST_RATE * torch.abs(dS - prev_dS) * (S_test[i, t] / policy_vs.S0)
                      + COST_RATE * torch.abs(dV - prev_dV) * (VS_test[i, t] / policy_vs.VS0))
                prev_dS = dS; prev_dV = dV
            pnl -= payoff_test(S_test[i, -1])
            pnl_vs.append(pnl.item())
    pnl_vs = np.array(pnl_vs)

    c_delta = cvar_np(pnl_delta)
    c_vs    = cvar_np(pnl_vs)

    print(f"  BS Delta hedge CVaR : {c_delta:.2f}")
    print(f"  Stock+VS deep hedge : {c_vs:.2f}  (Delta {c_vs - c_delta:>+.2f})")

    results[name] = {
        "pnl_delta": pnl_delta,
        "pnl_vs":    pnl_vs,
        "cvar_delta": c_delta,
        "cvar_vs":    c_vs,
        "improvement": c_vs - c_delta,
    }


# ------------------------------------------------------------------ #
# Plot A: grouped bar chart                                          #
# ------------------------------------------------------------------ #
portfolio_names = list(results.keys())
cvar_delta_vals = [results[n]["cvar_delta"]   for n in portfolio_names]
cvar_vs_vals    = [results[n]["cvar_vs"]      for n in portfolio_names]
improvements    = [results[n]["improvement"]  for n in portfolio_names]

x = np.arange(len(portfolio_names))
w = 0.32

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
b1 = ax.bar(x - w/2, cvar_delta_vals, width=w, color="#607D8B",
            edgecolor="black", linewidth=0.5, label="BS Delta Hedge")
b2 = ax.bar(x + w/2, cvar_vs_vals,    width=w, color="#FF5722",
            edgecolor="black", linewidth=0.5, label="Stock + VS (deep hedge)")

ax.set_xticks(x)
ax.set_xticklabels(portfolio_names, fontsize=9)
ax.set_ylabel("CVaR at 95%  (higher = better)", fontsize=9)
ax.set_title("CVaR Across Portfolio Types\nDoes VS Help More for Vega-Dominated Payoffs?",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

ax2 = axes[1]
colours_imp = ["#FF5722" if v > 0 else "#2196F3" for v in improvements]
ax2.bar(portfolio_names, improvements, color=colours_imp,
        edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")
for i, (name, val) in enumerate(zip(portfolio_names, improvements)):
    ax2.text(i, val + (0.15 if val >= 0 else -0.4),
             f"{val:+.2f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("CVaR improvement vs BS Delta Hedge", fontsize=9)
ax2.set_title("Marginal Benefit of Variance Swap\nby Portfolio Type", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/portfolio_cvar_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/portfolio_cvar_comparison.png")

plt.show()
print("\nDone.")
