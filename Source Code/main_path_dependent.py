"""
main_path_dependent.py
-----------------------
Week 5 Experiment A: Path-Dependent Option Hedging

Deep hedging is tested on TWO exotic payoffs where analytic delta hedges break down:

  1. Asian call:
       payoff = max( mean(S_0, S_1, ..., S_T) - K, 0 )
       The payoff depends on the running average -- not just the terminal price.
       A vanilla delta hedger sees the wrong Greek at every step.

  2. Down-and-out barrier call:
       payoff = max(S_T - K, 0) x I(min_t S_t >= B)
       The option is knocked out (dies) if S ever touches the barrier B = 90.
       Once knocked out the optimal hedge is zero -- something BS delta ignores.

Key insight (novel):
  The deep hedger receives the RUNNING PATH STATISTIC (running average OR
  running minimum) as an extra state input. It naturally learns:
    - For Asian: to hedge the difference between current S and the locked-in average
    - For Barrier: to unwind immediately when the barrier is threatened

Strategy comparison:
  [1] BS delta of vanilla ATM call (path-agnostic, analytically wrong baseline)
  [2] Stock-only deep hedge  (4-dim state: S/S0, t/T, path_stat/S0, prev_δ)
  [3] Stock + VS deep hedge  (6-dim state: S/S0, VS/VS0, t/T, path_stat/S0, prev_δS, prev_δV)

Produces:
  results/path_dependent_cvar.png   -- grouped bar chart (CVaR per strategy x payoff)
  results/path_dependent_pnl.png    -- PnL distribution overlay
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from market.heston_with_var_swap import simulate_heston_with_var_swap
from baselines.delta_hedge import bs_call_delta
from risk.cvar import cvar as cvar_torch

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# -- Hyperparameters -------------------------------------------------- #
COST_RATE  = 0.0002
ALPHA      = 0.95
N_TEST     = 4000
EPOCHS     = 700
BATCH_SIZE = 64
S0         = 100.0
K          = 100.0
B          = 90.0    # barrier level (down-and-out)
VS0        = 0.04 * S0 / 0.30   # ≈ 13.33


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


def make_net(in_dim, hidden=64):
    """MLP hedging network."""
    net = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 2),           # [delta_S, delta_V]
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net.to(device)


# ================================================================== #
# BS delta baseline (path-agnostic -- wrong for both exotic payoffs)  #
# ================================================================== #
@torch.no_grad()
def vanilla_bs_delta_pnl(S, payoff_fn, cost_rate=COST_RATE):
    """
    Standard BS delta hedge for ATM call (sigma=0.20, K=100).
    This is the 'naive' baseline that ignores path structure.
    """
    N, Tt = S.shape[0], S.shape[1] - 1
    pnl       = torch.zeros(N, device=S.device)
    prev_delta = torch.zeros(N, device=S.device)

    for t in range(Tt):
        tau   = max((Tt - t) / Tt, 1e-6)
        delta = bs_call_delta(S[:, t], K, tau, 0.0, 0.20)
        pnl  += prev_delta * (S[:, t + 1] - S[:, t])
        pnl  -= cost_rate  * torch.abs(delta - prev_delta)
        prev_delta = delta

    pnl -= payoff_fn(S)
    return pnl


# ================================================================== #
# Training: Stock-only  (4-dim state)                                #
# ================================================================== #
def train_stock_only(payoff_type, epochs, batch_size):
    """
    4-dim state: [S/S0, t/T, path_stat/S0, prev_delta_S]
    1 action  : delta_S
    """
    net = nn.Sequential(
        nn.Linear(4, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)

    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    alpha_start, alpha_end = 0.80, 0.95

    for epoch in range(epochs):
        alpha_train = alpha_start + (alpha_end - alpha_start) * epoch / max(epochs - 1, 1)
        S_b, VS_b, _ = simulate_heston_with_var_swap(N=batch_size, device=device)
        T = S_b.shape[1] - 1

        pnl_list = []
        for i in range(batch_size):
            pnl      = torch.zeros((), device=device)
            prev_dS  = torch.zeros((), device=device)
            run_sum  = S_b[i, 0].clone()
            min_s    = S_b[i, 0].clone()

            for t in range(T):
                run_sum = run_sum + S_b[i, t]
                min_s   = torch.min(min_s, S_b[i, t])
                stat    = run_sum / (t + 1) if payoff_type == "asian" else min_s

                state  = torch.stack([S_b[i, t] / S0,
                                      torch.tensor(t / T, device=device),
                                      stat / S0,
                                      prev_dS])
                dS = torch.tanh(net(state)[0]) * 5.0
                pnl   += prev_dS * (S_b[i, t + 1] - S_b[i, t])
                pnl   -= COST_RATE * torch.abs(dS - prev_dS)
                prev_dS = dS

            S_T    = S_b[i, -1]
            run_sum = run_sum + S_T
            min_s   = torch.min(min_s, S_T)

            if payoff_type == "asian":
                payoff = torch.clamp(run_sum / (T + 1) - K, min=0.0)
            else:
                alive  = (min_s >= B).float()
                payoff = torch.clamp(S_T - K, min=0.0) * alive

            pnl -= payoff
            pnl_list.append(pnl)

        loss = -cvar_torch(torch.stack(pnl_list), alpha_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()

    net.eval()
    return net


# ================================================================== #
# Training: Stock + VS  (6-dim state)                                #
# ================================================================== #
def train_vs_policy(payoff_type, epochs, batch_size):
    """
    6-dim state: [S/S0, VS/VS0, t/T, path_stat/S0, prev_dS, prev_dV]
    2 actions  : delta_S, delta_V
    """
    net = make_net(6)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    alpha_start, alpha_end = 0.80, 0.95

    for epoch in range(epochs):
        alpha_train = alpha_start + (alpha_end - alpha_start) * epoch / max(epochs - 1, 1)
        S_b, VS_b, _ = simulate_heston_with_var_swap(N=batch_size, device=device)
        T = S_b.shape[1] - 1

        pnl_list = []
        for i in range(batch_size):
            pnl      = torch.zeros((), device=device)
            prev_dS  = torch.zeros((), device=device)
            prev_dV  = torch.zeros((), device=device)
            run_sum  = S_b[i, 0].clone()
            min_s    = S_b[i, 0].clone()

            for t in range(T):
                run_sum = run_sum + S_b[i, t]
                min_s   = torch.min(min_s, S_b[i, t])
                stat    = run_sum / (t + 1) if payoff_type == "asian" else min_s

                state = torch.stack([S_b[i, t]  / S0,
                                     VS_b[i, t] / VS0,
                                     torch.tensor(t / T, device=device),
                                     stat / S0,
                                     prev_dS, prev_dV])
                action = net(state)
                dS = torch.tanh(action[0]) * 5.0
                dV = torch.tanh(action[1]) * 5.0

                pnl += prev_dS * (S_b[i, t + 1] - S_b[i, t])
                pnl += prev_dV * (VS_b[i, t + 1] - VS_b[i, t])
                pnl -= (COST_RATE * torch.abs(dS - prev_dS) * (S_b[i, t] / S0)
                      + COST_RATE * torch.abs(dV - prev_dV) * (VS_b[i, t] / VS0))
                prev_dS = dS
                prev_dV = dV

            S_T    = S_b[i, -1]
            run_sum = run_sum + S_T
            min_s   = torch.min(min_s, S_T)

            if payoff_type == "asian":
                payoff = torch.clamp(run_sum / (T + 1) - K, min=0.0)
            else:
                alive  = (min_s >= B).float()
                payoff = torch.clamp(S_T - K, min=0.0) * alive

            pnl -= payoff
            pnl_list.append(pnl)

        loss = -cvar_torch(torch.stack(pnl_list), alpha_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()

    net.eval()
    return net


# ================================================================== #
# Evaluation                                                         #
# ================================================================== #
@torch.no_grad()
def eval_stock_only_net(net, S_test, payoff_type):
    N, Tt = S_test.shape[0], S_test.shape[1] - 1
    pnls = []
    for i in range(N):
        pnl     = torch.zeros((), device=device)
        prev_dS = torch.zeros((), device=device)
        run_sum = S_test[i, 0].clone()
        min_s   = S_test[i, 0].clone()

        for t in range(Tt):
            run_sum = run_sum + S_test[i, t]
            min_s   = torch.min(min_s, S_test[i, t])
            stat    = run_sum / (t + 1) if payoff_type == "asian" else min_s

            state = torch.stack([S_test[i, t] / S0,
                                 torch.tensor(t / Tt, device=device),
                                 stat / S0,
                                 prev_dS])
            dS = torch.tanh(net(state)[0]) * 5.0
            pnl    += prev_dS * (S_test[i, t + 1] - S_test[i, t])
            pnl    -= COST_RATE * torch.abs(dS - prev_dS)
            prev_dS = dS

        S_T     = S_test[i, -1]
        run_sum = run_sum + S_T
        min_s   = torch.min(min_s, S_T)

        if payoff_type == "asian":
            payoff = torch.clamp(run_sum / (Tt + 1) - K, min=0.0)
        else:
            alive  = (min_s >= B).float()
            payoff = torch.clamp(S_T - K, min=0.0) * alive

        pnl -= payoff
        pnls.append(pnl.item())
    return np.array(pnls)


@torch.no_grad()
def eval_vs_net(net, S_test, VS_test, payoff_type):
    N, Tt = S_test.shape[0], S_test.shape[1] - 1
    pnls = []
    for i in range(N):
        pnl     = torch.zeros((), device=device)
        prev_dS = torch.zeros((), device=device)
        prev_dV = torch.zeros((), device=device)
        run_sum = S_test[i, 0].clone()
        min_s   = S_test[i, 0].clone()

        for t in range(Tt):
            run_sum = run_sum + S_test[i, t]
            min_s   = torch.min(min_s, S_test[i, t])
            stat    = run_sum / (t + 1) if payoff_type == "asian" else min_s

            state = torch.stack([S_test[i, t]  / S0,
                                 VS_test[i, t] / VS0,
                                 torch.tensor(t / Tt, device=device),
                                 stat / S0,
                                 prev_dS, prev_dV])
            action = net(state)
            dS = torch.tanh(action[0]) * 5.0
            dV = torch.tanh(action[1]) * 5.0

            pnl += prev_dS * (S_test[i, t + 1] - S_test[i, t])
            pnl += prev_dV * (VS_test[i, t + 1] - VS_test[i, t])
            pnl -= (COST_RATE * torch.abs(dS - prev_dS) * (S_test[i, t] / S0)
                  + COST_RATE * torch.abs(dV - prev_dV) * (VS_test[i, t] / VS0))
            prev_dS = dS
            prev_dV = dV

        S_T     = S_test[i, -1]
        run_sum = run_sum + S_T
        min_s   = torch.min(min_s, S_T)

        if payoff_type == "asian":
            payoff = torch.clamp(run_sum / (Tt + 1) - K, min=0.0)
        else:
            alive  = (min_s >= B).float()
            payoff = torch.clamp(S_T - K, min=0.0) * alive

        pnl -= payoff
        pnls.append(pnl.item())
    return np.array(pnls)


# ================================================================== #
# Payoff functions for test set evaluation (need full path S)        #
# ================================================================== #
def asian_payoff_paths(S):
    """S: (N, T+1), returns (N,) payoffs"""
    return torch.clamp(S.mean(dim=1) - K, min=0.0)


def barrier_payoff_paths(S, barrier=B):
    """Down-and-out: S: (N, T+1), returns (N,) payoffs"""
    min_prices = S.min(dim=1).values
    alive = (min_prices >= barrier).float()
    return torch.clamp(S[:, -1] - K, min=0.0) * alive


# ================================================================== #
# Main experiment loop                                               #
# ================================================================== #
PAYOFF_SPECS = {
    "Asian Call\n(K=100)": {
        "type": "asian",
        "payoff_paths": asian_payoff_paths,
        "description": "payoff = max(mean(S_0,...,S_T) - 100, 0)",
    },
    "Barrier Call\n(K=100, B=90)": {
        "type": "barrier",
        "payoff_paths": barrier_payoff_paths,
        "description": "payoff = max(S_T - 100, 0) x I(min_t S_t >= 90)",
    },
}

# Fixed test set
print("Generating fixed test paths ...")
torch.manual_seed(0)
S_test, VS_test, _ = simulate_heston_with_var_swap(N=N_TEST, device=device)

results = {}

for name, spec in PAYOFF_SPECS.items():
    ptype = spec["type"]
    clean = name.replace("\n", " ")
    print(f"\n{'='*60}")
    print(f"  {clean}")
    print(f"  {spec['description']}")
    print(f"{'='*60}")

    # ---- BS delta baseline ----------------------------------------
    def _payoff_batch(S_batch):
        return spec["payoff_paths"](S_batch)

    pnl_bs = vanilla_bs_delta_pnl(S_test, _payoff_batch).cpu().numpy()

    # ---- Stock-only deep hedge -----------------------------------
    print(f"  Training stock-only ({EPOCHS} epochs) ...")
    net_s = train_stock_only(ptype, EPOCHS, BATCH_SIZE)
    pnl_s = eval_stock_only_net(net_s, S_test, ptype)

    # ---- Stock + VS deep hedge ----------------------------------
    print(f"  Training stock+VS   ({EPOCHS} epochs) ...")
    net_v = train_vs_policy(ptype, EPOCHS, BATCH_SIZE)
    pnl_v = eval_vs_net(net_v, S_test, VS_test, ptype)

    c_bs = cvar_np(pnl_bs)
    c_s  = cvar_np(pnl_s)
    c_v  = cvar_np(pnl_v)

    print(f"\n  BS delta (naive)   CVaR : {c_bs:>8.2f}")
    print(f"  Stock-only deep    CVaR : {c_s:>8.2f}  (Delta {c_s - c_bs:>+.2f} vs BS)")
    print(f"  Stock+VS deep      CVaR : {c_v:>8.2f}  (Delta {c_v - c_bs:>+.2f} vs BS)")

    results[name] = {
        "pnl_bs": pnl_bs, "pnl_s": pnl_s, "pnl_v": pnl_v,
        "cvar_bs": c_bs,  "cvar_s": c_s,   "cvar_v": c_v,
    }


# ================================================================== #
# Plot A: Grouped bar chart                                          #
# ================================================================== #
option_names  = list(results.keys())
n_opts        = len(option_names)
strategies    = ["BS Delta\n(naive)", "Stock-only\ndeep hedge", "Stock+VS\ndeep hedge"]
colours       = ["#607D8B", "#2196F3", "#FF5722"]

x  = np.arange(n_opts)
w  = 0.24

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for j, (strat, col) in enumerate(zip(strategies, colours)):
    vals = [results[n][["cvar_bs", "cvar_s", "cvar_v"][j]] for n in option_names]
    ax.bar(x + (j - 1) * w, vals, width=w, color=col, edgecolor="black",
           linewidth=0.5, label=strat)

ax.set_xticks(x)
ax.set_xticklabels(option_names, fontsize=9)
ax.set_ylabel("CVaR at 95%  (higher = better)", fontsize=9)
ax.set_title("Path-Dependent Option Hedging\nCVaR by Strategy (Heston model, N=4000 paths)",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---- improvement bars ----
ax2 = axes[1]
imp_s = [results[n]["cvar_s"] - results[n]["cvar_bs"] for n in option_names]
imp_v = [results[n]["cvar_v"] - results[n]["cvar_bs"] for n in option_names]

ax2.bar(x - w / 2, imp_s, width=w, color="#2196F3", edgecolor="black",
        linewidth=0.5, label="Stock-only vs BS")
ax2.bar(x + w / 2, imp_v, width=w, color="#FF5722", edgecolor="black",
        linewidth=0.5, label="Stock+VS vs BS")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

for i, (vs, vv) in enumerate(zip(imp_s, imp_v)):
    offset = 0.15
    ax2.text(i - w / 2, vs + offset, f"{vs:+.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2.text(i + w / 2, vv + offset, f"{vv:+.1f}", ha="center", fontsize=9, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(option_names, fontsize=9)
ax2.set_ylabel("CVaR improvement over naive BS delta", fontsize=9)
ax2.set_title("Deep Hedging Advantage\nfor Path-Dependent Payoffs", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/path_dependent_cvar.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/path_dependent_cvar.png")


# ================================================================== #
# Plot B: PnL distributions                                          #
# ================================================================== #
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))

for ax_i, (name, res) in enumerate(results.items()):
    ax_ = axes2[ax_i]
    lo  = min(res["pnl_bs"].min(), res["pnl_s"].min(), res["pnl_v"].min()) - 1
    hi  = max(res["pnl_bs"].max(), res["pnl_s"].max(), res["pnl_v"].max()) + 1
    bins = np.linspace(lo, hi, 70)

    ax_.hist(res["pnl_bs"], bins=bins, alpha=0.40, color="#607D8B",
             label=f"BS delta  (CVaR={res['cvar_bs']:.1f})")
    ax_.hist(res["pnl_s"],  bins=bins, alpha=0.50, color="#2196F3",
             label=f"Stock-only (CVaR={res['cvar_s']:.1f})")
    ax_.hist(res["pnl_v"],  bins=bins, alpha=0.55, color="#FF5722",
             label=f"Stock+VS   (CVaR={res['cvar_v']:.1f})")

    ax_.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_.set_xlabel("Final P&L")
    ax_.set_ylabel("Frequency")
    ax_.set_title(name.replace("\n", " "))
    ax_.legend(fontsize=8)

plt.suptitle("P&L Distributions: Path-Dependent Options", fontsize=11)
plt.tight_layout()
plt.savefig("results/path_dependent_pnl.png", dpi=150, bbox_inches="tight")
print("Saved: results/path_dependent_pnl.png")

plt.show()
print("\nDone.")
