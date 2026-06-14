"""
main_risk_comparison.py
------------------------
Week 5 Experiment C: Comprehensive Risk Measure Comparison (Clean Rewrite)

Previous Week 1 comparison had inconsistent hyperparameters:
  - Variance/Entropic: 300 epochs, cost_rate=0.001, no gradient clipping
  - CVaR: 600+ epochs, cost_rate=0.0002, with gradient clipping

This script fixes all inconsistencies and extends to 4 objectives:

  1. CVaR(95%) -- Conditional Value at Risk (tail loss minimisation)
               Buehler et al. (2019) extended; our primary objective
  2. Variance  -- Minimise Var(PnL)  (symmetric, penalises gains too)
               Buehler et al. (2018) original formulation
  3. Entropic  -- Exponential utility:  (1/lambda) log E[exp(-lambda · PnL)]
               Risk-averse investor, lambda=1.0
  4. Mean-Std  -- -(mean(PnL) - 0.5 · std(PnL))  (Markowitz-style)
               Balances expected gain against standard deviation

All 4 objectives x 2 instrument sets (stock-only, stock+VS) = 8 policies.

Consistent hyperparameters for ALL 8 policies:
  - epochs     = 500
  - batch_size = 64
  - cost_rate  = 0.0002
  - hidden_dim = 64
  - lr         = 3e-4
  - grad clip  = 1.0
  - N_test     = 5000

Evaluation metrics (on same fixed test set):
  - CVaR at 95%  (higher = better)
  - Mean PnL     (higher = better)
  - Std PnL      (lower = better)
  - Sharpe-like  = Mean / Std  (higher = better)
  - % beating delta hedge by CVaR

Produces:
  results/risk_comparison_cvar.png       -- CVaR: 8 policies + baseline
  results/risk_comparison_scatter.png    -- Mean vs Std scatter (Pareto frontier)
  results/risk_comparison_table.png      -- Full metrics table
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from market.heston_with_var_swap import simulate_heston_with_var_swap
from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# -- Hyperparameters (consistent for ALL 8 policies) ---------------- #
EPOCHS     = 500
BATCH_SIZE = 64
COST_RATE  = 0.0002
HIDDEN     = 64
LR         = 3e-4
GRAD_CLIP  = 1.0
N_TEST     = 5000
ALPHA      = 0.95
LAM        = 1.0   # entropic risk aversion


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# -- Risk measure implementations ----------------------------------- #
def loss_cvar(pnls, alpha):
    """CVaR loss with annealing handled outside."""
    sorted_pnl, _ = torch.sort(pnls)
    k = max(int((1 - alpha) * len(pnls)), 1)
    return -sorted_pnl[:k].mean()


def loss_variance(pnls, _=None):
    """Variance minimisation (original Buehler et al. 2018)."""
    return pnls.var()


def loss_entropic(pnls, _=None):
    """Numerically stable entropic risk measure."""
    x     = -LAM * pnls
    x_max = x.max()
    return (1.0 / LAM) * (x_max + torch.log(torch.mean(torch.exp(x - x_max))))


def loss_mean_std(pnls, _=None):
    """Markowitz-style: -(mean - 0.5 * std)."""
    return -(pnls.mean() - 0.5 * pnls.std())


OBJECTIVES = {
    "CVaR(95%)"  : loss_cvar,
    "Variance"   : loss_variance,
    "Entropic"   : loss_entropic,
    "Mean-Std"   : loss_mean_std,
}

# -- Network factory ------------------------------------------------ #
def make_stock_only_net():
    """3-dim state: [S_t, t/T, prev_delta]  ->  1 action."""
    net = nn.Sequential(
        nn.Linear(3, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, 1),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net.to(device)


def make_vs_net():
    """5-dim state: [S/S0, VS/VS0, t/T, prev_dS, prev_dV]  ->  2 actions."""
    net = nn.Sequential(
        nn.Linear(5, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, 2),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net.to(device)


S0  = 100.0
VS0 = 0.04 * S0 / 0.30   # ≈ 13.33


# ================================================================== #
# Training: stock-only                                               #
# ================================================================== #
def train_stock_only(loss_fn, label, epochs=EPOCHS):
    net = make_stock_only_net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    alpha_start, alpha_end = 0.80, 0.95

    net.train()
    for epoch in range(epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / max(epochs - 1, 1)
        S_b, payoff_fn = simulate_heston(N=BATCH_SIZE, device=device)
        T = S_b.shape[1] - 1

        pnl_list = []
        for i in range(BATCH_SIZE):
            pnl      = torch.zeros((), device=device)
            prev_dS  = torch.zeros((), device=device)

            for t in range(T):
                state = torch.stack([S_b[i, t],
                                     torch.tensor(t / T, device=device),
                                     prev_dS])
                dS = torch.tanh(net(state)[0]) * 5.0
                pnl    += prev_dS * (S_b[i, t + 1] - S_b[i, t])
                pnl    -= COST_RATE * torch.abs(dS - prev_dS)
                prev_dS = dS

            pnl -= payoff_fn(S_b[i, -1])
            pnl_list.append(pnl)

        pnls = torch.stack(pnl_list)
        loss = loss_fn(pnls, alpha)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
        opt.step()

    net.eval()
    return net


# ================================================================== #
# Training: stock + VS                                               #
# ================================================================== #
def train_vs(loss_fn, label, epochs=EPOCHS):
    net = make_vs_net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    alpha_start, alpha_end = 0.80, 0.95

    net.train()
    for epoch in range(epochs):
        alpha = alpha_start + (alpha_end - alpha_start) * epoch / max(epochs - 1, 1)
        S_b, VS_b, payoff_fn = simulate_heston_with_var_swap(N=BATCH_SIZE, device=device)
        T = S_b.shape[1] - 1

        pnl_list = []
        for i in range(BATCH_SIZE):
            pnl     = torch.zeros((), device=device)
            prev_dS = torch.zeros((), device=device)
            prev_dV = torch.zeros((), device=device)

            for t in range(T):
                state = torch.stack([S_b[i, t]  / S0,
                                     VS_b[i, t] / VS0,
                                     torch.tensor(t / T, device=device),
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

            pnl -= payoff_fn(S_b[i, -1])
            pnl_list.append(pnl)

        pnls = torch.stack(pnl_list)
        loss = loss_fn(pnls, alpha)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
        opt.step()

    net.eval()
    return net


# ================================================================== #
# Evaluation helpers                                                 #
# ================================================================== #
@torch.no_grad()
def eval_stock_only_net(net, S_test, payoff_fn):
    N, Tt = S_test.shape[0], S_test.shape[1] - 1
    pnls = []
    for i in range(N):
        pnl     = torch.zeros((), device=device)
        prev_dS = torch.zeros((), device=device)
        for t in range(Tt):
            state = torch.stack([S_test[i, t],
                                 torch.tensor(t / Tt, device=device),
                                 prev_dS])
            dS = torch.tanh(net(state)[0]) * 5.0
            pnl    += prev_dS * (S_test[i, t + 1] - S_test[i, t])
            pnl    -= COST_RATE * torch.abs(dS - prev_dS)
            prev_dS = dS
        pnl -= payoff_fn(S_test[i, -1])
        pnls.append(pnl.item())
    return np.array(pnls)


@torch.no_grad()
def eval_vs_net(net, S_test, VS_test, payoff_fn):
    N, Tt = S_test.shape[0], S_test.shape[1] - 1
    pnls = []
    for i in range(N):
        pnl     = torch.zeros((), device=device)
        prev_dS = torch.zeros((), device=device)
        prev_dV = torch.zeros((), device=device)
        for t in range(Tt):
            state = torch.stack([S_test[i, t]  / S0,
                                 VS_test[i, t] / VS0,
                                 torch.tensor(t / Tt, device=device),
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
        pnl -= payoff_fn(S_test[i, -1])
        pnls.append(pnl.item())
    return np.array(pnls)


def metrics(pnl):
    cv = cvar_np(pnl)
    m  = float(pnl.mean())
    s  = float(pnl.std())
    sh = m / s if s > 0 else 0.0
    return {"CVaR": cv, "Mean": m, "Std": s, "Sharpe": sh}


# ================================================================== #
# Fixed test set                                                     #
# ================================================================== #
print("Generating fixed test set ...")
torch.manual_seed(999)
S_test, VS_test, payoff_fn_test = simulate_heston_with_var_swap(N=N_TEST, device=device)
pnl_delta = delta_hedge_pnl(S_test, payoff_fn_test, K=100.0).cpu().numpy()
m_delta   = metrics(pnl_delta)
print(f"  Delta hedge  CVaR={m_delta['CVaR']:.2f}  Mean={m_delta['Mean']:.2f}  Std={m_delta['Std']:.2f}")


# ================================================================== #
# Training & evaluation loop (8 policies)                           #
# ================================================================== #
all_results = {}

for obj_name, loss_fn in OBJECTIVES.items():
    print(f"\n{'-'*55}")
    print(f"  Objective: {obj_name}")
    print(f"{'-'*55}")

    # ---- Stock-only ----
    print(f"  Training stock-only ({EPOCHS} epochs) ...")
    net_s  = train_stock_only(loss_fn, obj_name)
    pnl_s  = eval_stock_only_net(net_s, S_test, payoff_fn_test)
    m_s    = metrics(pnl_s)
    print(f"    Stock-only CVaR={m_s['CVaR']:.2f}  Mean={m_s['Mean']:.2f}  Std={m_s['Std']:.2f}")

    # ---- Stock + VS ----
    print(f"  Training stock+VS   ({EPOCHS} epochs) ...")
    net_v  = train_vs(loss_fn, obj_name)
    pnl_v  = eval_vs_net(net_v, S_test, VS_test, payoff_fn_test)
    m_v    = metrics(pnl_v)
    print(f"    Stock+VS   CVaR={m_v['CVaR']:.2f}  Mean={m_v['Mean']:.2f}  Std={m_v['Std']:.2f}")

    all_results[obj_name] = {
        "stock": {"pnl": pnl_s, **m_s},
        "vs":    {"pnl": pnl_v, **m_v},
    }


# ================================================================== #
# Results table                                                      #
# ================================================================== #
print(f"\n\n{'='*75}")
print(f"{'Policy':<30} {'CVaR':>8} {'Mean':>8} {'Std':>8} {'Sharpe':>8} {'DeltaCVaR':>8}")
print(f"{'='*75}")
print(f"{'Delta Hedge (baseline)':<30} {m_delta['CVaR']:>8.2f} {m_delta['Mean']:>8.2f} {m_delta['Std']:>8.2f} {m_delta['Sharpe']:>8.3f} {'--':>8}")

for obj_name, res in all_results.items():
    for iset, key in [("stock-only", "stock"), ("stock+VS", "vs")]:
        m    = res[key]
        dCV  = m["CVaR"] - m_delta["CVaR"]
        name = f"{obj_name} ({iset})"
        print(f"{name:<30} {m['CVaR']:>8.2f} {m['Mean']:>8.2f} {m['Std']:>8.2f} {m['Sharpe']:>8.3f} {dCV:>+8.2f}")

print(f"{'='*75}")


# ================================================================== #
# Plot A: CVaR comparison bar chart                                  #
# ================================================================== #
obj_names = list(OBJECTIVES.keys())
x = np.arange(len(obj_names))
w = 0.30

COLOUR_S = "#2196F3"   # stock-only -- blue
COLOUR_V = "#FF5722"   # stock+VS   -- orange

fig, ax = plt.subplots(figsize=(11, 5))

cvar_s_vals = [all_results[n]["stock"]["CVaR"] for n in obj_names]
cvar_v_vals = [all_results[n]["vs"]["CVaR"]    for n in obj_names]

b1 = ax.bar(x - w / 2, cvar_s_vals, width=w, color=COLOUR_S,
            edgecolor="black", linewidth=0.5, label="Stock-only")
b2 = ax.bar(x + w / 2, cvar_v_vals, width=w, color=COLOUR_V,
            edgecolor="black", linewidth=0.5, label="Stock + VS")

ax.axhline(m_delta["CVaR"], color="black", linewidth=1.5, linestyle="--",
           label=f"BS Delta baseline ({m_delta['CVaR']:.2f})")

for bar, val in zip(list(b1) + list(b2),
                    cvar_s_vals + cvar_v_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25,
            f"{val:.1f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(obj_names, fontsize=10)
ax.set_ylabel("CVaR at 95%  (higher = better)", fontsize=9)
ax.set_title("Risk Measure Comparison: CVaR Performance\n"
             "4 Objectives x 2 Instrument Sets  (consistent hyperparams: 500 epochs, cost=0.02%)",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/risk_comparison_cvar.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/risk_comparison_cvar.png")


# ================================================================== #
# Plot B: Mean vs Std scatter -- Pareto frontier view                 #
# ================================================================== #
fig2, ax2 = plt.subplots(figsize=(8, 6))

markers = {"stock": "o", "vs": "D"}
colours_obj = {
    "CVaR(95%)": "#E91E63",
    "Variance":  "#9C27B0",
    "Entropic":  "#00BCD4",
    "Mean-Std":  "#FF9800",
}

for obj_name, res in all_results.items():
    col = colours_obj[obj_name]
    for iset, key, mk in [("stock", "stock", "o"), ("vs", "vs", "D")]:
        m = res[key]
        label = f"{obj_name} ({'stock+VS' if iset == 'vs' else 'stock-only'})"
        ax2.scatter(m["Std"], m["Mean"], color=col, marker=mk, s=120,
                    zorder=5, edgecolors="black", linewidth=0.8, label=label)

# Delta hedge
ax2.scatter(m_delta["Std"], m_delta["Mean"], color="black", marker="x",
            s=150, zorder=10, linewidth=2, label="Delta Hedge")

ax2.set_xlabel("Std(PnL)  (lower = better)", fontsize=10)
ax2.set_ylabel("Mean(PnL)  (higher = better)", fontsize=10)
ax2.set_title("Mean-Std Trade-off Across Risk Objectives\n"
              "Upper-left = Pareto-optimal (high mean, low std)",
              fontsize=10)

# Annotate legend compactly
legend_elements = [
    mpatches.Patch(color=c, label=n) for n, c in colours_obj.items()
] + [
    plt.Line2D([0], [0], marker="o", color="gray", label="Stock-only",
               linestyle="none", markersize=8),
    plt.Line2D([0], [0], marker="D", color="gray", label="Stock+VS",
               linestyle="none", markersize=8),
    plt.Line2D([0], [0], marker="x", color="black", label="Delta hedge",
               linestyle="none", markersize=10, markeredgewidth=2),
]
ax2.legend(handles=legend_elements, fontsize=8, loc="best")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/risk_comparison_scatter.png", dpi=150, bbox_inches="tight")
print("Saved: results/risk_comparison_scatter.png")

plt.show()
print("\nDone.")
