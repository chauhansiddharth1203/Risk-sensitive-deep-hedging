"""
main_learning_curve.py
-----------------------
Week 4 Experiment D: Learning curve analysis.

Trains stock+VS and stock-only CVaR policies for 1000 epochs and evaluates
CVaR on a fixed test set every 25 epochs.

Key questions:
  1. How many training epochs until stock+VS beats the delta hedge?
  2. Does the stock-only policy ever beat the delta hedge? (Answer: no)
  3. How stable is training -- how much does CVaR fluctuate epoch to epoch?
  4. What is the gap between stock+VS and stock-only over time?

Produces:
  results/learning_curve.png      -- CVaR vs epoch for both policies
  results/learning_curve_gap.png  -- VS advantage over stock-only vs epoch
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.network import HedgingPolicy
from policy.network_varswap import HedgingPolicyVarSwap
from market.heston_with_var_swap import simulate_heston_with_var_swap
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_EPOCHS  = 1000
EVAL_EVERY    = 25
BATCH_SIZE    = 64
N_TEST        = 3000
ALPHA         = 0.95
COST_RATE     = 0.0002


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ------------------------------------------------------------------ #
# Fixed test set (same throughout training)                          #
# ------------------------------------------------------------------ #
print("Generating fixed test set ...")
torch.manual_seed(999)
S_test, VS_test, payoff_test = simulate_heston_with_var_swap(N=N_TEST, device=device)
pnl_delta = delta_hedge_pnl(S_test, payoff_test, K=100.0).cpu().numpy()
delta_cvar = cvar_np(pnl_delta)
print(f"  Delta hedge CVaR (baseline): {delta_cvar:.2f}")


# ------------------------------------------------------------------ #
# Helper: evaluate a policy on the fixed test set                    #
# ------------------------------------------------------------------ #
def eval_stock_only(policy):
    policy.eval()
    pnls = []
    with torch.no_grad():
        for i in range(N_TEST):
            pnls.append(policy.rollout(S_test[i], payoff_test).item())
    policy.train()
    return cvar_np(np.array(pnls))


def eval_varswap(policy):
    policy.eval()
    pnls = []
    with torch.no_grad():
        for i in range(N_TEST):
            pnls.append(policy.rollout(S_test[i], VS_test[i], payoff_test).item())
    policy.train()
    return cvar_np(np.array(pnls))


# ------------------------------------------------------------------ #
# Initialise both policies                                           #
# ------------------------------------------------------------------ #
torch.manual_seed(42)

policy_stock = HedgingPolicy(state_dim=3, action_dim=1, cost_rate=COST_RATE).to(device)
policy_vs    = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)

opt_stock = torch.optim.Adam(policy_stock.parameters(), lr=3e-4)
opt_vs    = torch.optim.Adam(policy_vs.parameters(),    lr=3e-4)

alpha_start, alpha_end = 0.80, 0.95


# ------------------------------------------------------------------ #
# Training loop with periodic evaluation                             #
# ------------------------------------------------------------------ #
epochs_log   = []
cvar_stock_log = []
cvar_vs_log    = []
loss_stock_log = []
loss_vs_log    = []

print(f"\nTraining both policies for {TOTAL_EPOCHS} epochs ...")
print(f"Evaluating on {N_TEST} fixed test paths every {EVAL_EVERY} epochs.\n")

for epoch in range(TOTAL_EPOCHS + 1):
    alpha = alpha_start + (alpha_end - alpha_start) * min(epoch, 300) / 300

    # -- Stock-only training step --
    if epoch < TOTAL_EPOCHS:
        policy_stock.train()
        S_b, payoff_b = torch.zeros(BATCH_SIZE, 31), None

        from market.heston import simulate_heston
        S_b, payoff_b = simulate_heston(N=BATCH_SIZE, device=device)
        pnls_s = torch.stack([policy_stock.rollout(S_b[i], payoff_b)
                               for i in range(BATCH_SIZE)])
        loss_s = -cvar(pnls_s, alpha)
        opt_stock.zero_grad()
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(policy_stock.parameters(), max_norm=1.0)
        opt_stock.step()

        # -- Stock+VS training step --
        policy_vs.train()
        S_b2, VS_b2, payoff_b2 = simulate_heston_with_var_swap(N=BATCH_SIZE, device=device)
        pnls_v = torch.stack([policy_vs.rollout(S_b2[i], VS_b2[i], payoff_b2)
                               for i in range(BATCH_SIZE)])
        loss_v = -cvar(pnls_v, alpha)
        opt_vs.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(policy_vs.parameters(), max_norm=1.0)
        opt_vs.step()

    # -- Evaluation --
    if epoch % EVAL_EVERY == 0:
        cv_stock = eval_stock_only(policy_stock)
        cv_vs    = eval_varswap(policy_vs)

        epochs_log.append(epoch)
        cvar_stock_log.append(cv_stock)
        cvar_vs_log.append(cv_vs)

        if epoch < TOTAL_EPOCHS:
            loss_stock_log.append(loss_s.item())
            loss_vs_log.append(loss_v.item())
        else:
            loss_stock_log.append(loss_stock_log[-1])
            loss_vs_log.append(loss_vs_log[-1])

        print(f"  Epoch {epoch:4d}  alpha={alpha:.3f}  "
              f"Stock CVaR={cv_stock:.2f}  VS CVaR={cv_vs:.2f}  "
              f"Delta={delta_cvar:.2f}")


# ------------------------------------------------------------------ #
# Find epoch where stock+VS first beats delta hedge                  #
# ------------------------------------------------------------------ #
beat_epoch = None
for ep, cv in zip(epochs_log, cvar_vs_log):
    if cv > delta_cvar:
        beat_epoch = ep
        break

print(f"\nStock+VS first beats delta hedge at epoch: {beat_epoch}")
print(f"Stock-only final CVaR: {cvar_stock_log[-1]:.2f}  (delta hedge: {delta_cvar:.2f})")
print(f"Stock+VS  final CVaR: {cvar_vs_log[-1]:.2f}")


# ------------------------------------------------------------------ #
# Plot A: Learning curves                                            #
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax = axes[0]
ax.plot(epochs_log, cvar_stock_log, "-",  color="#2196F3", linewidth=1.8,
        label="Stock only", alpha=0.85)
ax.plot(epochs_log, cvar_vs_log,    "-",  color="#FF5722", linewidth=1.8,
        label="Stock + VS", alpha=0.85)
ax.axhline(delta_cvar, color="black", linewidth=1.5, linestyle="--",
           label=f"Delta hedge baseline ({delta_cvar:.2f})")

if beat_epoch is not None:
    ax.axvline(beat_epoch, color="#FF5722", linewidth=1.0, linestyle=":",
               alpha=0.7)
    ax.text(beat_epoch + 10, delta_cvar + 0.3,
            f"VS beats delta\nat epoch {beat_epoch}",
            color="#FF5722", fontsize=8)

# Shade the region where VS beats delta hedge
epochs_arr   = np.array(epochs_log)
cvar_vs_arr  = np.array(cvar_vs_log)
above_mask   = cvar_vs_arr > delta_cvar
ax.fill_between(epochs_arr, cvar_vs_arr, delta_cvar,
                where=above_mask, alpha=0.12, color="#FF5722",
                label="VS beats delta hedge")

ax.set_ylabel("CVaR at 95%  (higher = better)", fontsize=9)
ax.set_title("Learning Curve: How Quickly Does Deep Hedging Beat the Delta Hedge?\n"
             "CVaR on 3,000 Fixed Test Paths vs Training Epoch",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax2 = axes[1]
gap = np.array(cvar_vs_log) - np.array(cvar_stock_log)
ax2.plot(epochs_log, gap, "-", color="#4CAF50", linewidth=2.0,
         label="VS advantage over stock-only")
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax2.fill_between(epochs_log, gap, 0,
                 where=gap > 0, alpha=0.15, color="#4CAF50")
ax2.set_xlabel("Training Epoch", fontsize=10)
ax2.set_ylabel("CVaR gap  (VS minus stock-only)", fontsize=9)
ax2.set_title("How Quickly Does the Variance Swap Start Helping?", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/learning_curve.png")

plt.show()
print("\nDone.")
