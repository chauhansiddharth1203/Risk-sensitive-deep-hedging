"""
main_robust_training.py
-----------------------
Week 5 Experiment D: Domain Randomization for Robust Deep Hedging

═══════════════════════════════════════════════════════════════════════
MOTIVATION
═══════════════════════════════════════════════════════════════════════
Experiment B (main_misspecification.py) showed a striking failure:

    Environment          Stock+VS CVaR
    Heston (σ_v=0.30)    -12.13   <- trained here
    Crisis (σ_v=0.80)    -45.07   <- 4x worse; also worse than stock-only

The stock+VS policy learned a vega hedge ratio specific to σ_v=0.30.
When σ_v jumps to 0.80 at test time, that ratio is off by a factor of
~2.7 and the mis-sized VS position amplifies losses.

═══════════════════════════════════════════════════════════════════════
SOLUTION: Domain Randomization
═══════════════════════════════════════════════════════════════════════
Train on RANDOMLY sampled Heston parameters in every batch:

    κ     in [0.50, 3.00]
    theta     in [0.02, 0.12]
    σ_v   in [0.20, 0.80]   <- critical: spans normal AND crisis
    ρ     in [-0.90, -0.40]
    v0    in [0.02, 0.10]

The policy is forced to learn a hedging rule that works for ALL
parameter regimes, rather than memorising one specific regime.

═══════════════════════════════════════════════════════════════════════
HYPOTHESIS
═══════════════════════════════════════════════════════════════════════
H1: Robust policy ≈ naive on in-sample Heston (minimal accuracy cost)
H2: Robust policy ≫ naive on Crisis Heston  (closes failure)
H3: Robust policy ≈ naive on SABR           (partial transfer)

═══════════════════════════════════════════════════════════════════════
PRODUCES
═══════════════════════════════════════════════════════════════════════
results/robust_training_cvar.png       -- 3 envs x 3 policies bar chart
results/robust_training_gap_closure.png -- naive vs robust degradation
results/varswap_cvar_robust.pth         -- the trained robust policy
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from policy.network_varswap import HedgingPolicyVarSwap
from market.heston_with_var_swap import simulate_heston_with_var_swap
from market.heston_random import simulate_heston_random
from market.sabr import simulate_sabr
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar as cvar_torch

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# -- Hyperparameters ----------------------------------------------- #
COST_RATE  = 0.0002
ALPHA      = 0.95
EPOCHS     = 800
BATCH_SIZE = 64
N_TEST     = 2000
S0         = 100.0
VS0        = 0.04 * S0 / 0.30   # ≈ 13.33 (training scale)


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# =================================================================== #
# Crisis Heston (for testing)                                         #
# =================================================================== #
def simulate_crisis_heston(N=1000, device="cpu"):
    kappa, theta, sigma_v, rho, v0 = 0.5, 0.12, 0.80, -0.85, 0.09
    T, S0_c, K_c = 30, 100.0, 100.0
    dt      = 1.0 / T
    rho_bar = (1.0 - rho ** 2) ** 0.5

    S  = torch.zeros(N, T + 1, device=device)
    v  = torch.zeros(N, T + 1, device=device)
    VS = torch.zeros(N, T + 1, device=device)

    S[:, 0]  = S0_c
    v[:, 0]  = v0
    VS[:, 0] = v0 * S0_c / 0.30   # use SAME scaling as training

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
        VS[:, t + 1] = v[:, t + 1] * S0_c / 0.30

    def payoff_fn(S_T):
        return torch.clamp(S_T - K_c, min=0.0)

    return S, VS, payoff_fn


# =================================================================== #
# Train robust policy with domain randomization                       #
# =================================================================== #
def train_robust_policy(epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train stock+VS policy where each batch uses a freshly sampled
    Heston parameter set.
    """
    policy = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=3e-4)

    alpha_start, alpha_end = 0.80, 0.95
    policy.train()

    log_epoch = []
    log_loss  = []
    log_sigma_v = []

    for epoch in range(epochs):
        alpha_train = alpha_start + (alpha_end - alpha_start) * epoch / max(epochs - 1, 1)

        # Sample fresh Heston params for this batch
        S_b, VS_b, _, params = simulate_heston_random(N=batch_size, device=device)
        sigma_v_this = params[2]

        pnls = torch.stack([
            policy.rollout(S_b[i], VS_b[i], lambda S_T: torch.clamp(S_T - 100.0, min=0.0))
            for i in range(batch_size)
        ])
        loss = -cvar_torch(pnls, alpha_train)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        if epoch % 50 == 0:
            log_epoch.append(epoch)
            log_loss.append(loss.item())
            log_sigma_v.append(sigma_v_this)
            print(f"  epoch {epoch:4d}  σ_v={sigma_v_this:.3f}  loss={loss.item():.3f}")

    policy.eval()
    return policy, log_epoch, log_loss, log_sigma_v


# =================================================================== #
# Evaluate policy on given paths                                      #
# =================================================================== #
@torch.no_grad()
def evaluate_policy(policy, S, VS, payoff_fn):
    N = S.shape[0]
    pnls = []
    for i in range(N):
        pnls.append(policy.rollout(S[i], VS[i], payoff_fn).item())
    return np.array(pnls)


# =================================================================== #
# MAIN                                                                 #
# =================================================================== #
print("=" * 65)
print("  ROBUST DEEP HEDGING -- DOMAIN RANDOMIZATION")
print("=" * 65)

# ---- 1. Load naive policy (from Experiment B baseline) ----
print("\n[1/3] Loading NAIVE policy (trained on fixed Heston σ_v=0.30) ...")
policy_naive = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
policy_naive.load_state_dict(
    torch.load("results/varswap_cvar.pth", map_location=device)
)
policy_naive.eval()
print("  Loaded: results/varswap_cvar.pth")

# ---- 2. Train robust policy ----
print(f"\n[2/3] Training ROBUST policy with domain randomization ({EPOCHS} epochs) ...")
policy_robust, log_epoch, log_loss, log_sig = train_robust_policy()
torch.save(policy_robust.state_dict(), "results/varswap_cvar_robust.pth")
print("  Saved: results/varswap_cvar_robust.pth")

# ---- 3. Evaluate both policies on 3 environments ----
print(f"\n[3/3] Evaluating both policies on 3 environments (N_TEST={N_TEST}) ...")

ENVIRONMENTS = [
    ("Heston\n(training)",          lambda: simulate_heston_with_var_swap(N=N_TEST, device=device)),
    ("SABR\n(misspecified)",        lambda: simulate_sabr(N=N_TEST, device=device)),
    ("Crisis Heston\n(σ_v=0.8)",    lambda: simulate_crisis_heston(N=N_TEST, device=device)),
]

results = []
for name, sim in ENVIRONMENTS:
    torch.manual_seed(0)
    S, VS, payoff_fn = sim()

    pnl_delta  = delta_hedge_pnl(S, payoff_fn, K=100.0).cpu().numpy()
    pnl_naive  = evaluate_policy(policy_naive,  S, VS, payoff_fn)
    pnl_robust = evaluate_policy(policy_robust, S, VS, payoff_fn)

    c_d = cvar_np(pnl_delta)
    c_n = cvar_np(pnl_naive)
    c_r = cvar_np(pnl_robust)

    results.append({
        "name": name,
        "cvar_delta":  c_d,
        "cvar_naive":  c_n,
        "cvar_robust": c_r,
        "pnl_delta":   pnl_delta,
        "pnl_naive":   pnl_naive,
        "pnl_robust":  pnl_robust,
    })


# ---- Results table ----
print(f"\n{'='*70}")
print(f"{'Environment':<22}{'BS Delta':>10}{'Naive VS':>12}{'Robust VS':>12}{'Delta (R-N)':>12}")
print(f"{'='*70}")
for r in results:
    name = r["name"].replace("\n", " ")
    delta_rn = r["cvar_robust"] - r["cvar_naive"]
    print(f"{name:<22}{r['cvar_delta']:>10.2f}{r['cvar_naive']:>12.2f}"
          f"{r['cvar_robust']:>12.2f}{delta_rn:>+12.2f}")
print(f"{'='*70}")


# =================================================================== #
# Plot A: 3x3 CVaR bar chart                                         #
# =================================================================== #
env_names = [r["name"] for r in results]
x = np.arange(len(env_names))
w = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(x - w, [r["cvar_delta"]  for r in results], width=w, color="#607D8B",
       edgecolor="black", linewidth=0.5, label="BS Delta")
ax.bar(x,     [r["cvar_naive"]  for r in results], width=w, color="#FF5722",
       edgecolor="black", linewidth=0.5, label="Naive VS (fixed σ_v=0.30)")
ax.bar(x + w, [r["cvar_robust"] for r in results], width=w, color="#4CAF50",
       edgecolor="black", linewidth=0.5, label="Robust VS (randomized)")

ax.set_xticks(x)
ax.set_xticklabels(env_names, fontsize=9)
ax.set_ylabel("CVaR at 95% (higher = better)", fontsize=9)
ax.set_title("Domain-Randomized Training Closes the Crisis Gap\n"
             "Robust policy is trained on random Heston parameters",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---- improvement bar: robust vs naive ----
ax2  = axes[1]
gaps = [r["cvar_robust"] - r["cvar_naive"] for r in results]
cols = ["#4CAF50" if g > 0 else "#F44336" for g in gaps]

bars = ax2.bar(env_names, gaps, color=cols, edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")

for bar, val in zip(bars, gaps):
    ypos = val + 0.5 if val >= 0 else val - 1.5
    ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
             f"{val:+.2f}", ha="center", fontsize=11, fontweight="bold")

ax2.set_ylabel("Robust - Naive CVaR", fontsize=9)
ax2.set_title("Robustness Gain Across Environments\n"
              "Positive = robust training helped", fontsize=10)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/robust_training_cvar.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/robust_training_cvar.png")


# =================================================================== #
# Plot B: Degradation from in-sample                                  #
# =================================================================== #
in_sample_naive  = results[0]["cvar_naive"]
in_sample_robust = results[0]["cvar_robust"]

fig2, ax3 = plt.subplots(figsize=(10, 5))
env_short = ["Heston", "SABR", "Crisis"]
deg_naive  = [r["cvar_naive"]  - in_sample_naive  for r in results]
deg_robust = [r["cvar_robust"] - in_sample_robust for r in results]

x2 = np.arange(len(env_short))
w2 = 0.35

ax3.bar(x2 - w2 / 2, deg_naive,  width=w2, color="#FF5722",
        edgecolor="black", linewidth=0.5, label="Naive (fixed training)")
ax3.bar(x2 + w2 / 2, deg_robust, width=w2, color="#4CAF50",
        edgecolor="black", linewidth=0.5, label="Robust (randomized training)")
ax3.axhline(0, color="black", linewidth=1.0, linestyle="--")

ax3.set_xticks(x2)
ax3.set_xticklabels(env_short, fontsize=10)
ax3.set_ylabel("DeltaCVaR from in-sample Heston", fontsize=9)
ax3.set_title("Generalization Gap: Naive vs Robust Training\n"
              "Smaller bars = more robust across environments", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/robust_training_gap_closure.png", dpi=150, bbox_inches="tight")
print("Saved: results/robust_training_gap_closure.png")

plt.show()
print("\nDone.")
