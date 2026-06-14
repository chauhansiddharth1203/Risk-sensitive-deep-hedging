"""
main_hedge_analysis.py
-----------------------
Week 4 Experiment B: Hedge ratio interpretability.

Visualises what the trained neural network has actually learned:
  - delta_S  : stock position as a function of (S, v, t)
  - delta_V  : variance swap position as a function of (S, v, t)

Compares with:
  - Black-Scholes delta  N(d1) at instantaneous vol sqrt(v)
  - BS vega              S * N'(d1) * sqrt(tau)  (proxy for VS position)

Key questions:
  1. Does the network's delta_S track the BS delta? (sanity check)
  2. Does delta_V increase with v? (more volatile = hold more VS to hedge)
  3. Where does the network deviate most from BS delta? (novel hedging)

Produces:
  results/hedge_delta_S_surface.png  -- stock hedge ratio heatmap
  results/hedge_delta_V_surface.png  -- VS hedge ratio heatmap
  results/hedge_vs_bs.png            -- network vs BS delta comparison
  results/hedge_delta_V_vs_v.png     -- VS position vs instantaneous variance
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from policy.network_varswap import HedgingPolicyVarSwap

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

S0     = 100.0
K      = 100.0
T_full = 30
VS0    = 0.04 * S0 / 0.3    # ≈ 13.33


# ------------------------------------------------------------------ #
# Helper: BS delta and vega                                          #
# ------------------------------------------------------------------ #
def bs_delta(S, K, tau, v):
    """Black-Scholes delta using instantaneous variance v."""
    if tau <= 1e-6:
        return 1.0 if S > K else 0.0
    sigma = np.sqrt(v)
    d1 = (np.log(S / K) + 0.5 * v * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bs_vega(S, K, tau, v):
    """Black-Scholes vega (∂C/∂σ), proxy for VS position sizing."""
    if tau <= 1e-6:
        return 0.0
    sigma = np.sqrt(v)
    d1 = (np.log(S / K) + 0.5 * v * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)


# ------------------------------------------------------------------ #
# Load trained policy                                                 #
# ------------------------------------------------------------------ #
print("Loading trained varswap CVaR policy ...")
policy = HedgingPolicyVarSwap(cost_rate=0.0002).to(device)
policy.load_state_dict(torch.load("results/varswap_cvar.pth", map_location=device))
policy.eval()


# ------------------------------------------------------------------ #
# Query network at a grid of (S, v) for three time points            #
# ------------------------------------------------------------------ #
S_grid  = np.linspace(75, 125, 40)
v_grid  = np.linspace(0.01, 0.16, 40)
t_fracs = [0.1, 0.5, 0.9]    # early, mid, near expiry

# Containers
net_delta_S = {t: np.zeros((len(v_grid), len(S_grid))) for t in t_fracs}
net_delta_V = {t: np.zeros((len(v_grid), len(S_grid))) for t in t_fracs}
bs_delta_   = {t: np.zeros((len(v_grid), len(S_grid))) for t in t_fracs}
bs_vega_    = {t: np.zeros((len(v_grid), len(S_grid))) for t in t_fracs}

print("Querying network over (S, v, t) grid ...")

with torch.no_grad():
    for ti, t_frac in enumerate(t_fracs):
        tau = 1.0 - t_frac    # remaining time (fraction of 1 year)
        for vi, v in enumerate(v_grid):
            VS_val = v * S0 / 0.3    # VS at this variance level

            for si, S in enumerate(S_grid):
                state = torch.tensor([
                    S     / S0,
                    VS_val / VS0,
                    t_frac,
                    0.0,      # prev_delta_S = 0 (clean state)
                    0.0,      # prev_delta_V = 0
                ], dtype=torch.float32, device=device)

                action = policy(state)
                d_S = torch.tanh(action[0]).item() * 5.0
                d_V = torch.tanh(action[1]).item() * 5.0

                net_delta_S[t_frac][vi, si] = d_S
                net_delta_V[t_frac][vi, si] = d_V
                bs_delta_[t_frac][vi, si]   = bs_delta(S, K, tau, v)
                bs_vega_[t_frac][vi, si]    = bs_vega(S, K, tau, v)


# ------------------------------------------------------------------ #
# Plot A: delta_S heatmaps at 3 time points                          #
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, t_frac in zip(axes, t_fracs):
    im = ax.pcolormesh(S_grid, v_grid, net_delta_S[t_frac],
                       cmap="RdBu_r", vmin=-1, vmax=2)
    ax.set_title(f"t/T = {t_frac:.1f}  (τ = {1-t_frac:.1f})")
    ax.set_xlabel("Stock price S")
    ax.axvline(K, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_ylabel("Instantaneous variance v") if ax == axes[0] else None

plt.colorbar(im, ax=axes[-1], label="δ_S (stock position)")
fig.suptitle("Network Stock Position δ_S  vs  (S, v, t)\nBlack dashed = strike K=100",
             fontsize=11)
plt.tight_layout()
plt.savefig("results/hedge_delta_S_surface.png", dpi=150, bbox_inches="tight")
print("Saved: results/hedge_delta_S_surface.png")


# ------------------------------------------------------------------ #
# Plot B: delta_V heatmaps at 3 time points                          #
# ------------------------------------------------------------------ #
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

for ax, t_frac in zip(axes2, t_fracs):
    im2 = ax.pcolormesh(S_grid, v_grid, net_delta_V[t_frac],
                        cmap="PuOr", vmin=-2, vmax=2)
    ax.set_title(f"t/T = {t_frac:.1f}  (τ = {1-t_frac:.1f})")
    ax.set_xlabel("Stock price S")
    ax.axvline(K, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_ylabel("Instantaneous variance v") if ax == axes2[0] else None

plt.colorbar(im2, ax=axes2[-1], label="δ_V (variance swap position)")
fig2.suptitle("Network Variance Swap Position δ_V  vs  (S, v, t)\nBlack dashed = strike K=100",
              fontsize=11)
plt.tight_layout()
plt.savefig("results/hedge_delta_V_surface.png", dpi=150, bbox_inches="tight")
print("Saved: results/hedge_delta_V_surface.png")


# ------------------------------------------------------------------ #
# Plot C: Network delta_S vs BS delta at mid-time (t/T = 0.5)       #
# ------------------------------------------------------------------ #
t_mid = 0.5
# Slice at ATM-ish variance (v ≈ 0.04)
v_idx_mid = np.argmin(np.abs(v_grid - 0.04))

net_slice  = net_delta_S[t_mid][v_idx_mid, :]
bs_slice   = bs_delta_[t_mid][v_idx_mid, :]

# Slice at higher variance (v ≈ 0.09)
v_idx_high = np.argmin(np.abs(v_grid - 0.09))
net_slice_h = net_delta_S[t_mid][v_idx_high, :]
bs_slice_h  = bs_delta_[t_mid][v_idx_high, :]

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(S_grid, bs_slice,   "--",  color="#607D8B", linewidth=1.8,
         label="BS delta  (v=0.04, σ=20%)")
ax3.plot(S_grid, net_slice,  "o-",  color="#FF5722", linewidth=2, markersize=4,
         label="Network δ_S  (v=0.04)")
ax3.plot(S_grid, bs_slice_h, "--",  color="#90A4AE", linewidth=1.8,
         label="BS delta  (v=0.09, σ=30%)")
ax3.plot(S_grid, net_slice_h, "s-", color="#FF9800", linewidth=2, markersize=4,
         label="Network δ_S  (v=0.09)")
ax3.axvline(K, color="black", linewidth=1.0, linestyle=":", alpha=0.7)
ax3.set_xlabel("Stock price S", fontsize=10)
ax3.set_ylabel("Stock position δ_S", fontsize=10)
ax3.set_title("Network Stock Position vs Black-Scholes Delta\nt/T=0.5, two variance levels",
              fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/hedge_vs_bs.png", dpi=150, bbox_inches="tight")
print("Saved: results/hedge_vs_bs.png")


# ------------------------------------------------------------------ #
# Plot D: delta_V vs instantaneous variance v (at ATM, 3 time pts)  #
# ------------------------------------------------------------------ #
S_idx_atm = np.argmin(np.abs(S_grid - S0))   # ATM slice

fig4, ax4 = plt.subplots(figsize=(7, 5))
colours_t = ["#1565C0", "#E65100", "#2E7D32"]

for t_frac, col in zip(t_fracs, colours_t):
    dv_slice = net_delta_V[t_frac][:, S_idx_atm]
    ax4.plot(v_grid, dv_slice, "o-", color=col, linewidth=2, markersize=5,
             label=f"t/T = {t_frac:.1f}")

ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax4.set_xlabel("Instantaneous variance v  (vol = √v)", fontsize=10)
ax4.set_ylabel("Variance swap position δ_V", fontsize=10)
ax4.set_title("Network VS Position vs Instantaneous Variance\nat ATM (S=100), three time points",
              fontsize=10)

# Add secondary x-axis for vol
ax4b = ax4.twiny()
v_ticks = [0.01, 0.04, 0.09, 0.16]
ax4b.set_xlim(ax4.get_xlim())
ax4b.set_xticks(v_ticks)
ax4b.set_xticklabels([f"{np.sqrt(v):.0%}" for v in v_ticks])
ax4b.set_xlabel("Implied vol  (√v)", fontsize=9)

ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/hedge_delta_V_vs_v.png", dpi=150, bbox_inches="tight")
print("Saved: results/hedge_delta_V_vs_v.png")

plt.show()

# ------------------------------------------------------------------ #
# Print summary statistics                                            #
# ------------------------------------------------------------------ #
print("\n=== Hedge Ratio Summary (t/T=0.5, ATM) ===")
t_mid = 0.5
si_atm = np.argmin(np.abs(S_grid - S0))
for v_label, v_val in [("v=0.04 (σ=20%)", 0.04), ("v=0.09 (σ=30%)", 0.09)]:
    vi = np.argmin(np.abs(v_grid - v_val))
    d_S = net_delta_S[t_mid][vi, si_atm]
    d_V = net_delta_V[t_mid][vi, si_atm]
    d_bs = bs_delta_[t_mid][vi, si_atm]
    print(f"  {v_label}: δ_S={d_S:.3f}  BS_delta={d_bs:.3f}  δ_V={d_V:.3f}")

print("\nDone.")
