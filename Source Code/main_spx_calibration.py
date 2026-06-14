"""
main_spx_calibration.py
------------------------
Week 4 Experiment C: Real SPX market calibration.

Downloads SPY options from Yahoo Finance, fits Heston parameters to the
observed implied vol surface, then trains and evaluates the deep hedging
strategy using those market-calibrated parameters.

If the download fails (no internet / stale data), falls back to published
SPX parameters (Gatheral & Jacquier 2011).

Pipeline:
  1. Download SPY options + calibrate Heston to IV surface
  2. Build a Heston simulator with calibrated parameters
  3. Train stock+VS CVaR policy on calibrated paths
  4. Evaluate: calibrated policy vs textbook policy vs BS delta hedge
  5. Plot IV surface fit + CVaR comparison

Produces:
  results/spx_iv_fit.png          -- market vs model implied vol smile
  results/spx_calib_comparison.png -- CVaR comparison bar chart
  results/spx_calib_varswap.pth   -- calibrated policy weights
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from calibration.calibrate_spx import get_calibrated_params, FALLBACK_SPX_PARAMS
from calibration.heston_pricer import heston_implied_vol, implied_vol, heston_call_price
from policy.network_varswap import HedgingPolicyVarSwap
from market.heston_with_var_swap import simulate_heston_with_var_swap
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

COST_RATE = 0.0002
ALPHA     = 0.95
N_TEST    = 5000
EPOCHS    = 300
T         = 30


def cvar_np(pnl, alpha=ALPHA):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ------------------------------------------------------------------ #
# 1. Calibrate to market                                             #
# ------------------------------------------------------------------ #
print("=" * 60)
print("Step 1: Calibrating Heston to SPY options ...")
print("=" * 60)
params = get_calibrated_params(target_days=30)

kappa_c   = params["kappa"]
theta_c   = params["theta"]
sigma_v_c = params["sigma_v"]
rho_c     = params["rho"]
v0_c      = params["v0"]
S0_c      = 100.0    # normalise to 100 for simulation
r_c       = 0.0

print(f"\n  Final parameters used for simulation:")
print(f"    kappa={kappa_c:.3f}, theta={theta_c:.4f}, "
      f"sigma_v={sigma_v_c:.3f}, rho={rho_c:.3f}, v0={v0_c:.4f}")


# ------------------------------------------------------------------ #
# 2. IV smile plot (if real data available)                          #
# ------------------------------------------------------------------ #
if params.get("rmse") is not None:
    print("\nStep 2: Plotting IV smile fit ...")
    # Re-download to plot
    from calibration.calibrate_spx import download_spy_options
    data = download_spy_options(target_days=30)

    if data is not None:
        S0_mkt, strikes, T_mkt, market_ivs, r_mkt, exp_str = data
        model_ivs = []
        for K in strikes:
            iv = heston_implied_vol(S0_mkt, K, T_mkt, r_mkt,
                                    v0_c, kappa_c, theta_c, sigma_v_c, rho_c)
            model_ivs.append(iv if iv else np.nan)

        moneyness = strikes / S0_mkt

        fig_iv, ax_iv = plt.subplots(figsize=(8, 4))
        ax_iv.plot(moneyness, np.array(market_ivs) * 100,
                   "o", color="#2196F3", markersize=7, label="Market IV (SPY)")
        ax_iv.plot(moneyness, np.array(model_ivs) * 100,
                   "-", color="#FF5722", linewidth=2, label=f"Heston fit (RMSE={params['rmse']*100:.2f}%)")
        ax_iv.set_xlabel("Moneyness  K/S₀", fontsize=10)
        ax_iv.set_ylabel("Implied Volatility (%)", fontsize=10)
        ax_iv.set_title(f"Heston IV Smile Fit -- SPY Options  (expiry: {exp_str})", fontsize=10)
        ax_iv.legend(fontsize=9)
        ax_iv.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/spx_iv_fit.png", dpi=150, bbox_inches="tight")
        print("  Saved: results/spx_iv_fit.png")
else:
    print("\nStep 2: Using fallback parameters -- no live IV data to plot.")


# ------------------------------------------------------------------ #
# 3. Build calibrated simulator (inject calibrated params)           #
# ------------------------------------------------------------------ #
import torch as _torch

def simulate_heston_calibrated_varswap(N, T=30, S0=100.0, K=100.0, device="cpu"):
    """
    Heston+VS simulator with market-calibrated parameters.
    VS scaling uses calibrated sigma_v for proper per-step balance.
    """
    dt = 1.0 / T

    rho_bar = _torch.sqrt(_torch.tensor(1.0 - rho_c ** 2, device=device))

    VS_scale = sigma_v_c    # use calibrated vol-of-vol for scaling

    S  = _torch.zeros(N, T + 1, device=device)
    v  = _torch.zeros(N, T + 1, device=device)
    VS = _torch.zeros(N, T + 1, device=device)

    S[:, 0]  = S0
    v[:, 0]  = v0_c
    VS[:, 0] = v0_c * S0 / 0.3    # keep same normalisation as trained policy

    for t in range(T):
        z1 = _torch.randn(N, device=device)
        z2 = rho_c * z1 + rho_bar * _torch.randn(N, device=device)

        v[:, t + 1] = _torch.clamp(
            v[:, t]
            + kappa_c * (theta_c - v[:, t]) * dt
            + sigma_v_c * _torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )

        S[:, t + 1] = S[:, t] * _torch.exp(
            (r_c - 0.5 * v[:, t]) * dt
            + _torch.sqrt(v[:, t] * dt) * z1
        )

        VS[:, t + 1] = v[:, t + 1] * S0 / 0.3

    def payoff_fn(S_T):
        return _torch.clamp(S_T - K, min=0.0)

    return S, VS, payoff_fn


# ------------------------------------------------------------------ #
# 4. Train stock+VS policy on calibrated paths                       #
# ------------------------------------------------------------------ #
print("\nStep 3: Training stock+VS CVaR policy on calibrated paths ...")
policy_calib = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
opt_calib    = torch.optim.Adam(policy_calib.parameters(), lr=3e-4)

from risk.cvar import cvar as cvar_fn

policy_calib.train()
alpha_start, alpha_end = 0.80, 0.95

for epoch in range(EPOCHS):
    alpha = alpha_start + (alpha_end - alpha_start) * epoch / max(EPOCHS - 1, 1)
    S_b, VS_b, pf_b = simulate_heston_calibrated_varswap(N=64, device=device)

    pnls = torch.stack([
        policy_calib.rollout(S_b[i], VS_b[i], pf_b)
        for i in range(64)
    ])
    loss = -cvar_fn(pnls, alpha)
    opt_calib.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_calib.parameters(), max_norm=1.0)
    opt_calib.step()

    if epoch % 30 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:3d}  alpha={alpha:.3f}  CVaR Loss {loss.item():.4f}")

torch.save(policy_calib.state_dict(), "results/spx_calib_varswap.pth")
policy_calib.eval()


# ------------------------------------------------------------------ #
# 5. Load textbook-trained policy for comparison                     #
# ------------------------------------------------------------------ #
print("\nStep 4: Loading textbook policy for comparison ...")
policy_textbook = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
policy_textbook.load_state_dict(
    torch.load("results/varswap_cvar.pth", map_location=device)
)
policy_textbook.eval()


# ------------------------------------------------------------------ #
# 6. Evaluate on calibrated test paths                               #
# ------------------------------------------------------------------ #
print("\nStep 5: Evaluating on calibrated test paths ...")
torch.manual_seed(0)
S_test, VS_test, payoff_test = simulate_heston_calibrated_varswap(
    N=N_TEST, device=device
)

pnl_delta = delta_hedge_pnl(S_test, payoff_test, K=100.0).cpu().numpy()

def rollout(pol, S, VS, payoff_fn):
    pnls = []
    with torch.no_grad():
        for i in range(S.shape[0]):
            pnls.append(pol.rollout(S[i], VS[i], payoff_fn).item())
    return np.array(pnls)

pnl_textbook = rollout(policy_textbook, S_test, VS_test, payoff_test)
pnl_calib    = rollout(policy_calib,    S_test, VS_test, payoff_test)

c_delta    = cvar_np(pnl_delta)
c_textbook = cvar_np(pnl_textbook)
c_calib    = cvar_np(pnl_calib)

print("\n======= Calibrated Model Results =======")
print(f"  Parameters: kappa={kappa_c:.2f}, theta={theta_c:.4f}, "
      f"sigma_v={sigma_v_c:.2f}, rho={rho_c:.2f}, v0={v0_c:.4f}")
print(f"\n  Delta Hedge (BS)                : {c_delta:>8.2f}")
print(f"  Textbook policy (zero-shot)     : {c_textbook:>8.2f}  (Delta {c_textbook-c_delta:>+.2f})")
print(f"  Calibrated policy               : {c_calib:>8.2f}  (Delta {c_calib-c_delta:>+.2f})")


# ------------------------------------------------------------------ #
# 7. Compare: Textbook Heston vs Calibrated (summary panel)         #
# ------------------------------------------------------------------ #
# Load textbook results for comparison
textbook_delta  = -15.92   # from Week 1 (corrected delta hedge)
textbook_vs     = -12.38   # from Week 1 bootstrap table

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Calibrated model bar chart
ax = axes[0]
labels  = ["Delta Hedge\n(BS)", "Textbook policy\n(zero-shot)", "Calibrated\npolicy"]
vals    = [c_delta, c_textbook, c_calib]
colours = ["#607D8B", "#FF9800", "#FF5722"]
bars = ax.bar(labels, vals, color=colours, edgecolor="black", linewidth=0.6, width=0.5)
ax.axhline(c_delta, color="#607D8B", linewidth=1.2, linestyle="--", alpha=0.7)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val - 0.5,
            f"{val:.2f}", ha="center", va="top", fontsize=10,
            fontweight="bold", color="white")
ax.set_ylabel("CVaR at 95%  (higher = better)", fontsize=9)
ax.set_title(f"Calibrated Heston Parameters\n"
             f"κ={kappa_c:.1f}, theta={theta_c:.3f}, σ_v={sigma_v_c:.2f}, ρ={rho_c:.2f}",
             fontsize=10)
ax.set_ylim(min(vals) - 3, max(vals) + 3)
ax.grid(axis="y", alpha=0.3)

# Right: Textbook vs Calibrated improvement comparison
ax2 = axes[1]
models     = ["Textbook Heston\n(Week 1)", "Calibrated Heston\n(Week 4)"]
imp_stock  = [c_textbook - c_delta,    c_textbook - c_delta]   # zero-shot in both
imp_vs     = [textbook_vs - textbook_delta, c_calib - c_delta]

x = np.array([0.0, 1.0])
w = 0.28
ax2.bar(x - w/2, [c_textbook - c_delta, c_textbook - c_delta],
        width=w, color="#FF9800", edgecolor="black", linewidth=0.5,
        label="Zero-shot transfer")
ax2.bar(x + w/2, [textbook_vs - textbook_delta, c_calib - c_delta],
        width=w, color="#FF5722", edgecolor="black", linewidth=0.5,
        label="Trained on same params")
ax2.axhline(0, color="black", linewidth=1.0, linestyle="--")
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylabel("CVaR improvement vs Delta Hedge", fontsize=9)
ax2.set_title("Calibration Matters:\nZero-Shot vs Trained-on-Same-Params", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/spx_calib_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/spx_calib_comparison.png")

plt.show()
print("\nDone.")
