"""
main_multi_asset.py
--------------------
Week 6: Multi-asset spread-option hedging under correlated Heston dynamics.

Contribution (novel vs literature):
  Deep hedging with CVaR on a multi-asset product, with BOTH stochastic-vol
  variance-swap instruments per asset. No published deep-hedging paper we
  reviewed (Buehler '19, Horvath '21, Carbonneau-Godin '21, Imaki '23,
  Marzban '22) addresses multi-asset + stochastic vol + CVaR + VS jointly.

Target:  max(S^1_T - S^2_T - K, 0)        (spread call, ATM: K=0)
Instruments (4):   S^1, S^2, VS^1, VS^2

Baseline: Margrabe-Kirk BS approximation for spread option delta.
  For K=0 this reduces to the exact Margrabe (1978) exchange-option
  formula -- we use its two deltas (dΠ/dS^1, dΠ/dS^2) with a blended
  volatility estimate.

State (9-dim):
  [S1/S01, S2/S02, VS1/VS10, VS2/VS20, t/T,
   prev_d1, prev_d2, prev_dV1, prev_dV2]

Outputs:
  results/multi_asset_learning_curve.png
  results/multi_asset_pnl.png
  results/multi_asset_cvar_by_corr.png
  results/multi_asset_hedge_composition.png
  results/multi_asset_policy.pth
  results/multi_asset_metrics.txt
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import norm

from market.multi_heston import (
    simulate_multi_heston, S0_1, S0_2, SIGMA_V_FIXED,
)
from risk.cvar import cvar as cvar_torch

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# -- hyper-params ------------------------------------------------------ #
COST_RATE  = 0.0002
ALPHA_LO   = 0.80
ALPHA_HI   = 0.95
S01, S02   = S0_1, S0_2
VS10       = 0.04 * S01 / SIGMA_V_FIXED        # ≈ 13.33
VS20       = 0.04 * S02 / SIGMA_V_FIXED
K_STRIKE   = 0.0                                # ATM spread


# ---------------------------------------------------------------------- #
# Policy: 9-dim state -> 4 hedge positions                              #
# ---------------------------------------------------------------------- #
class SpreadPolicy(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        raw = self.net(state)
        # Spot deltas range [-5, 5]; VS deltas range [-5, 5]
        return torch.tanh(raw) * 5.0


# ---------------------------------------------------------------------- #
# Vectorised rollout (across batch)                                     #
# ---------------------------------------------------------------------- #
def rollout_policy(policy, S1, S2, VS1, VS2, payoff_fn, cost_rate=COST_RATE):
    """
    Fully-vectorised rollout over batch.
    Returns P&L (N,) and optionally position tensors for diagnostics.
    """
    N, Tp1 = S1.shape
    T = Tp1 - 1

    pnl = torch.zeros(N, device=S1.device)
    prev = torch.zeros(N, 4, device=S1.device)     # [d1, d2, dV1, dV2]

    pos_log = torch.zeros(N, T, 4, device=S1.device)

    for t in range(T):
        state = torch.stack([
            S1[:, t] / S01,
            S2[:, t] / S02,
            VS1[:, t] / VS10,
            VS2[:, t] / VS20,
            torch.full((N,), t / T, device=S1.device),
            prev[:, 0], prev[:, 1], prev[:, 2], prev[:, 3],
        ], dim=1)
        act = policy(state)                     # (N, 4)
        pos_log[:, t, :] = act

        # P&L from positions held over [t, t+1]
        pnl = pnl + prev[:, 0] * (S1[:, t + 1]  - S1[:, t])
        pnl = pnl + prev[:, 1] * (S2[:, t + 1]  - S2[:, t])
        pnl = pnl + prev[:, 2] * (VS1[:, t + 1] - VS1[:, t])
        pnl = pnl + prev[:, 3] * (VS2[:, t + 1] - VS2[:, t])

        # Transaction costs on position *change*, scaled by instrument notional
        pnl = pnl - (
              cost_rate * torch.abs(act[:, 0] - prev[:, 0]) * (S1[:, t] / S01)
            + cost_rate * torch.abs(act[:, 1] - prev[:, 1]) * (S2[:, t] / S02)
            + cost_rate * torch.abs(act[:, 2] - prev[:, 2]) * (VS1[:, t] / VS10)
            + cost_rate * torch.abs(act[:, 3] - prev[:, 3]) * (VS2[:, t] / VS20)
        )
        prev = act

    payoff = payoff_fn(S1[:, -1], S2[:, -1])
    pnl = pnl - payoff
    return pnl, pos_log


# ---------------------------------------------------------------------- #
# Margrabe baseline: exchange option delta (K=0 spread)                 #
# ---------------------------------------------------------------------- #
def margrabe_deltas(S1, S2, v1, v2, rho_12, tau):
    """
    Margrabe (1978) exchange-option deltas for payoff max(S1 - S2, 0).
    Uses instantaneous variances v1, v2 as implied vol proxies.

    sigma_eff = sqrt(v1 + v2 - 2 rho_12 sqrt(v1 v2))
    d1 = (log(S1/S2) + 0.5 sigma_eff^2 tau) / (sigma_eff sqrt(tau))
    Delta_1 =  N(d1)
    Delta_2 = -N(d1 - sigma_eff sqrt(tau))
    """
    tau = max(tau, 1e-6)
    sig2 = v1 + v2 - 2.0 * rho_12 * np.sqrt(np.maximum(v1 * v2, 1e-12))
    sig2 = np.maximum(sig2, 1e-8)
    sig  = np.sqrt(sig2)
    d1 = (np.log(np.maximum(S1, 1e-8) / np.maximum(S2, 1e-8))
          + 0.5 * sig2 * tau) / (sig * np.sqrt(tau))
    d2 = d1 - sig * np.sqrt(tau)
    D1 =  norm.cdf(d1)
    D2 = -norm.cdf(d2)
    return D1, D2


@torch.no_grad()
def margrabe_pnl(S1, S2, v1, v2, rho_12, payoff_fn, cost_rate=COST_RATE):
    N, Tp1 = S1.shape
    T = Tp1 - 1
    S1n, S2n = S1.cpu().numpy(), S2.cpu().numpy()
    v1n, v2n = v1.cpu().numpy(), v2.cpu().numpy()

    pnl = np.zeros(N)
    prev_d1 = np.zeros(N)
    prev_d2 = np.zeros(N)

    for t in range(T):
        tau = (T - t) / T
        d1, d2 = margrabe_deltas(S1n[:, t], S2n[:, t],
                                 v1n[:, t], v2n[:, t], rho_12, tau)
        pnl += prev_d1 * (S1n[:, t + 1] - S1n[:, t])
        pnl += prev_d2 * (S2n[:, t + 1] - S2n[:, t])
        pnl -= cost_rate * np.abs(d1 - prev_d1) * (S1n[:, t] / S01)
        pnl -= cost_rate * np.abs(d2 - prev_d2) * (S2n[:, t] / S02)
        prev_d1, prev_d2 = d1, d2

    payoff = payoff_fn(S1[:, -1], S2[:, -1]).cpu().numpy()
    pnl -= payoff
    return pnl


# ---------------------------------------------------------------------- #
# Training                                                              #
# ---------------------------------------------------------------------- #
def train(policy, epochs, batch_size, rho_12):
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    curve = []
    for epoch in range(epochs):
        a = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * epoch / max(epochs - 1, 1)
        S1, S2, VS1, VS2, v1, v2, payoff_fn, _ = simulate_multi_heston(
            N=batch_size, rho_12=rho_12, device=device, K=K_STRIKE)
        pnl, _ = rollout_policy(policy, S1, S2, VS1, VS2, payoff_fn)
        loss = -cvar_torch(pnl, a)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()
        curve.append(-loss.item())

        if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:4d}/{epochs}  "
                  f"alpha={a:.3f}  CVaR={-loss.item():+.3f}")
    return np.array(curve)


# ---------------------------------------------------------------------- #
# CVaR helper                                                           #
# ---------------------------------------------------------------------- #
def cvar_np(pnl, alpha=0.95):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ---------------------------------------------------------------------- #
# Main                                                                  #
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch",  type=int, default=256)
    parser.add_argument("--n_test", type=int, default=4000)
    parser.add_argument("--smoke",  action="store_true",
                        help="short run for sanity check (50 epochs, 1000 test)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.batch, args.n_test = 50, 128, 1000

    rho_train = 0.5     # main training correlation

    print(f"Device: {device}")
    print(f"Training spread-call hedger: epochs={args.epochs}, "
          f"batch={args.batch}, rho_12={rho_train}")

    # ------- train ------------------------------------------------ #
    policy = SpreadPolicy().to(device)
    curve = train(policy, args.epochs, args.batch, rho_train)
    policy.eval()
    torch.save(policy.state_dict(), "results/multi_asset_policy.pth")

    # ------- evaluate at rho_train -------------------------------- #
    torch.manual_seed(0)
    S1, S2, VS1, VS2, v1, v2, payoff_fn, _ = simulate_multi_heston(
        N=args.n_test, rho_12=rho_train, device=device, K=K_STRIKE)

    with torch.no_grad():
        pnl_dh, pos_log = rollout_policy(policy, S1, S2, VS1, VS2, payoff_fn)
    pnl_dh = pnl_dh.cpu().numpy()
    pnl_mg = margrabe_pnl(S1, S2, v1, v2, rho_train, payoff_fn)

    cv_dh = cvar_np(pnl_dh)
    cv_mg = cvar_np(pnl_mg)
    print(f"\n=== Spread-call hedging at rho_12={rho_train} ===")
    print(f"  Margrabe baseline  : mean={pnl_mg.mean():+.3f}  "
          f"std={pnl_mg.std():.3f}  CVaR95={cv_mg:+.3f}")
    print(f"  Deep hedger (S+VS) : mean={pnl_dh.mean():+.3f}  "
          f"std={pnl_dh.std():.3f}  CVaR95={cv_dh:+.3f}")
    print(f"  Delta CVaR (higher=better): {cv_dh - cv_mg:+.3f}")

    # ------- CVaR sweep across rho_12 ----------------------------- #
    rhos = [-0.5, 0.0, 0.3, 0.5, 0.8]
    sweep = []
    print("\n=== Correlation sweep (policy trained at rho=0.5) ===")
    for r in rhos:
        S1_, S2_, VS1_, VS2_, v1_, v2_, pf_, _ = simulate_multi_heston(
            N=args.n_test, rho_12=r, device=device, K=K_STRIKE)
        with torch.no_grad():
            pnl_p, _ = rollout_policy(policy, S1_, S2_, VS1_, VS2_, pf_)
        pnl_p = pnl_p.cpu().numpy()
        pnl_b = margrabe_pnl(S1_, S2_, v1_, v2_, r, pf_)
        sweep.append((r, cvar_np(pnl_b), cvar_np(pnl_p)))
        print(f"  rho={r:+.2f}   Margrabe CVaR={sweep[-1][1]:+7.2f}   "
              f"DeepHedge CVaR={sweep[-1][2]:+7.2f}   "
              f"Delta={sweep[-1][2] - sweep[-1][1]:+.2f}")

    # ------- plots ------------------------------------------------ #
    # learning curve
    plt.figure(figsize=(7, 4))
    plt.plot(curve, color="#FF5722", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("CVaR (train)")
    plt.title(f"Spread-call CVaR training curve "
              f"(ρ₁₂={rho_train}, epochs={args.epochs})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/multi_asset_learning_curve.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # P&L histogram
    plt.figure(figsize=(8, 4.5))
    lo = min(pnl_dh.min(), pnl_mg.min()) - 1
    hi = max(pnl_dh.max(), pnl_mg.max()) + 1
    bins = np.linspace(lo, hi, 80)
    plt.hist(pnl_mg, bins=bins, alpha=0.45, color="#607D8B",
             label=f"Margrabe baseline  (CVaR={cv_mg:+.2f})")
    plt.hist(pnl_dh, bins=bins, alpha=0.55, color="#FF5722",
             label=f"Deep hedger (S+VS) (CVaR={cv_dh:+.2f})")
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Terminal P&L")
    plt.ylabel("Frequency")
    plt.title(f"Spread call max(S¹-S²-K,0):  P&L distribution\n"
              f"ρ₁₂={rho_train}, N={args.n_test}, cost={COST_RATE}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/multi_asset_pnl.png", dpi=150, bbox_inches="tight")
    plt.close()

    # CVaR by correlation
    rs = [s[0] for s in sweep]
    bs = [s[1] for s in sweep]
    ps = [s[2] for s in sweep]
    x  = np.arange(len(rs))
    w  = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - w / 2, bs, width=w, color="#607D8B", edgecolor="black",
            linewidth=0.5, label="Margrabe baseline")
    plt.bar(x + w / 2, ps, width=w, color="#FF5722", edgecolor="black",
            linewidth=0.5, label="Deep hedger (S+VS)")
    for i, (b, p) in enumerate(zip(bs, ps)):
        plt.text(i - w / 2, b - 0.5, f"{b:.1f}", ha="center", fontsize=8)
        plt.text(i + w / 2, p - 0.5, f"{p:.1f}", ha="center", fontsize=8)
    plt.xticks(x, [f"{r:+.1f}" for r in rs])
    plt.xlabel("Cross-asset correlation ρ₁₂")
    plt.ylabel("CVaR at 95% (higher = better)")
    plt.title("Spread-call hedging CVaR vs cross-asset correlation\n"
              "(policy trained at ρ₁₂=0.5)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/multi_asset_cvar_by_corr.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Hedge composition over time -- mean absolute positions
    pos_np = pos_log.cpu().numpy()    # (N, T, 4)
    mean_abs = np.mean(np.abs(pos_np), axis=0)   # (T, 4)
    T_steps = mean_abs.shape[0]
    ts = np.arange(T_steps) / T_steps
    plt.figure(figsize=(8, 4.5))
    plt.plot(ts, mean_abs[:, 0], label="|δ¹| (stock 1)",   color="#2196F3")
    plt.plot(ts, mean_abs[:, 1], label="|δ²| (stock 2)",   color="#4CAF50")
    plt.plot(ts, mean_abs[:, 2], label="|ν¹| (VS 1)",      color="#FF5722",
             linestyle="--")
    plt.plot(ts, mean_abs[:, 3], label="|ν²| (VS 2)",      color="#9C27B0",
             linestyle="--")
    plt.xlabel("Time t/T")
    plt.ylabel("Mean |position|")
    plt.title("Learned hedge composition across the 4 instruments")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/multi_asset_hedge_composition.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # metrics
    with open("results/multi_asset_metrics.txt", "w") as f:
        f.write("Week 6 Multi-Asset Spread Call -- Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Epochs: {args.epochs}, batch: {args.batch}, "
                f"N_test: {args.n_test}\n\n")
        f.write(f"Training rho_12 = {rho_train}\n")
        f.write(f"  Margrabe  : mean={pnl_mg.mean():+.3f}  "
                f"std={pnl_mg.std():.3f}  CVaR95={cv_mg:+.3f}\n")
        f.write(f"  Deep (S+VS): mean={pnl_dh.mean():+.3f}  "
                f"std={pnl_dh.std():.3f}  CVaR95={cv_dh:+.3f}\n")
        f.write(f"  Improvement: {cv_dh - cv_mg:+.3f}\n\n")
        f.write("Correlation sweep (policy trained at rho=0.5):\n")
        for r, b, p in sweep:
            f.write(f"  rho={r:+.2f}   Margrabe={b:+7.2f}   "
                    f"Deep={p:+7.2f}   delta={p - b:+.2f}\n")

    print("\nSaved:")
    print("  results/multi_asset_learning_curve.png")
    print("  results/multi_asset_pnl.png")
    print("  results/multi_asset_cvar_by_corr.png")
    print("  results/multi_asset_hedge_composition.png")
    print("  results/multi_asset_policy.pth")
    print("  results/multi_asset_metrics.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
