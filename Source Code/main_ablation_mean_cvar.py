"""
main_ablation_mean_cvar.py
---------------------------
Week 7 Ablation A: Mean-CVaR trade-off.

Motivation:
    Pure-CVaR loss L = -CVaR(Π) makes no attempt to keep Π's mean at zero.
    Week 6's multi-asset hedger had mean P&L = +51.8, so the policy is
    partly SPECULATING rather than purely HEDGING. This ablation adds a
    mean-penalty term and sweeps the weight lambda to trace the hedging /
    speculation Pareto frontier.

Loss:
    L(lambda) = -CVaR_alpha(Π) + lambda * |E[Π]|

    - lambda = 0     : pure CVaR (Week 6 baseline -- may speculate)
    - lambda -> ∞     : mean-zero hedger (pure hedging, higher CVaR cost)
    - intermediate lambda : trade-off curve

For each lambda in {0, 0.5, 1.0, 2.0, 5.0} we train a fresh spread-call hedger
on the two-asset Heston model (same setup as Week 6) and report mean,
std, CVaR_95, and CVaR_99.

Output:
    results/ablation_mean_cvar_frontier.png
    results/ablation_mean_cvar_table.txt
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from market.multi_heston import simulate_multi_heston
from main_multi_asset import (
    SpreadPolicy, rollout_policy, cvar_np, ALPHA_LO, ALPHA_HI,
)
from risk.cvar import cvar as cvar_torch

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


def train_with_lambda(lam, epochs, batch, rho_12=0.5):
    policy = SpreadPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    for epoch in range(epochs):
        a = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * epoch / max(epochs - 1, 1)
        S1, S2, VS1, VS2, v1, v2, pf, _ = simulate_multi_heston(
            N=batch, rho_12=rho_12, device=device, K=0.0)
        pnl, _ = rollout_policy(policy, S1, S2, VS1, VS2, pf)
        cvar_pen  = -cvar_torch(pnl, a)
        mean_pen  = torch.abs(pnl.mean())
        loss = cvar_pen + lam * mean_pen
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"    lambda={lam:.2f}  ep {epoch + 1:4d}  "
                  f"CVaR={-cvar_pen.item():+.2f}  |mean|={mean_pen.item():.2f}")
    return policy


def evaluate(policy, n_test=4000, rho_12=0.5):
    torch.manual_seed(0)
    S1, S2, VS1, VS2, v1, v2, pf, _ = simulate_multi_heston(
        N=n_test, rho_12=rho_12, device=device, K=0.0)
    policy.eval()
    with torch.no_grad():
        pnl, _ = rollout_policy(policy, S1, S2, VS1, VS2, pf)
    pnl = pnl.cpu().numpy()
    return dict(
        mean=float(pnl.mean()),
        std=float(pnl.std()),
        cvar95=cvar_np(pnl, 0.95),
        cvar99=cvar_np(pnl, 0.99),
        pnl=pnl,
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--n_test", type=int, default=4000)
    p.add_argument("--smoke",  action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.batch, args.n_test = 40, 128, 1000

    lambdas = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}

    for lam in lambdas:
        print(f"\n=== Training lambda = {lam} ===")
        policy = train_with_lambda(lam, args.epochs, args.batch)
        r = evaluate(policy, args.n_test)
        results[lam] = r
        print(f"  lambda={lam:.2f}  mean={r['mean']:+.3f}  std={r['std']:.3f}  "
              f"CVaR95={r['cvar95']:+.3f}  CVaR99={r['cvar99']:+.3f}")

    # ---- plot frontier ---- #
    means  = [abs(results[l]['mean'])  for l in lambdas]
    cv95   = [results[l]['cvar95']     for l in lambdas]
    stds   = [results[l]['std']        for l in lambdas]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(means, cv95, "o-", color="#FF5722", linewidth=2, markersize=8)
    for i, l in enumerate(lambdas):
        ax.annotate(f"lambda={l}", (means[i], cv95[i]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("|E[P&L]|   (lower = less speculation)")
    ax.set_ylabel("CVaR₉₅   (higher = better tail)")
    ax.set_title("Mean-CVaR Pareto frontier\n"
                 "Spread-call hedger, 2-asset Heston, ρ₁₂=0.5")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    ax = axes[1]
    x = np.arange(len(lambdas))
    w = 0.28
    ax.bar(x - w, [results[l]['mean']   for l in lambdas], width=w,
           color="#4CAF50", edgecolor="black", linewidth=0.5, label="mean P&L")
    ax.bar(x,     [results[l]['std']    for l in lambdas], width=w,
           color="#2196F3", edgecolor="black", linewidth=0.5, label="std P&L")
    ax.bar(x + w, [results[l]['cvar95'] for l in lambdas], width=w,
           color="#FF5722", edgecolor="black", linewidth=0.5, label="CVaR₉₅")
    ax.set_xticks(x)
    ax.set_xticklabels([f"lambda={l}" for l in lambdas])
    ax.set_ylabel("Value")
    ax.set_title("P&L statistics vs mean-penalty weight")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/ablation_mean_cvar_frontier.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ---- table ---- #
    with open("results/ablation_mean_cvar_table.txt", "w") as f:
        f.write("Week 7 -- Mean-CVaR trade-off ablation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Spread-call (2-asset Heston, rho_12=0.5), N_test={args.n_test}\n")
        f.write(f"Epochs per run: {args.epochs}, batch: {args.batch}\n\n")
        f.write(f"{'lambda':>8} {'mean':>10} {'std':>10} "
                f"{'CVaR95':>10} {'CVaR99':>10}\n")
        f.write("-" * 52 + "\n")
        for l in lambdas:
            r = results[l]
            f.write(f"{l:>8.2f} {r['mean']:>+10.3f} {r['std']:>10.3f} "
                    f"{r['cvar95']:>+10.3f} {r['cvar99']:>+10.3f}\n")

    print("\nSaved:")
    print("  results/ablation_mean_cvar_frontier.png")
    print("  results/ablation_mean_cvar_table.txt")


if __name__ == "__main__":
    main()
