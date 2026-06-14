"""
main_sim_to_real.py
--------------------
Week 8 -- Sim-to-real sprint.

Goal: produce a deep hedger that is competitive with BS-delta on REAL
SPY crash windows (2008, 2020), where Week 7's vanilla-Heston-trained
policy failed.

Pipeline:
  1. Train a VS-CVaR policy on the combined Bates + SPX-calibrated +
     domain-randomized simulator (market/bates_spx_random.py).
  2. Evaluate on the same three historical windows used in Week 7:
     2008 GFC, 2020 COVID, 2017 Calm, with 95% bootstrap CIs.
  3. Compare against (a) BS-delta, (b) unhedged, (c) the Week-5 vanilla
     Heston-trained policy (`varswap_cvar_robust.pth`).

Outputs:
  results/varswap_cvar_spx_bates.pth
  results/sim_to_real_learning_curve.png
  results/sim_to_real_historical.png
  results/sim_to_real_metrics.txt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from market.bates_spx_random import simulate_bates_spx_random
from policy.network_varswap import HedgingPolicyVarSwap
from risk.cvar import cvar as cvar_torch
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci

# reuse Week-7 historical machinery
from main_historical_stress import (
    download_prices, build_windows, bs_pnl_per_window,
    unhedged_pnl, deep_hedger_pnl, cvar_np, COST_RATE, K,
)

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

ALPHA_LO, ALPHA_HI = 0.80, 0.95


# ---------------------------------------------------------------------- #
# Vectorised rollout using the HedgingPolicyVarSwap network              #
# ---------------------------------------------------------------------- #
def rollout_batch(policy, S, VS, payoff_fn):
    """
    Vectorised version of HedgingPolicyVarSwap.rollout across the batch.
    policy : HedgingPolicyVarSwap (5-dim state, 2 actions)
    S, VS  : (N, T+1) tensors
    """
    N, Tp1 = S.shape
    Tt = Tp1 - 1
    pnl = torch.zeros(N, device=S.device)
    prev_dS = torch.zeros(N, device=S.device)
    prev_dV = torch.zeros(N, device=S.device)

    S0_ = policy.S0
    VS0_ = policy.VS0
    c = policy.cost_rate

    for t in range(Tt):
        state = torch.stack([
            S[:, t]  / S0_,
            VS[:, t] / VS0_,
            torch.full((N,), t / Tt, device=S.device),
            prev_dS,
            prev_dV,
        ], dim=1)
        action  = policy.forward(state)          # (N, 2)
        dS = torch.tanh(action[:, 0]) * 5.0
        dV = torch.tanh(action[:, 1]) * 5.0

        pnl = pnl + prev_dS * (S[:, t + 1]  - S[:, t])
        pnl = pnl + prev_dV * (VS[:, t + 1] - VS[:, t])
        pnl = pnl - (c * torch.abs(dS - prev_dS) * (S[:, t]  / S0_)
                   + c * torch.abs(dV - prev_dV) * (VS[:, t] / VS0_))
        prev_dS, prev_dV = dS, dV

    pnl = pnl - payoff_fn(S[:, -1])
    return pnl


def train(policy, epochs, batch, lam=0.5):
    """Mean-penalised CVaR, lambda=0.5 chosen from Week 7 Pareto sweep."""
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    curve = []
    for epoch in range(epochs):
        a = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * epoch / max(epochs - 1, 1)
        S, VS, pf, _ = simulate_bates_spx_random(N=batch, device=device)
        pnl = rollout_batch(policy, S, VS, pf)
        loss = -cvar_torch(pnl, a) + lam * torch.abs(pnl.mean())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()
        curve.append(-(-cvar_torch(pnl, a).item()))  # training CVaR (=-loss+lambda|mean|)
        # record actual CVaR separately
        curve[-1] = cvar_torch(pnl, a).item()
        if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
            print(f"  ep {epoch + 1:4d}/{epochs}  alpha={a:.3f}  "
                  f"CVaR={curve[-1]:+.2f}  mean={pnl.mean().item():+.2f}")
    return np.array(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",  type=int, default=800)
    ap.add_argument("--batch",   type=int, default=256)
    ap.add_argument("--lam",     type=float, default=0.5,
                    help="mean-penalty weight")
    ap.add_argument("--smoke",   action="store_true")
    ap.add_argument("--tag",     type=str, default="spx_bates",
                    help="suffix for output file names")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.batch = 50, 128

    print(f"Device: {device}")
    print(f"Training Bates+SPX+DR hedger: epochs={args.epochs}, "
          f"batch={args.batch}, lambda={args.lam}")

    policy = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    curve = train(policy, args.epochs, args.batch, args.lam)
    policy.eval()
    tag = args.tag if hasattr(args, "tag") and args.tag else "spx_bates"
    torch.save(policy.state_dict(),
               f"results/varswap_cvar_{tag}.pth")

    # -- Learning-curve plot -- #
    plt.figure(figsize=(7, 4))
    plt.plot(curve, color="#FF5722", linewidth=1)
    plt.xlabel("Epoch"); plt.ylabel("CVaR (train)")
    plt.title(f"Sim-to-real training curve\n"
              f"Bates + SPX-calibrated + DR, lambda={args.lam}, {args.epochs} epochs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/sim_to_real_learning_curve.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # -- Historical evaluation -- same methodology as Week 7 -- #
    periods = {
        "2008 GFC":   ("2007-09-01", "2009-06-30"),
        "2020 COVID": ("2019-11-01", "2020-12-31"),
        "2017 Calm":  ("2016-07-01", "2017-12-31"),
    }

    # Week 7 policy for comparison (vanilla Heston + DR, no jumps, narrow range)
    old_path = "results/varswap_cvar_robust.pth"
    old = None
    if os.path.exists(old_path):
        old = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
        old.load_state_dict(torch.load(old_path, map_location=device))
        old.eval()

    all_res = {}
    for label, (start, end) in periods.items():
        print(f"\n=== {label}  ({start} -> {end}) ===")
        prices = download_prices(start, end)
        S_w, VS_w = build_windows(prices)
        print(f"  {len(S_w)} rolling 30-day windows (step=5)")

        pnl_bs   = bs_pnl_per_window(S_w)
        pnl_un   = unhedged_pnl(S_w)
        pnl_new  = deep_hedger_pnl(policy, S_w, VS_w)
        pnl_old  = deep_hedger_pnl(old,    S_w, VS_w) if old is not None else None

        def stats(x):
            c, lo, hi = bootstrap_cvar_ci(x, 0.95, B=500)
            return dict(mean=float(x.mean()), std=float(x.std()),
                        cvar=c, cvar_lo=lo, cvar_hi=hi)

        r = {
            "Unhedged":    stats(pnl_un),
            "BS":          stats(pnl_bs),
            "DeepHedge (new)": stats(pnl_new),
        }
        if pnl_old is not None:
            r["DeepHedge (Heston DR, W5)"] = stats(pnl_old)
        r["pnl"] = dict(bs=pnl_bs, un=pnl_un, new=pnl_new, old=pnl_old,
                        n_win=len(S_w))
        all_res[label] = r

        for k in ["Unhedged", "BS", "DeepHedge (Heston DR, W5)",
                  "DeepHedge (new)"]:
            if k not in r: continue
            s = r[k]
            print(f"  {k:<28} mean={s['mean']:+7.2f}  std={s['std']:5.2f}  "
                  f"CVaR95={s['cvar']:+7.2f}  "
                  f"[{s['cvar_lo']:+6.2f}, {s['cvar_hi']:+6.2f}]")

        # paired diff deep-new vs BS
        dd, lo, hi = bootstrap_diff_ci(pnl_new, pnl_bs, 0.95, B=500)
        print(f"  Delta(new-BS) CVaR = {dd:+.2f}  [{lo:+.2f}, {hi:+.2f}]")

    # -- Headline bar chart: new vs old vs BS -- #
    labels  = list(all_res.keys())
    methods = ["Unhedged", "BS", "DeepHedge (Heston DR, W5)", "DeepHedge (new)"]
    colours = {"Unhedged": "#9E9E9E", "BS": "#2196F3",
               "DeepHedge (Heston DR, W5)": "#FFC107",
               "DeepHedge (new)": "#FF5722"}

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(labels))
    w = 0.20
    for j, m in enumerate(methods):
        if m not in all_res[labels[0]]: continue
        cv  = [all_res[L][m]['cvar']    for L in labels]
        lo  = [all_res[L][m]['cvar_lo'] for L in labels]
        hi  = [all_res[L][m]['cvar_hi'] for L in labels]
        err = [[cv[i] - lo[i] for i in range(len(cv))],
               [hi[i] - cv[i] for i in range(len(cv))]]
        pos = x + (j - 1.5) * w
        ax.bar(pos, cv, width=w, color=colours[m],
               yerr=err, capsize=3, edgecolor="black", linewidth=0.5,
               label=m)
        for i, v in enumerate(cv):
            ax.text(pos[i], v - 0.5, f"{v:.1f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n(n={all_res[L]['pnl']['n_win']})" for L in labels])
    ax.set_ylabel("CVaR₉₅ (higher = better)")
    ax.set_title("Sim-to-real: Bates+SPX+DR policy vs. Week-5 Heston-DR policy vs. BS\n"
                 "All evaluated on REAL SPY 30-day windows")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/sim_to_real_historical.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # -- Metrics file -- #
    with open("results/sim_to_real_metrics.txt", "w") as f:
        f.write("Week 8 -- Sim-to-real training and evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Simulator: Bates + SPX-calibrated + domain randomisation\n")
        f.write(f"Loss: CVaR + {args.lam}*|mean|\n")
        f.write(f"Epochs: {args.epochs}, batch: {args.batch}\n\n")
        for L in labels:
            r = all_res[L]
            f.write(f"{L}  (n={r['pnl']['n_win']} windows)\n")
            for k in ["Unhedged", "BS", "DeepHedge (Heston DR, W5)",
                      "DeepHedge (new)"]:
                if k not in r: continue
                s = r[k]
                f.write(f"  {k:<28} mean={s['mean']:+7.3f}  "
                        f"std={s['std']:6.3f}  CVaR95={s['cvar']:+7.3f}  "
                        f"[{s['cvar_lo']:+6.3f}, {s['cvar_hi']:+6.3f}]\n")
            dd, lo, hi = bootstrap_diff_ci(
                r["pnl"]["new"], r["pnl"]["bs"], 0.95, B=500)
            f.write(f"  Delta(new-BS) CVaR = {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]\n\n")

    print("\nSaved:")
    print("  results/varswap_cvar_spx_bates.pth")
    print("  results/sim_to_real_learning_curve.png")
    print("  results/sim_to_real_historical.png")
    print("  results/sim_to_real_metrics.txt")


if __name__ == "__main__":
    main()
