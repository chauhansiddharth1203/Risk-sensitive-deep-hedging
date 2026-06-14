"""
main_sim_to_real_earlystop.py
------------------------------
Week 8 follow-up -- sim-to-real with proper early stopping.

Week 8 finding: naive long training on the Bates+SPX+DR simulator
overfits to simulator quirks and degrades real-data performance.
Fix: checkpoint every K epochs, evaluate each on a SIMULATED
held-out crisis validation set (Heston at σ_v=0.8, ρ=-0.85, v0=0.09),
pick the best checkpoint. No real-data leakage into training.

Selection metric: validation-set CVaR_95 minus |mean| penalty
(same form as training loss but lambda=1 and alpha=0.95 fixed).

Outputs:
  results/varswap_cvar_spx_bates_best.pth      -- best-by-validation checkpoint
  results/sim_to_real_es_history.png           -- train & val CVaR over epochs
  results/sim_to_real_es_historical.png        -- final historical bars
  results/sim_to_real_es_metrics.txt
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
from main_historical_stress import (
    download_prices, build_windows, bs_pnl_per_window,
    unhedged_pnl, deep_hedger_pnl, COST_RATE, K,
)
from main_sim_to_real import rollout_batch

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

ALPHA_LO, ALPHA_HI = 0.80, 0.95


# ---------------------------------------------------------------------- #
# Validation set: simulated crisis Heston (different from training)      #
# ---------------------------------------------------------------------- #
def make_validation_set(N=2000):
    """
    Crisis Heston: kappa=0.5, theta=0.12, sigma_v=0.8, rho=-0.85, v0=0.09.
    Intentionally at the edge of (and beyond) the training distribution.
    No jumps (so not a trivial replay of training).
    """
    torch.manual_seed(1234)
    from market.heston_with_var_swap import simulate_heston_with_var_swap
    # override parameters inline by monkey-patching not ideal; simpler
    # to write the paths directly here
    import math
    kappa, theta, sigma_v, rho, v0 = 0.5, 0.12, 0.8, -0.85, 0.09
    T = 30
    S0 = 100.0
    K_ = 100.0
    dt = 1.0 / T
    rho_bar = (1.0 - rho ** 2) ** 0.5
    S = torch.zeros(N, T + 1, device=device)
    v = torch.zeros(N, T + 1, device=device)
    VS = torch.zeros(N, T + 1, device=device)
    S[:, 0]  = S0;  v[:, 0]  = v0
    VS[:, 0] = v0 * S0 / 0.30
    for t in range(T):
        z1 = torch.randn(N, device=device)
        z2 = rho * z1 + rho_bar * torch.randn(N, device=device)
        v[:, t + 1] = torch.clamp(
            v[:, t] + kappa * (theta - v[:, t]) * dt
            + sigma_v * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6)
        S[:, t + 1] = S[:, t] * torch.exp(
            -0.5 * v[:, t] * dt + torch.sqrt(v[:, t] * dt) * z1)
        VS[:, t + 1] = v[:, t + 1] * S0 / 0.30
    def payoff_fn(S_T):
        return torch.clamp(S_T - K_, min=0.0)
    return S, VS, payoff_fn


def val_score(policy, S_val, VS_val, pf_val, lam=1.0, alpha=0.95):
    """Higher is better. Score = CVaR - lambda|mean| (NOT the negative loss)."""
    policy.eval()
    with torch.no_grad():
        pnl = rollout_batch(policy, S_val, VS_val, pf_val)
    c = cvar_torch(pnl, alpha).item()
    m = abs(pnl.mean().item())
    policy.train()
    return c - lam * m, c, pnl.mean().item()


def train_with_earlystop(policy, epochs, batch, lam, check_every=25):
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    S_val, VS_val, pf_val = make_validation_set(N=2000)

    best_score = -np.inf
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    history = {"epoch": [], "train_cvar": [], "val_cvar": [],
               "val_mean": [], "val_score": []}

    for epoch in range(1, epochs + 1):
        a = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * (epoch - 1) / max(epochs - 1, 1)
        S, VS, pf, _ = simulate_bates_spx_random(N=batch, device=device)
        pnl = rollout_batch(policy, S, VS, pf)
        loss = -cvar_torch(pnl, a) + lam * torch.abs(pnl.mean())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        if epoch % check_every == 0 or epoch == 1:
            score, vcvar, vmean = val_score(policy, S_val, VS_val, pf_val,
                                            lam=lam)
            history["epoch"].append(epoch)
            history["train_cvar"].append(cvar_torch(pnl, a).item())
            history["val_cvar"].append(vcvar)
            history["val_mean"].append(vmean)
            history["val_score"].append(score)
            tag = ""
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone()
                              for k, v in policy.state_dict().items()}
                tag = "  *** new best"
            print(f"  ep {epoch:4d}  alpha={a:.3f}  "
                  f"trainCVaR={history['train_cvar'][-1]:+7.2f}  "
                  f"valCVaR={vcvar:+7.2f}  valMean={vmean:+6.2f}  "
                  f"score={score:+7.2f}{tag}")

    policy.load_state_dict(best_state)
    print(f"\nBest checkpoint: epoch {best_epoch}, val score {best_score:+.2f}")
    return history, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--lam",    type=float, default=1.0)
    ap.add_argument("--check_every", type=int, default=25)
    args = ap.parse_args()

    print(f"Device: {device}")
    print(f"Bates+SPX+DR with early stopping: epochs={args.epochs}, "
          f"batch={args.batch}, lambda={args.lam}, check_every={args.check_every}")

    policy = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    history, best_ep = train_with_earlystop(
        policy, args.epochs, args.batch, args.lam, args.check_every)
    policy.eval()
    torch.save(policy.state_dict(),
               "results/varswap_cvar_spx_bates_best.pth")

    # -- train/val history plot -- #
    plt.figure(figsize=(9, 4.5))
    ep = history["epoch"]
    plt.plot(ep, history["train_cvar"], "o-", color="#2196F3",
             label="Train CVaR (per-batch)")
    plt.plot(ep, history["val_cvar"],   "o-", color="#FF5722",
             label="Validation CVaR (crisis Heston, N=2000)")
    plt.plot(ep, history["val_score"], "s--", color="#4CAF50",
             label=f"Val score = CVaR - {args.lam}|mean|", alpha=0.6)
    plt.axvline(best_ep, color="black", linewidth=0.8, linestyle=":",
                label=f"Best @ epoch {best_ep}")
    plt.xlabel("Epoch"); plt.ylabel("CVaR / score")
    plt.title("Early-stopping history: train vs. held-out crisis validation")
    plt.legend(fontsize=9); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/sim_to_real_es_history.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # -- Historical evaluation -- #
    periods = {
        "2008 GFC":   ("2007-09-01", "2009-06-30"),
        "2020 COVID": ("2019-11-01", "2020-12-31"),
        "2017 Calm":  ("2016-07-01", "2017-12-31"),
    }
    all_res = {}
    print("\n=== Historical evaluation (best checkpoint) ===")
    for label, (start, end) in periods.items():
        prices = download_prices(start, end)
        S_w, VS_w = build_windows(prices)
        pnl_bs  = bs_pnl_per_window(S_w)
        pnl_un  = unhedged_pnl(S_w)
        pnl_new = deep_hedger_pnl(policy, S_w, VS_w)

        def stats(x):
            c, lo, hi = bootstrap_cvar_ci(x, 0.95, B=500)
            return dict(mean=float(x.mean()), std=float(x.std()),
                        cvar=c, cvar_lo=lo, cvar_hi=hi)

        r = dict(Unhedged=stats(pnl_un), BS=stats(pnl_bs),
                 DeepHedge=stats(pnl_new),
                 pnl=dict(bs=pnl_bs, un=pnl_un, new=pnl_new, n_win=len(S_w)))
        all_res[label] = r
        dd, lo, hi = bootstrap_diff_ci(pnl_new, pnl_bs, 0.95, B=500)
        print(f"  {label}:  BS CVaR={r['BS']['cvar']:+.2f}   "
              f"Deep CVaR={r['DeepHedge']['cvar']:+.2f}   "
              f"Delta={dd:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    # -- Final bar chart -- #
    labels = list(all_res.keys())
    methods = ["Unhedged", "BS", "DeepHedge"]
    colours = {"Unhedged": "#9E9E9E", "BS": "#2196F3", "DeepHedge": "#FF5722"}
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, m in enumerate(methods):
        cv  = [all_res[L][m]['cvar']    for L in labels]
        lo  = [all_res[L][m]['cvar_lo'] for L in labels]
        hi  = [all_res[L][m]['cvar_hi'] for L in labels]
        err = [[cv[i] - lo[i] for i in range(len(cv))],
               [hi[i] - cv[i] for i in range(len(cv))]]
        pos = x + (j - 1) * w
        ax.bar(pos, cv, width=w, color=colours[m], yerr=err, capsize=4,
               edgecolor="black", linewidth=0.5, label=m)
        for i, v in enumerate(cv):
            ax.text(pos[i], v - 0.4, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n(n={all_res[L]['pnl']['n_win']})" for L in labels])
    ax.set_ylabel("CVaR₉₅")
    ax.set_title("Sim-to-real with early stopping (best-val checkpoint)\n"
                 "Bates + SPX-calibrated + DR, evaluated on REAL SPY")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/sim_to_real_es_historical.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    with open("results/sim_to_real_es_metrics.txt", "w") as f:
        f.write("Week 8 follow-up: early-stopping sim-to-real\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best checkpoint: epoch {best_ep}/{args.epochs}\n\n")
        for L in labels:
            r = all_res[L]
            f.write(f"{L}  (n={r['pnl']['n_win']} windows)\n")
            for k in ["Unhedged", "BS", "DeepHedge"]:
                s = r[k]
                f.write(f"  {k:<10} mean={s['mean']:+7.3f}  "
                        f"std={s['std']:6.3f}  CVaR95={s['cvar']:+7.3f}  "
                        f"[{s['cvar_lo']:+6.3f}, {s['cvar_hi']:+6.3f}]\n")
            dd, lo, hi = bootstrap_diff_ci(r["pnl"]["new"], r["pnl"]["bs"],
                                           0.95, B=500)
            f.write(f"  Delta(Deep-BS) CVaR = {dd:+.3f}  "
                    f"[{lo:+.3f}, {hi:+.3f}]\n\n")

    print("\nSaved:")
    print("  results/varswap_cvar_spx_bates_best.pth")
    print("  results/sim_to_real_es_history.png")
    print("  results/sim_to_real_es_historical.png")
    print("  results/sim_to_real_es_metrics.txt")


if __name__ == "__main__":
    main()
