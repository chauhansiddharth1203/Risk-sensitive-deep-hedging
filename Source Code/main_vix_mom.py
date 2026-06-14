"""
main_vix_mom.py
----------------
Single-seed driver for Median-of-Means gradient training on the VIX-as-
vega policy.

Drop-in replacement for main_vix_futures_v2.py with one change: the
gradient aggregator is MoM instead of plain mean. Everything else
(architecture, action scales, lr, batch size, evaluation pipeline) is
identical, so any difference in results isolates the optimiser change.
"""

import os
import argparse
import torch

from data.vix_windows import load as load_spy_vix
from market.vix_bootstrap import VIXBootstrap
from training.trainer_mom import train_mom
from main_vix_futures_v2 import evaluate
from main_vix_futures import write_metrics, plot_learning, plot_cvar

os.makedirs("results", exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--k",      type=int, default=9,
                    help="MoM block count (odd recommended)")
    ap.add_argument("--save",   default="results/vix_mom.pth")
    args = ap.parse_args()

    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)
    print(f"Train weekly obs: {len(tr_s)}, Test: {len(te_s)}")
    print(f"MoM gradient: k={args.k} blocks of ~{args.N // args.k} samples each")
    print(f"lr={args.lr}  N={args.N}  epochs={args.epochs}")

    policy, hist = train_mom(s_tr, s_te, args.epochs, args.N, args.lam,
                             args.lr, args.seed, k=args.k)
    torch.save(policy.state_dict(), args.save)
    print(f"Saved -> {args.save}")

    print("\n=== Evaluation ===")
    res = evaluate(policy)

    tag = f"vix_mom_k{args.k}_s{args.seed}"
    write_metrics(res, hist, f"results/{tag}_metrics.txt")
    plot_learning(hist, f"results/{tag}_learning.png")
    plot_cvar(res, f"results/{tag}_cvar.png")
    print(f"\nArtifacts in results/{tag}_*.{{txt,png}}")


if __name__ == "__main__":
    main()
