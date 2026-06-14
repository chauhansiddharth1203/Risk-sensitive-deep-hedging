"""
main_vix_multiseed.py
----------------------
Sprint 3: multi-seed verification of Sprint 2b.

Runs Sprint 2b training with 5 different random seeds and aggregates
evaluation metrics (mean +/- std across seeds) on every real SPY window.
The critical question: is the 2024 point-estimate win and the COVID
closure seed-stable, or noise?

Decision rule (pre-registered before looking at results):
  - 2024 OOT: if mean Delta > 0 with seeds-std < |mean|, declare real win.
  - 2020 COVID: if mean Delta > -2.5 with seeds-std < 2.0, declare closure.
  - Otherwise: Sprint 2 hypothesis not supported; pivot to Neural SDE.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.vix_windows import load as load_spy_vix
from market.vix_bootstrap import VIXBootstrap
from policy.network_vix import VIXHedgingPolicy
from main_vix_futures_v2 import train as train_one, evaluate as evaluate_one

os.makedirs("results", exist_ok=True)


def one_seed(seed, epochs, N, lam, lr):
    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)
    print(f"\n[seed={seed}] training...")
    policy, hist = train_one(s_tr, s_te, epochs, N, lam, lr, seed)
    print(f"[seed={seed}] evaluating...")
    res = evaluate_one(policy)
    # keep only the numerical delta, to aggregate later
    deltas = {L: r["diff"][0] for L, r in res.items()}
    return deltas, res


def aggregate(all_deltas):
    """all_deltas: list[dict[label -> delta]]"""
    labels = list(all_deltas[0].keys())
    rows = []
    for L in labels:
        xs = np.array([d[L] for d in all_deltas])
        rows.append((L, xs.mean(), xs.std(), xs.min(), xs.max()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",  type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    args = ap.parse_args()

    all_deltas = []
    per_seed_raw = []
    for seed in range(args.seeds):
        d, raw = one_seed(seed, args.epochs, args.N, args.lam, args.lr)
        all_deltas.append(d)
        per_seed_raw.append({L: {"bs": r["bs"], "dh": r["dh"],
                                 "diff": r["diff"]}
                             for L, r in raw.items()})

    rows = aggregate(all_deltas)

    # -- Write metrics -- #
    with open("results/vix_multiseed_metrics.txt", "w") as f:
        f.write(f"Sprint 3 -- Multi-seed verification of Sprint 2b\n")
        f.write("=" * 72 + "\n")
        f.write(f"seeds={args.seeds}  epochs={args.epochs}  "
                f"N={args.N}  lr={args.lr}  lam={args.lam}\n\n")
        f.write(f"{'Window':<32}  {'mean Delta':>8}  {'std':>6}  "
                f"{'min':>7}  {'max':>7}\n")
        for L, mean, std, mn, mx in rows:
            f.write(f"{L:<32}  {mean:+8.3f}  {std:6.3f}  "
                    f"{mn:+7.3f}  {mx:+7.3f}\n")
        f.write("\n")
        # per-seed detail
        for i, d in enumerate(all_deltas):
            f.write(f"seed {i}:\n")
            for L, v in d.items():
                f.write(f"  {L:<32}  Delta={v:+7.3f}\n")
            f.write("\n")

    print("\n" + "=" * 72)
    print("Aggregated across seeds:")
    print(f"{'Window':<32}  {'mean Delta':>8}  {'std':>6}  "
          f"{'min':>7}  {'max':>7}")
    for L, mean, std, mn, mx in rows:
        print(f"{L:<32}  {mean:+8.3f}  {std:6.3f}  "
              f"{mn:+7.3f}  {mx:+7.3f}")

    # -- Plot mean Delta with std bars -- #
    labels = [r[0] for r in rows]
    means  = [r[1] for r in rows]
    stds   = [r[2] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5))
    colours = ["#4CAF50" if m > 0 else "#E57373" for m in means]
    ax.bar(x, means, yerr=stds, capsize=5, color=colours,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("CVaR$_{95}$ gap Delta = Deep - BS (mean +/- seed-std)")
    ax.set_title(f"Sprint 3: Multi-seed verification "
                 f"(n={args.seeds} seeds, {args.epochs} epochs each)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/vix_multiseed_delta.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Decision rule check
    print("\nDecision-rule check:")
    d_2024 = [r for r in rows if "2024" in r[0]][0]
    d_cov  = [r for r in rows if "COVID" in r[0]][0]
    win_2024 = d_2024[1] > 0 and d_2024[2] < abs(d_2024[1])
    closure_covid = d_cov[1] > -2.5 and d_cov[2] < 2.0
    print(f"  2024 mean Delta={d_2024[1]:+.3f}, std={d_2024[2]:.3f}  "
          f"-> real win? {win_2024}")
    print(f"  COVID mean Delta={d_cov[1]:+.3f}, std={d_cov[2]:.3f}  "
          f"-> closure? {closure_covid}")
    if win_2024 and closure_covid:
        print("\n>>> BOTH criteria met -> declare Sprint 2 hypothesis supported.")
    elif win_2024 or closure_covid:
        print("\n>>> Mixed: one criterion met. Report as weak evidence.")
    else:
        print("\n>>> Neither criterion met -> Sprint 2 hypothesis not supported. "
              "Pivot to Neural SDE.")


if __name__ == "__main__":
    main()
