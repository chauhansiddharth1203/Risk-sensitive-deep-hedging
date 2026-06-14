"""
main_vix_mom_multiseed.py
--------------------------
Multi-seed verification of the Median-of-Means optimiser on the VIX-as-
vega policy.

Two modes:
  --mode calm    : sanity check -- train as usual, but only report the
                   two calm windows (2017, 2024 OOT). The previous
                   plain-Adam multi-seed run gave +0.48 +/- 0.14 (2017)
                   and +1.16 +/- 0.06 (2024). MoM must not materially
                   degrade these. Pre-registered tolerance: 2024 mean
                   stays positive and within 0.3 of +1.16; std stays
                   below 0.30.
  --mode covid   : the actual test -- multi-seed on COVID with the
                   pre-registered decision rule from the plain-Adam run:
                   closure iff mean Delta > -2.5 AND std < 2.0.

Both modes also write the full per-window table for inspection.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.vix_windows import load as load_spy_vix
from market.vix_bootstrap import VIXBootstrap
from training.trainer_mom import train_mom
from main_vix_futures_v2 import evaluate

os.makedirs("results", exist_ok=True)


def one_seed(seed, epochs, N, lam, lr, k):
    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)
    print(f"\n[seed={seed}  MoM k={k}] training...")
    policy, hist = train_mom(s_tr, s_te, epochs, N, lam, lr, seed, k=k)
    print(f"[seed={seed}] evaluating...")
    res = evaluate(policy)
    deltas = {L: r["diff"][0] for L, r in res.items()}
    return deltas, res


def aggregate(all_deltas):
    labels = list(all_deltas[0].keys())
    rows = []
    for L in labels:
        xs = np.array([d[L] for d in all_deltas])
        rows.append((L, xs.mean(), xs.std(), xs.min(), xs.max()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calm", "covid"], default="calm",
                    help="calm = sanity check on 2017+2024; "
                         "covid = COVID closure test")
    ap.add_argument("--seeds",  type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--k",      type=int, default=9)
    args = ap.parse_args()

    all_deltas, per_seed_raw = [], []
    for seed in range(args.seeds):
        d, raw = one_seed(seed, args.epochs, args.N, args.lam,
                          args.lr, args.k)
        all_deltas.append(d)
        per_seed_raw.append({L: {"bs": r["bs"], "dh": r["dh"],
                                 "diff": r["diff"]}
                             for L, r in raw.items()})

    rows = aggregate(all_deltas)

    tag = f"vix_mom_{args.mode}_k{args.k}_n{args.seeds}"
    out_path = f"results/{tag}_metrics.txt"
    with open(out_path, "w") as f:
        f.write(f"MoM multi-seed -- mode={args.mode}\n")
        f.write("=" * 72 + "\n")
        f.write(f"seeds={args.seeds}  epochs={args.epochs}  "
                f"N={args.N}  lr={args.lr}  lam={args.lam}  k={args.k}\n\n")
        f.write(f"{'Window':<32}  {'mean Delta':>8}  {'std':>6}  "
                f"{'min':>7}  {'max':>7}\n")
        for L, mean, std, mn, mx in rows:
            f.write(f"{L:<32}  {mean:+8.3f}  {std:6.3f}  "
                    f"{mn:+7.3f}  {mx:+7.3f}\n")
        f.write("\n")
        for i, d in enumerate(all_deltas):
            f.write(f"seed {i}:\n")
            for L, v in d.items():
                f.write(f"  {L:<32}  Delta={v:+7.3f}\n")
            f.write("\n")
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 72)
    print(f"Aggregated across {args.seeds} seeds (MoM k={args.k}):")
    print(f"{'Window':<32}  {'mean Delta':>8}  {'std':>6}  "
          f"{'min':>7}  {'max':>7}")
    for L, mean, std, mn, mx in rows:
        print(f"{L:<32}  {mean:+8.3f}  {std:6.3f}  "
              f"{mn:+7.3f}  {mx:+7.3f}")

    # Pre-registered decision rules.
    print("\nDecision-rule check:")
    by_label = {r[0]: r for r in rows}

    def find_one(needle):
        for L, row in by_label.items():
            if needle in L:
                return row
        return None

    if args.mode == "calm":
        r2017 = find_one("2017")
        r2024 = find_one("2024")
        if r2017:
            print(f"  2017 calm:  mean Delta={r2017[1]:+.3f}  std={r2017[2]:.3f}  "
                  f"(plain-Adam: +0.48 +/- 0.14)")
        if r2024:
            ok_mean = r2024[1] > 0 and abs(r2024[1] - 1.16) < 0.30
            ok_std = r2024[2] < 0.30
            print(f"  2024 OOT:   mean Delta={r2024[1]:+.3f}  std={r2024[2]:.3f}  "
                  f"(plain-Adam: +1.16 +/- 0.06)")
            print(f"    -> calm sanity passes? "
                  f"mean-OK={ok_mean}  std-OK={ok_std}")
            if ok_mean and ok_std:
                print("\n>>> SANITY PASS: MoM preserves the calm-window wins. "
                      "Cleared to run COVID test.")
            else:
                print("\n>>> SANITY FAIL: MoM degraded a window we know works. "
                      "Debug before COVID.")
    else:  # covid
        rcov = find_one("COVID")
        if rcov:
            ok_mean = rcov[1] > -2.5
            ok_std = rcov[2] < 2.0
            print(f"  COVID:  mean Delta={rcov[1]:+.3f}  std={rcov[2]:.3f}  "
                  f"(plain-Adam: -8.05 +/- 4.96)")
            print(f"    -> closure criterion: mean>-2.5? {ok_mean}  "
                  f"std<2.0? {ok_std}")
            if ok_mean and ok_std:
                print("\n>>> CLOSURE CONFIRMED: heavy-tailed-gradient "
                      "diagnosis vindicated.")
            elif ok_mean or ok_std:
                print("\n>>> PARTIAL: one criterion met. Diagnosis partially "
                      "correct; gradient tails matter but aren't the whole "
                      "story.")
            else:
                print("\n>>> REJECTED: MoM did not close COVID. Diagnosis "
                      "incomplete -- escalate to Option A "
                      "(regime-gated architecture).")

    # Plot
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
    ax.set_title(f"MoM multi-seed verification "
                 f"(mode={args.mode}, k={args.k}, "
                 f"n={args.seeds} seeds)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{tag}_delta.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
