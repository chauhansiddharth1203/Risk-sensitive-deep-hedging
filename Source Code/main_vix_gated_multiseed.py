"""
main_vix_gated_multiseed.py
----------------------------
Multi-seed verification of the regime-gated VIX policy (Option A).

Modes mirror main_vix_mom_multiseed.py so the comparison is direct:
  --mode calm   : sanity check on 2017 + 2024. Must preserve calm wins.
  --mode covid  : closure test on COVID, pre-registered decision rule
                  unchanged from plain-Adam baseline (mean > -2.5 AND
                  std < 2.0).

Evaluation pipeline is identical to main_vix_futures_v2.evaluate except
the rollout uses the gated rollout function -- handled by overriding the
deep-PnL function in main_vix_futures_v2.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.vix_windows import load as load_spy_vix
from market.vix_bootstrap import VIXBootstrap, payoff_call_atm
from training.trainer_gated import train_gated, rollout_batch_gated
from main_vix_futures_v2 import evaluate as evaluate_v2
import main_vix_futures_v2 as v2_mod

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_gated(policy):
    """Re-uses main_vix_futures_v2.evaluate() but with the gated rollout.

    We monkey-patch rollout_batch_vix inside the v2 module for the
    duration of the eval call so that evaluate() picks up the gated
    rollout (since it calls deep_pnl_vix -> rollout_batch_vix internally).
    """
    original_rollout = v2_mod.rollout_batch_vix
    v2_mod.rollout_batch_vix = rollout_batch_gated
    try:
        res = evaluate_v2(policy)
    finally:
        v2_mod.rollout_batch_vix = original_rollout
    return res


def one_seed(seed, epochs, N, lam, lr, thr_init, scale_init, freeze_gate):
    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)
    print(f"\n[seed={seed}  gate thr_init={thr_init} sc_init={scale_init} "
          f"frozen={freeze_gate}] training...")
    policy, hist = train_gated(
        s_tr, s_te, epochs, N, lam, lr, seed,
        gate_threshold_init=thr_init,
        gate_scale_init=scale_init,
        freeze_gate=freeze_gate,
    )
    final_thr = float(policy.gate_threshold.detach().cpu())
    final_sc  = float(policy.gate_scale.detach().cpu())
    print(f"[seed={seed}] evaluating... "
          f"(final gate: thr={final_thr:.3f}, sc={final_sc:.3f})")
    res = evaluate_gated(policy)
    deltas = {L: r["diff"][0] for L, r in res.items()}
    return deltas, res, (final_thr, final_sc)


def aggregate(all_deltas):
    labels = list(all_deltas[0].keys())
    rows = []
    for L in labels:
        xs = np.array([d[L] for d in all_deltas])
        rows.append((L, xs.mean(), xs.std(), xs.min(), xs.max()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calm", "covid"], default="calm")
    ap.add_argument("--seeds",  type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--thr",    type=float, default=1.5,
                    help="Gate threshold init (normalised VS units)")
    ap.add_argument("--scale",  type=float, default=0.2,
                    help="Gate width init")
    ap.add_argument("--freeze-gate", action="store_true",
                    help="Freeze gate at init values (ablation)")
    args = ap.parse_args()

    all_deltas, per_seed_raw, gate_finals = [], [], []
    for seed in range(args.seeds):
        d, raw, gate_final = one_seed(
            seed, args.epochs, args.N, args.lam, args.lr,
            args.thr, args.scale, args.freeze_gate)
        all_deltas.append(d)
        gate_finals.append(gate_final)
        per_seed_raw.append({L: {"bs": r["bs"], "dh": r["dh"],
                                 "diff": r["diff"]}
                             for L, r in raw.items()})

    rows = aggregate(all_deltas)

    tag = f"vix_gated_{args.mode}_n{args.seeds}"
    if args.freeze_gate:
        tag += "_frozen"
    out_path = f"results/{tag}_metrics.txt"
    with open(out_path, "w") as f:
        f.write(f"Regime-gated multi-seed -- mode={args.mode}\n")
        f.write("=" * 72 + "\n")
        f.write(f"seeds={args.seeds}  epochs={args.epochs}  "
                f"N={args.N}  lr={args.lr}  lam={args.lam}\n")
        f.write(f"gate init: thr={args.thr}  scale={args.scale}  "
                f"frozen={args.freeze_gate}\n\n")
        f.write(f"{'Window':<32}  {'mean Delta':>8}  {'std':>6}  "
                f"{'min':>7}  {'max':>7}\n")
        for L, mean, std, mn, mx in rows:
            f.write(f"{L:<32}  {mean:+8.3f}  {std:6.3f}  "
                    f"{mn:+7.3f}  {mx:+7.3f}\n")
        f.write("\nFinal gate parameters per seed:\n")
        for i, (thr, sc) in enumerate(gate_finals):
            f.write(f"  seed {i}: threshold={thr:+.4f}  scale={sc:+.4f}\n")
        f.write("\n")
        for i, d in enumerate(all_deltas):
            f.write(f"seed {i}:\n")
            for L, v in d.items():
                f.write(f"  {L:<32}  Delta={v:+7.3f}\n")
            f.write("\n")
    print(f"\nWrote {out_path}")

    print("\n" + "=" * 72)
    print(f"Aggregated across {args.seeds} seeds:")
    print(f"{'Window':<32}  {'mean Delta':>8}  {'std':>6}  "
          f"{'min':>7}  {'max':>7}")
    for L, mean, std, mn, mx in rows:
        print(f"{L:<32}  {mean:+8.3f}  {std:6.3f}  "
              f"{mn:+7.3f}  {mx:+7.3f}")
    print("\nLearned gate parameters per seed:")
    for i, (thr, sc) in enumerate(gate_finals):
        print(f"  seed {i}: threshold={thr:+.4f}  scale={sc:+.4f}")

    by_label = {r[0]: r for r in rows}
    def find_one(needle):
        for L, row in by_label.items():
            if needle in L:
                return row
        return None

    print("\nDecision-rule check:")
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
                print("\n>>> SANITY PASS: gate preserves calm wins.")
            else:
                print("\n>>> SANITY FAIL: gate degraded a window we know works.")
    else:
        rcov = find_one("COVID")
        if rcov:
            ok_mean = rcov[1] > -2.5
            ok_std = rcov[2] < 2.0
            print(f"  COVID:  mean Delta={rcov[1]:+.3f}  std={rcov[2]:.3f}  "
                  f"(plain-Adam: -8.05 +/- 4.96)")
            print(f"    -> closure: mean>-2.5? {ok_mean}  std<2.0? {ok_std}")
            if ok_mean and ok_std:
                print("\n>>> CLOSURE CONFIRMED: regime gate fixes COVID.")
            elif ok_mean or ok_std:
                print("\n>>> PARTIAL: one criterion met.")
            else:
                print("\n>>> REJECTED: gate did not close COVID. "
                      "Diagnosis incomplete.")

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
    ax.set_title(f"Regime-gated multi-seed "
                 f"(mode={args.mode}, n={args.seeds} seeds)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{tag}_delta.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
