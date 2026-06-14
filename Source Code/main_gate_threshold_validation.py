"""
main_gate_threshold_validation.py
-----------------------------------
Step 1 of the post-Week-13 roadmap: principled gate threshold selection.

PROBLEM WITH WEEK 13
--------------------
The theta = 3.0 result was chosen by sweeping over ALL seven evaluation
windows including 2024. That means the final-test window (2024) was used
to pick the hyperparameter, which is test-set tuning.

FIX IMPLEMENTED HERE
---------------------
We split the OOT period into:
  - VALIDATION  : 2022 Rate shock + 2023 SVB/banking
                  (used to select theta; never influences 2024)
  - FINAL TEST  : 2024 full year   (held out -- revealed only at the end)
  - IN-TRAINING : 2008 GFC, 2017 Calm, 2020 COVID
                  (sanity checks; theta selection is not driven by these)

SELECTION CRITERION (pre-registered before running)
-----------------------------------------------------
  val_score(theta)  = mean(Delta_2022, Delta_2023)         <- higher is better
  covid_mean(theta) = mean seed Delta on 2020 COVID
  covid_std(theta)  = std  seed Delta on 2020 COVID

  theta* = argmax val_score(theta)
       subject to  covid_mean > -2.5
               AND covid_std  <  2.0

If no theta satisfies the COVID constraint, report the best unconstrained theta
and flag it explicitly.

SWEEP
-----
  theta in {2.0, 2.5, 3.0, 3.5}
  w  = 0.3  (frozen, same as Week 13)
  5 seeds per theta
  400 epochs per seed (same as Week 13)

FINAL REPORT
------------
Only after theta* is selected do we evaluate on 2024 and print the result.

Usage
-----
    python main_gate_threshold_validation.py [--epochs N] [--seeds N]
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from data.vix_windows import load as load_spy_vix
from market.vix_bootstrap import VIXBootstrap, payoff_call_atm
from training.trainer_gated import train_gated, rollout_batch_gated
from main_vix_futures import download_spy_vix, build_weekly_windows, bs_pnl_weekly
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci
import main_vix_futures_v2 as v2_mod

os.makedirs("results/gate_validation", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

import os as _os
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ------------------------------------------------------------------ #
# Window definitions                                                   #
# ------------------------------------------------------------------ #
VALIDATION_WINDOWS = {
    "2022 Rate shock":    ("2022-01-01", "2022-12-31"),
    "2023 SVB / banking": ("2023-01-01", "2023-12-31"),
}
FINAL_TEST_WINDOWS = {
    "2024 Full year":     ("2024-01-01", "2024-12-31"),
}
SANITY_WINDOWS = {
    "2008 GFC":           ("2007-09-01", "2009-06-30"),
    "2017 Calm":          ("2016-07-01", "2017-12-31"),
    "2020 COVID":         ("2019-11-01", "2020-12-31"),
}

# Pre-registered COVID constraint (same as Week 12)
COVID_MEAN_THRESH = -2.5
COVID_STD_THRESH  =  2.0


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def eval_window(policy, label, start, end):
    """Evaluate policy on one real-data window. Returns diff stats."""
    from market.vix_bootstrap import T
    df = download_spy_vix(start, end)
    if df is None or len(df) < T + 2:
        return None
    Sw, VSw = build_weekly_windows(df)
    if len(Sw) == 0:
        return None

    p_bs = bs_pnl_weekly(Sw)

    S_t  = torch.tensor(Sw,  dtype=torch.float32, device=device)
    VS_t = torch.tensor(VSw, dtype=torch.float32, device=device)
    with torch.no_grad():
        p_dh_t = rollout_batch_gated(policy, S_t, VS_t, payoff_call_atm)
    p_dh = p_dh_t.cpu().numpy()

    c_bs, lo_bs, hi_bs = bootstrap_cvar_ci(p_bs, 0.95, B=500)
    c_dh, lo_dh, hi_dh = bootstrap_cvar_ci(p_dh, 0.95, B=500)
    d, dlo, dhi         = bootstrap_diff_ci(p_dh, p_bs, 0.95, B=500)
    return dict(bs=(c_bs, lo_bs, hi_bs), dh=(c_dh, lo_dh, hi_dh),
                diff=(d, dlo, dhi), n=len(Sw))


def run_one_seed(seed, epochs, N, lam, lr, theta, width,
                 s_tr, s_te, eval_windows):
    """Train one seed at fixed gate threshold and evaluate."""
    policy, _ = train_gated(
        s_tr, s_te, epochs, N, lam, lr, seed,
        gate_threshold_init=theta,
        gate_scale_init=width,
        freeze_gate=True,          # freeze gate outside optimisation
    )
    results = {}
    for label, (start, end) in eval_windows.items():
        r = eval_window(policy, label, start, end)
        if r is not None:
            results[label] = r["diff"][0]   # point-estimate Delta
        else:
            results[label] = float("nan")
    return results


def aggregate_seeds(all_seed_results):
    """
    all_seed_results: list of dicts  {label: delta_float}
    Returns: {label: (mean, std)}
    """
    labels = list(all_seed_results[0].keys())
    out = {}
    for L in labels:
        xs = np.array([d[L] for d in all_seed_results
                       if not np.isnan(d.get(L, float("nan")))])
        out[L] = (float(xs.mean()), float(xs.std()))
    return out


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(
        description="Principled gate threshold selection via 2022-23 validation")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds",  type=int, default=5)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--width",  type=float, default=0.3,
                    help="Gate sigmoid width (frozen)")
    args = ap.parse_args()

    THRESHOLDS = [2.0, 2.5, 3.0, 3.5]

    # Load data once
    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)

    # Windows used for selection (validation + COVID sanity)
    selection_windows = {**VALIDATION_WINDOWS,
                         "2020 COVID": SANITY_WINDOWS["2020 COVID"]}

    print("=" * 72)
    print("GATE THRESHOLD VALIDATION  (2024 held out)")
    print(f"Sweep theta in {THRESHOLDS}   w={args.width}  "
          f"{args.seeds} seeds x {args.epochs} epochs")
    print("Selection criterion: argmax mean(Delta_2022, Delta_2023) "
          "s.t. COVID mean > -2.5 AND std < 2.0")
    print("=" * 72)

    sweep_results = {}   # theta -> {label: (mean, std)}

    for theta in THRESHOLDS:
        print(f"\n{'-'*60}")
        print(f"theta = {theta}")
        all_seed_diffs = []
        for seed in range(args.seeds):
            print(f"  seed {seed}...")
            seed_diffs = run_one_seed(
                seed, args.epochs, args.N, args.lam, args.lr,
                theta, args.width, s_tr, s_te, selection_windows)
            all_seed_diffs.append(seed_diffs)
            for L, v in seed_diffs.items():
                print(f"    {L:<32} Delta={v:+.3f}")

        agg = aggregate_seeds(all_seed_diffs)
        sweep_results[theta] = agg
        print(f"\n  Aggregated (theta={theta}):")
        for L, (mean, std) in agg.items():
            print(f"    {L:<32} mean={mean:+.3f}  std={std:.3f}")

    # ---------------------------------------------------------------- #
    # Threshold selection                                               #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("THRESHOLD SELECTION")
    print("=" * 72)

    best_theta = None
    best_score = -np.inf
    selection_log = []

    for theta in THRESHOLDS:
        agg = sweep_results[theta]

        # Validation score: mean of 2022 + 2023 deltas
        val_scores = []
        for L in VALIDATION_WINDOWS:
            if L in agg:
                val_scores.append(agg[L][0])
        val_score = float(np.mean(val_scores)) if val_scores else -np.inf

        # COVID constraint
        covid_mean, covid_std = agg.get("2020 COVID", (-999.0, 999.0))
        covid_ok = (covid_mean > COVID_MEAN_THRESH and
                    covid_std  < COVID_STD_THRESH)

        selection_log.append(
            (theta, val_score, covid_mean, covid_std, covid_ok))

        flag = "[OK] feasible" if covid_ok else "[FAIL] violates COVID constraint"
        print(f"  theta={theta:.1f}  val_score={val_score:+.3f}  "
              f"COVID mean={covid_mean:+.3f} std={covid_std:.3f}  {flag}")

        if covid_ok and val_score > best_score:
            best_score = val_score
            best_theta = theta

    # Fallback if no theta satisfies constraint
    if best_theta is None:
        print("\n  WARNING No theta satisfies COVID constraint -- "
              "selecting by val_score only (flag this in writeup)")
        best_theta = max(THRESHOLDS,
                         key=lambda t: sweep_results[t].get(
                             list(VALIDATION_WINDOWS.keys())[0], (-99,))[0])

    print(f"\n  ***  Selected theta* = {best_theta}  "
          f"(val_score={best_score:+.3f})\n")

    # ---------------------------------------------------------------- #
    # Final evaluation on held-out 2024                               #
    # ---------------------------------------------------------------- #
    print("=" * 72)
    print(f"FINAL TEST ON HELD-OUT 2024  (theta* = {best_theta})")
    print("=" * 72)

    all_2024_diffs = []
    for seed in range(args.seeds):
        print(f"  seed {seed} -- re-training at theta*={best_theta}...")
        tr_s2, tr_v2, te_s2, te_v2 = load_spy_vix()
        s_tr2 = VIXBootstrap(tr_s2, tr_v2)
        s_te2 = VIXBootstrap(te_s2, te_v2)
        policy, _ = train_gated(
            s_tr2, s_te2, args.epochs, args.N,
            args.lam, args.lr, seed,
            gate_threshold_init=best_theta,
            gate_scale_init=args.width,
            freeze_gate=True,
        )
        r = eval_window(policy, "2024 Full year", *FINAL_TEST_WINDOWS["2024 Full year"])
        delta = r["diff"][0] if r is not None else float("nan")
        all_2024_diffs.append(delta)
        print(f"    2024 Delta = {delta:+.3f}")

    xs_2024 = np.array([v for v in all_2024_diffs if not np.isnan(v)])
    mean_2024 = xs_2024.mean()
    std_2024  = xs_2024.std()

    # Pre-registered 2024 pass criterion (same as Week 12)
    win_ok = mean_2024 > 0 and std_2024 < abs(mean_2024)

    print(f"\n  2024 Final Test:  mean Delta = {mean_2024:+.3f}  "
          f"std = {std_2024:.3f}")
    print(f"  Pre-registered pass (mean>0 AND std<|mean|): "
          f"{'[OK] PASS' if win_ok else '[FAIL] FAIL'}")
    print(f"\n  Comparison:")
    print(f"    Week-13 (theta selected on all 7 windows): +1.11 +/- 0.10")
    print(f"    This run (theta selected on 2022-23 only): "
          f"{mean_2024:+.3f} +/- {std_2024:.3f}")

    # ---------------------------------------------------------------- #
    # Save results                                                      #
    # ---------------------------------------------------------------- #
    out_path = "results/gate_validation/threshold_selection_results.txt"
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("GATE THRESHOLD VALIDATION -- STEP 1 POST-WEEK-13\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"epochs={args.epochs}  seeds={args.seeds}  N={args.N}  "
                f"lr={args.lr}  lam={args.lam}  width={args.width}\n\n")
        f.write("SELECTION CRITERION (pre-registered):\n")
        f.write("  theta* = argmax mean(Delta_2022, Delta_2023) s.t. "
                "COVID mean > -2.5 AND std < 2.0\n\n")
        f.write("SWEEP (validation + COVID sanity windows only):\n")
        f.write(f"{'theta':<6} {'val_score':>10} {'COVID mean':>12} "
                f"{'COVID std':>10} {'feasible':>10}\n")
        for theta, vs, cm, cs, ok in selection_log:
            f.write(f"{theta:<6.1f} {vs:>+10.3f} {cm:>+12.3f} "
                    f"{cs:>10.3f} {'yes' if ok else 'no':>10}\n")
        f.write(f"\nSelected theta* = {best_theta}\n\n")
        f.write(f"HELD-OUT 2024 RESULT (theta* = {best_theta}):\n")
        f.write(f"  Per-seed Delta: {[f'{v:+.3f}' for v in all_2024_diffs]}\n")
        f.write(f"  mean = {mean_2024:+.3f}  std = {std_2024:.3f}\n")
        f.write(f"  Pre-registered pass: {'PASS' if win_ok else 'FAIL'}\n")
        f.write("\nComparison:\n")
        f.write("  Week-13 (test-set tuned):   +1.11 +/- 0.10\n")
        f.write(f"  This run (val-set selected): "
                f"{mean_2024:+.3f} +/- {std_2024:.3f}\n")
    print(f"\n  Results written -> {out_path}")

    # ---------------------------------------------------------------- #
    # Plot: validation sweep + held-out 2024                          #
    # ---------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: validation scores per threshold
    ax = axes[0]
    val_means = []
    val_stds  = []
    covid_means = []
    covid_stds  = []
    for theta in THRESHOLDS:
        agg = sweep_results[theta]
        vs = []
        for L in VALIDATION_WINDOWS:
            if L in agg:
                vs.append(agg[L][0])
        val_means.append(np.mean(vs) if vs else 0)
        val_stds.append(np.mean([agg[L][1] for L in VALIDATION_WINDOWS
                                  if L in agg]))
        cm, cs = agg.get("2020 COVID", (0, 0))
        covid_means.append(cm)
        covid_stds.append(cs)

    x = np.arange(len(THRESHOLDS))
    bars = ax.bar(x, val_means, yerr=val_stds, capsize=5,
                  color=["#4CAF50" if t == best_theta else "#90A4AE"
                         for t in THRESHOLDS],
                  edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"theta={t}" for t in THRESHOLDS])
    ax.set_ylabel("Validation score: mean(Delta_2022, Delta_2023)")
    ax.set_title("Threshold sweep -- validation windows only\n"
                 "(2024 held out during selection)")
    for i, (bar, cm, cs) in enumerate(zip(bars, covid_means, covid_stds)):
        ok = cm > COVID_MEAN_THRESH and cs < COVID_STD_THRESH
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (val_stds[i] or 0) + 0.05,
                "[OK]" if ok else "[FAIL]",
                ha="center", va="bottom", fontsize=14,
                color="#4CAF50" if ok else "#E53935")
    ax.text(0.97, 0.97,
            f"*** theta* = {best_theta}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color="#4CAF50",
            bbox=dict(facecolor="white", edgecolor="#4CAF50", boxstyle="round"))

    # Right: held-out 2024 per-seed results
    ax2 = axes[1]
    ax2.scatter(range(len(all_2024_diffs)), all_2024_diffs,
                color="#1565C0", s=80, zorder=3, label="Per-seed Delta")
    ax2.axhline(mean_2024, color="#1565C0", linewidth=1.5,
                linestyle="-", label=f"Mean {mean_2024:+.3f}")
    ax2.axhline(mean_2024 + std_2024, color="#1565C0",
                linewidth=0.8, linestyle=":")
    ax2.axhline(mean_2024 - std_2024, color="#1565C0",
                linewidth=0.8, linestyle=":")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(1.11, color="grey", linewidth=1, linestyle="--",
                label="Week-13 +1.11 (test-set tuned)")
    ax2.fill_between(range(len(all_2024_diffs)),
                     mean_2024 - std_2024, mean_2024 + std_2024,
                     alpha=0.15, color="#1565C0")
    ax2.set_xticks(range(len(all_2024_diffs)))
    ax2.set_xticklabels([f"seed {i}" for i in range(len(all_2024_diffs))])
    ax2.set_ylabel("CVaR$_{95}$ gap Delta = Deep - BS (2024 held-out)")
    ax2.set_title(f"Held-out 2024 test at theta* = {best_theta}\n"
                  f"mean={mean_2024:+.3f} +/- {std_2024:.3f}  "
                  f"{'PASS [OK]' if win_ok else 'FAIL [FAIL]'}")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = "results/gate_validation/threshold_selection.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure written -> {fig_path}")


if __name__ == "__main__":
    main()
