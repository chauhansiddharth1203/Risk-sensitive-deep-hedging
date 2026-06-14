"""
main_nsde_hedge.py
-------------------
Step 3b: Train the hedging policy on Neural SDE generated paths and
evaluate on real SPY windows.

HYPOTHESIS UNDER TEST (from Sprint 1 diagnosis)
------------------------------------------------
Sprint 1 showed that extending training to include 2020 COVID only closed
the COVID gap by ~23%, not the expected ~100% if coverage were the sole
issue. Sprint 1 concluded the residual failure was driven by the lagging
EWMA proxy.

This experiment tests a DIFFERENT but complementary hypothesis:
  "A generative Neural SDE simulator that can produce COVID-like
   episodes (extreme VIX spikes) gives the policy MORE relevant
   training signal than block bootstrap, because the Neural SDE can
   INTERPOLATE in latent space to produce novel spike scenarios rather
   than only RESAMPLING observed ones."

If this hypothesis is correct, Neural SDE training should produce lower
COVID CVaR than block bootstrap training, even though both use the same
2005-2020 training window.

PROTOCOL
--------
1. Load the trained NSDEGenerator from results/nsde/nsde_generator.pth
2. Create NSDEBootstrap (same interface as VIXBootstrap)
3. Train VIXGatedPolicy (theta=3.0, same as Week 13 / Step 1) on
   Neural SDE paths for 400 epochs
4. Evaluate on the same 7 real SPY windows as Steps 1-2
5. Compare: Neural SDE vs Block Bootstrap (Step 1 result)

PRE-REGISTERED DECISION RULE (before running)
----------------------------------------------
  Neural SDE is superior to block bootstrap if:
    COVID mean Delta(NSDE) > COVID mean Delta(bootstrap) by > 0.5 CVaR
    AND 2024 OOT mean Delta(NSDE) > 0

Usage
-----
    python main_nsde_hedge.py [--seeds N] [--epochs N]

    Run main_nsde_train.py first to create the generator.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from market.neural_sde import NSDEGenerator, NSDEBootstrap
from data.vix_windows import load as load_spy_vix
from market.vix_bootstrap import VIXBootstrap, payoff_call_atm
from training.trainer_gated import train_gated, rollout_batch_gated
from main_vix_futures import download_spy_vix, build_weekly_windows, bs_pnl_weekly
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci

os.makedirs("results/nsde", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Same evaluation windows as all prior experiments
EVAL_WINDOWS = {
    "2008 GFC":           ("2007-09-01", "2009-06-30"),
    "2017 Calm":          ("2016-07-01", "2017-12-31"),
    "2018 Volmageddon":   ("2018-01-01", "2018-12-31"),
    "2020 COVID":         ("2019-11-01", "2020-12-31"),
    "2022 Rate shock":    ("2022-01-01", "2022-12-31"),
    "2023 SVB":           ("2023-01-01", "2023-12-31"),
    "2024 Full year":     ("2024-01-01", "2024-12-31"),
}

# Gate params from Step 1 (principled selection)
GATE_THETA = 3.0
GATE_WIDTH = 0.3


def load_generator(path="results/nsde/nsde_generator.pth"):
    """Load saved NSDEGenerator and normalisation stats."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hparams"]
    gen = NSDEGenerator(
        noise_dim=hp["noise_dim"],
        hidden_dim=hp["hidden_dim"],
        n_layers=hp["n_layers"],
    )
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()
    return gen, ckpt["mu"], ckpt["std"]


def eval_window(policy, label, start, end):
    """Evaluate policy on one real-data window."""
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
    d, dlo, dhi = bootstrap_diff_ci(p_dh, p_bs, 0.95, B=500)
    return (d, dlo, dhi)


def main():
    ap = argparse.ArgumentParser(
        description="Train hedger on Neural SDE paths")
    ap.add_argument("--seeds",  type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--gen-path", default="results/nsde/nsde_generator.pth")
    args = ap.parse_args()

    # ---- Load generator ----
    if not os.path.exists(args.gen_path):
        print(f"ERROR: generator not found at {args.gen_path}")
        print("Run main_nsde_train.py first.")
        return

    print("=" * 72)
    print("NEURAL SDE HEDGER TRAINING")
    print(f"Generator: {args.gen_path}")
    print(f"seeds={args.seeds}  epochs={args.epochs}  N={args.N}")
    print(f"Gate: theta={GATE_THETA}  width={GATE_WIDTH}  (from Step 1)")
    print("=" * 72)
    print("\nPRE-REGISTERED DECISION RULE:")
    print("  NSDE beats bootstrap if:")
    print("    COVID Delta(NSDE) > COVID Delta(bootstrap) + 0.5 CVaR")
    print("    AND 2024 OOT Delta(NSDE) > 0")
    print("\n  Step 1 bootstrap COVID: mean Delta = -1.177 +/- 0.410")
    print("  NSDE needs COVID Delta > -0.677 to beat bootstrap")
    print("=" * 72)

    gen, mu, std = load_generator(args.gen_path)
    s_nsde = NSDEBootstrap(gen, mu, std, device=device)

    # Also prepare block bootstrap for validation set (same as trainer_gated)
    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_te = VIXBootstrap(te_s, te_v)

    # ---- Train seeds ----
    all_results = []
    for seed in range(args.seeds):
        print(f"\n[seed {seed}] Training on Neural SDE paths...")
        policy, _ = train_gated(
            s_nsde, s_te,
            epochs=args.epochs, N=args.N,
            lam=args.lam, lr=args.lr, seed=seed,
            gate_threshold_init=GATE_THETA,
            gate_scale_init=GATE_WIDTH,
            freeze_gate=True,
        )

        print(f"[seed {seed}] Evaluating on real SPY windows...")
        seed_res = {}
        for label, (start, end) in EVAL_WINDOWS.items():
            r = eval_window(policy, label, start, end)
            if r is not None:
                seed_res[label] = r[0]  # point estimate Delta
                print(f"  {label:<32} Delta={r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]")
            else:
                seed_res[label] = float("nan")
        all_results.append(seed_res)

    # ---- Aggregate ----
    labels = list(EVAL_WINDOWS.keys())
    print("\n" + "=" * 72)
    print("AGGREGATED RESULTS (Neural SDE training)")
    print("=" * 72)
    print(f"{'Window':<32}  {'mean Delta':>10}  {'std':>6}  {'verdict':>20}")
    print("-" * 72)

    nsde_agg = {}
    for L in labels:
        xs = np.array([r.get(L, float("nan")) for r in all_results])
        xs = xs[~np.isnan(xs)]
        m, s = (xs.mean(), xs.std()) if len(xs) else (float("nan"), 0)
        nsde_agg[L] = (m, s)
        verdict = "[OK] win" if m > 0 else ("[OK] small loss" if m > -2 else "[FAIL] loss")
        print(f"{L:<32}  {m:>+10.3f}  {s:>6.3f}  {verdict:>20}")

    # ---- Decision rule ----
    covid_m, covid_s = nsde_agg.get("2020 COVID", (float("nan"), 0))
    oot_m,   oot_s   = nsde_agg.get("2024 Full year", (float("nan"), 0))
    bootstrap_covid = -1.177   # from Step 1

    rule1 = covid_m > bootstrap_covid + 0.5
    rule2 = oot_m > 0

    print("\nDecision rule check:")
    print(f"  COVID Delta(NSDE)  = {covid_m:+.3f} +/- {covid_s:.3f}")
    print(f"  COVID Delta(bootstrap) = {bootstrap_covid:+.3f}")
    print(f"  Rule 1 (NSDE COVID > bootstrap + 0.5): "
          f"{covid_m:+.3f} > {bootstrap_covid + 0.5:+.3f} -> "
          f"{'PASS' if rule1 else 'FAIL'}")
    print(f"  Rule 2 (2024 OOT > 0): {oot_m:+.3f} > 0 -> "
          f"{'PASS' if rule2 else 'FAIL'}")

    if rule1 and rule2:
        print("\n  [PASS] Neural SDE is superior to block bootstrap.")
        print("  The generator provides novel COVID-like training signal.")
    elif rule1:
        print("\n  [PARTIAL] COVID improves but 2024 OOT win lost.")
    elif rule2:
        print("\n  [PARTIAL] 2024 OOT preserved but COVID not improved.")
        print("  Sprint 1 lagging-proxy hypothesis holds: generator alone")
        print("  is not sufficient -- the regime gate is load-bearing.")
    else:
        print("\n  [FAIL] Neural SDE does not outperform block bootstrap.")
        print("  Hypothesis: regime gate is more important than simulator.")

    # ---- Save results ----
    out_path = "results/nsde/hedge_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("NEURAL SDE HEDGER RESULTS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"seeds={args.seeds}  epochs={args.epochs}\n")
        f.write(f"Gate: theta={GATE_THETA}  width={GATE_WIDTH}\n\n")
        f.write(f"{'Window':<32}  {'mean Delta':>10}  {'std':>6}\n")
        f.write("-" * 52 + "\n")
        for L in labels:
            m, s = nsde_agg.get(L, (float("nan"), 0))
            f.write(f"{L:<32}  {m:>+10.3f}  {s:>6.3f}\n")
        f.write(f"\nCOVID: NSDE={covid_m:+.3f}  bootstrap={bootstrap_covid:+.3f}\n")
        f.write(f"Decision: Rule1={'PASS' if rule1 else 'FAIL'}  "
                f"Rule2={'PASS' if rule2 else 'FAIL'}\n")
    print(f"\nResults written -> {out_path}")

    # ---- Plot comparison ----
    # Compare NSDE vs block bootstrap (Step 1) on all 7 windows
    bootstrap_means = {
        "2008 GFC":       -2.53,
        "2017 Calm":      -0.44,
        "2018 Volmageddon": -0.44,
        "2020 COVID":     -1.177,
        "2022 Rate shock": -0.93,
        "2023 SVB":       -0.35,
        "2024 Full year": +1.112,
    }

    x = np.arange(len(labels))
    nsde_m = [nsde_agg[L][0] for L in labels]
    nsde_s = [nsde_agg[L][1] for L in labels]
    boot_m = [bootstrap_means.get(L, 0) for L in labels]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, boot_m, 0.38, label="Block bootstrap (Step 1)",
           color="#1565C0", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, nsde_m, 0.38,
           yerr=nsde_s, capsize=4,
           label="Neural SDE (Step 3)",
           color="#E53935", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([L.split(" ")[0] + "\n" + " ".join(L.split(" ")[1:])
                        for L in labels], fontsize=8)
    ax.set_ylabel("CVaR$_{95}$ gap Delta = Deep - BS")
    ax.set_title("Neural SDE vs Block Bootstrap\n"
                 "(same gate theta=3.0, same 7 evaluation windows)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = "results/nsde/hedge_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure written -> {fig_path}")


if __name__ == "__main__":
    main()
