"""
main_vix_with_costs.py
-----------------------
Step 2 of the post-Week-13 roadmap: VIX futures with realistic
bid-ask transaction costs on the VIX leg.

MOTIVATION
----------
Every real-data result in Sprints 2-3 and Week 13 uses VIX as a
tradable instrument but charges *only* the same flat cost_rate on the
VIX position as on the stock leg (0.02% of notional). In reality:

  - VIX futures (SPVXSP front-month) have a bid-ask spread of roughly
    0.05 - 0.30 VIX points on the front-month contract.
  - At a VIX level of ~20, that is 0.05/20 = 0.25% to 0.30/20 = 1.5%.
  - Rebalancing weekly means we pay this 6x per 30-day window.

KEY QUESTION
------------
Does the calm-market win (+1.11 CVaR at theta=3.0) survive after paying
a realistic VIX bid-ask spread?

This script sweeps four spread levels (in VIX points) and reports:
  - CVaR improvement Delta = Deep - BS on every evaluation window
  - Breakeven spread: highest spread at which Delta > 0 on 2024 OOT

SPREAD LEVELS SWEPT
-------------------
  0.00  VIX pts  -> baseline (no extra VIX cost, matches Week 13)
  0.05  VIX pts  -> tight market (liquid VIX futures, low vol regime)
  0.10  VIX pts  -> typical (mid-range, most common assumption)
  0.20  VIX pts  -> wide (high vol, stressed market)
  0.30  VIX pts  -> extreme (crisis bid-ask, e.g. March 2020 levels)

IMPLEMENTATION
--------------
The transaction cost on the VIX leg is modified from:
  tc_vix = cost_rate * |Delta_vix| * (VS/VS0)          [current]
to:
  tc_vix = cost_rate * |Delta_vix| * (VS/VS0)
         + (spread / 2) * |Delta_vix_position_in_contracts|   [new]

where spread is in normalised VIX units (VIX pts / VS0, and VS0=100
since VIX is normalised to 100 at window start). So:
  extra_tc = (spread / 100) * |Delta_vix|

This is applied symmetrically on both buys and sells.

Usage
-----
    python main_vix_with_costs.py [--seeds N] [--epochs N] [--theta F]
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from policy.network_vix_gated import VIXGatedPolicy
from market.vix_bootstrap import VIXBootstrap, payoff_call_atm, T
from data.vix_windows import load as load_spy_vix
from main_backtest_training import cvar_loss
from main_vix_futures import (
    download_spy_vix, build_weekly_windows, bs_pnl_weekly,
)
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci

os.makedirs("results/vix_costs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

import os as _os
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ------------------------------------------------------------------ #
# Spread levels (in VIX points -- absolute, not normalised)           #
# ------------------------------------------------------------------ #
SPREAD_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30]

# Evaluation windows
EVAL_WINDOWS = {
    "2008 GFC":           ("2007-09-01", "2009-06-30"),
    "2017 Calm":          ("2016-07-01", "2017-12-31"),
    "2018 Volmageddon":   ("2018-01-01", "2018-12-31"),
    "2020 COVID":         ("2019-11-01", "2020-12-31"),
    "2022 Rate shock":    ("2022-01-01", "2022-12-31"),
    "2023 SVB":           ("2023-01-01", "2023-12-31"),
    "2024 Full year":     ("2024-01-01", "2024-12-31"),
}


# ------------------------------------------------------------------ #
# Cost-aware rollout                                                  #
# ------------------------------------------------------------------ #
def rollout_with_spread(policy, S, VS, payoff_fn, vix_spread_pts=0.0):
    """
    Rollout with regime gate + explicit VIX bid-ask spread cost.

    vix_spread_pts : bid-ask spread in VIX points (absolute).
                     Per-trade cost = (spread/2) / VS0 per unit position.
                     Charged on the absolute change in VIX position.
    """
    N, Tp1 = S.shape
    Tn = Tp1 - 1
    pnl     = torch.zeros(N, device=S.device)
    prev_S  = torch.zeros(N, device=S.device)
    prev_V  = torch.zeros(N, device=S.device)

    # Spread in normalised units: VIX is normalised to VS0=100 at t=0
    spread_normalised = vix_spread_pts / policy.VS0   # e.g. 0.10/100 = 0.001

    for t in range(Tn):
        vs_norm = VS[:, t] / policy.VS0
        state = torch.stack([
            S[:, t] / policy.S0,
            vs_norm,
            torch.full((N,), t / Tn, device=S.device),
            prev_S,
            prev_V,
        ], dim=1)

        a  = policy.forward(state)
        dS = torch.tanh(a[:, 0]) * policy.stock_scale
        g  = policy.gate(vs_norm)
        dV = torch.tanh(a[:, 1]) * policy.vix_scale * g

        gain_S = prev_S * (S[:,  t + 1] - S[:,  t])
        gain_V = prev_V * (VS[:, t + 1] - VS[:, t])

        # Standard proportional cost (same as before)
        tc_stock = (policy.cost_rate
                    * torch.abs(dS - prev_S)
                    * (S[:, t] / policy.S0))
        tc_vix_prop = (policy.cost_rate
                       * torch.abs(dV - prev_V)
                       * (VS[:, t] / policy.VS0))
        # NEW: half-spread cost on VIX leg (paid on both sides)
        tc_vix_spread = spread_normalised * torch.abs(dV - prev_V)

        tc = tc_stock + tc_vix_prop + tc_vix_spread
        pnl = pnl + gain_S + gain_V - tc
        prev_S, prev_V = dS, dV

    pnl = pnl - payoff_fn(S[:, -1])
    return pnl


# ------------------------------------------------------------------ #
# Training (same as trainer_gated but uses rollout_with_spread)       #
# ------------------------------------------------------------------ #
def train_with_spread(s_tr, s_te, epochs, N, lam, lr, seed,
                      theta, width, train_spread=0.0):
    """
    Train the gated policy with a fixed VIX spread cost baked into
    training. Using train_spread=0 matches Week-13 (naive, no spread).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = VIXGatedPolicy(
        gate_threshold_init=theta,
        gate_scale_init=width,
        freeze_gate=True,
    ).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        alpha = 0.80 + 0.15 * min(ep / max(epochs - 1, 1), 1.0)
        S, VS = s_tr.sample_batch(N, device=device)
        pnl = rollout_with_spread(policy, S, VS, payoff_call_atm,
                                  vix_spread_pts=train_spread)
        c = cvar_loss(pnl, alpha)
        loss = c + lam * torch.abs(pnl.mean())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"  ep {ep+1:4d}  loss={loss.item():+.3f}  "
                  f"CVaR={-c.item():+.3f}  mean={pnl.mean().item():+.3f}")

    return policy


# ------------------------------------------------------------------ #
# Evaluation                                                          #
# ------------------------------------------------------------------ #
def evaluate_policy_spread(policy, eval_spread_pts):
    """
    Evaluate a trained policy at a given evaluation spread level.
    Returns {label: delta_float}.
    """
    results = {}
    for label, (start, end) in EVAL_WINDOWS.items():
        df = download_spy_vix(start, end)
        if df is None or len(df) < T + 2:
            continue
        Sw, VSw = build_weekly_windows(df)
        if len(Sw) == 0:
            continue

        p_bs = bs_pnl_weekly(Sw)
        S_t  = torch.tensor(Sw,  dtype=torch.float32, device=device)
        VS_t = torch.tensor(VSw, dtype=torch.float32, device=device)

        with torch.no_grad():
            p_dh_t = rollout_with_spread(
                policy, S_t, VS_t, payoff_call_atm,
                vix_spread_pts=eval_spread_pts)
        p_dh = p_dh_t.cpu().numpy()

        d, dlo, dhi = bootstrap_diff_ci(p_dh, p_bs, 0.95, B=500)
        results[label] = (d, dlo, dhi)
    return results


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(
        description="VIX futures with realistic bid-ask costs")
    ap.add_argument("--seeds",  type=int, default=3,
                    help="Seeds per spread level (3 is enough for cost check)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--theta",  type=float, default=3.0,
                    help="Gate threshold (use theta* from Step 1)")
    ap.add_argument("--width",  type=float, default=0.3)
    ap.add_argument("--train-spread", type=float, default=0.0,
                    help="VIX spread to bake into training (0=naive, "
                         "0.10=matched to typical evaluation spread)")
    args = ap.parse_args()

    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)

    print("=" * 72)
    print("VIX FUTURES WITH REALISTIC BID-ASK COSTS")
    print(f"theta = {args.theta}  width = {args.width}  "
          f"train_spread = {args.train_spread} pts")
    print(f"seeds = {args.seeds}  epochs = {args.epochs}")
    print(f"Evaluating at spreads: {SPREAD_LEVELS} VIX pts")
    print("=" * 72)

    # ---------------------------------------------------------------- #
    # Train policies (one per seed)                                     #
    # ---------------------------------------------------------------- #
    policies = []
    for seed in range(args.seeds):
        print(f"\n[seed {seed}] Training...")
        p = train_with_spread(
            s_tr, s_te,
            epochs=args.epochs, N=args.N,
            lam=args.lam, lr=args.lr, seed=seed,
            theta=args.theta, width=args.width,
            train_spread=args.train_spread,
        )
        policies.append(p)

    # ---------------------------------------------------------------- #
    # Evaluate each policy at every spread level                        #
    # ---------------------------------------------------------------- #
    # Structure: spread_results[spread][label] = list of per-seed deltas
    spread_results = {s: {L: [] for L in EVAL_WINDOWS} for s in SPREAD_LEVELS}

    for seed_idx, policy in enumerate(policies):
        for spread in SPREAD_LEVELS:
            print(f"\n  Evaluating seed {seed_idx} at spread={spread:.2f} pts...")
            res = evaluate_policy_spread(policy, eval_spread_pts=spread)
            for label, (d, dlo, dhi) in res.items():
                spread_results[spread][label].append(d)
                print(f"    {label:<32}  Delta={d:+.3f} [{dlo:+.3f},{dhi:+.3f}]")

    # ---------------------------------------------------------------- #
    # Aggregate and report                                              #
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 72)
    print("AGGREGATED RESULTS (mean +/- seed-std across seeds)")
    print("=" * 72)

    agg = {}   # spread -> {label: (mean, std)}
    for spread in SPREAD_LEVELS:
        agg[spread] = {}
        for label in EVAL_WINDOWS:
            xs = np.array(spread_results[spread][label])
            if len(xs):
                agg[spread][label] = (float(xs.mean()), float(xs.std()))

    # Print table
    header = f"{'Window':<32}"
    for spread in SPREAD_LEVELS:
        header += f"  {'sp=' + str(spread):>10}"
    print(header)
    print("-" * (32 + 12 * len(SPREAD_LEVELS)))
    for label in EVAL_WINDOWS:
        row = f"{label:<32}"
        for spread in SPREAD_LEVELS:
            m, s = agg[spread].get(label, (float("nan"), 0))
            row += f"  {m:>+7.2f}+/-{s:.2f}"
        print(row)

    # Breakeven spread on 2024 OOT
    print("\nBreakeven spread on 2024 Full year:")
    breakeven = None
    for spread in SPREAD_LEVELS:
        m, _ = agg[spread].get("2024 Full year", (-1, 0))
        sign = "+" if m > 0 else "-"
        print(f"  spread={spread:.2f}pts  Delta_2024={m:+.3f}  "
              f"{'above breakeven [OK]' if m > 0 else 'below breakeven [FAIL]'}")
        if m > 0 and breakeven is None:
            breakeven_candidate = spread
        if m <= 0 and breakeven is None and spread > 0:
            breakeven = spread

    if breakeven:
        print(f"\n  Breakeven is between {SPREAD_LEVELS[SPREAD_LEVELS.index(breakeven)-1]}"
              f" and {breakeven} VIX pts")
    else:
        print("\n  Win persists at all tested spread levels")

    # ---------------------------------------------------------------- #
    # Save results                                                      #
    # ---------------------------------------------------------------- #
    out_path = "results/vix_costs/spread_sensitivity_results.txt"
    with open(out_path, "w", encoding='utf-8') as f:
        f.write("VIX FUTURES BID-ASK SENSITIVITY -- STEP 2\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"theta={args.theta}  width={args.width}  "
                f"train_spread={args.train_spread}\n"
                f"seeds={args.seeds}  epochs={args.epochs}\n\n")
        f.write(f"{'Window':<32}")
        for s in SPREAD_LEVELS:
            f.write(f"  {'sp='+str(s):>10}")
        f.write("\n" + "-" * (32 + 12 * len(SPREAD_LEVELS)) + "\n")
        for label in EVAL_WINDOWS:
            f.write(f"{label:<32}")
            for spread in SPREAD_LEVELS:
                m, s = agg[spread].get(label, (float("nan"), 0))
                f.write(f"  {m:>+7.2f}+/-{s:.2f}")
            f.write("\n")
    print(f"\nResults written -> {out_path}")

    # ---------------------------------------------------------------- #
    # Plot                                                              #
    # ---------------------------------------------------------------- #
    # Focus on 2024 OOT (the headline win) and COVID (risk window)
    focus_windows = {
        "2024 Full year":  ("#1565C0", "solid"),
        "2020 COVID":      ("#C62828", "dashed"),
        "2017 Calm":       ("#2E7D32", "dotted"),
        "2022 Rate shock": ("#F57F17", "dashdot"),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (colour, ls) in focus_windows.items():
        means = []
        stds  = []
        for spread in SPREAD_LEVELS:
            m, s = agg[spread].get(label, (float("nan"), 0))
            means.append(m)
            stds.append(s)
        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(SPREAD_LEVELS, means, color=colour, linestyle=ls,
                linewidth=2, marker="o", markersize=6, label=label)
        ax.fill_between(SPREAD_LEVELS,
                        means - stds, means + stds,
                        alpha=0.12, color=colour)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("VIX bid-ask spread (pts)")
    ax.set_ylabel("CVaR$_{95}$ gap Delta = Deep - BS  (mean +/- seed-std)")
    ax.set_title(f"Calm-market win vs VIX transaction costs\n"
                 f"theta={args.theta}, w={args.width}, "
                 f"train_spread={args.train_spread} pts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = "results/vix_costs/spread_sensitivity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure written -> {fig_path}")


if __name__ == "__main__":
    main()
