"""
main_expanded_corpus.py
------------------------
Week 10 / Year-2 Sprint 1: direct test of the Week 9 regime-coverage
hypothesis.

Week 9 claimed the sim-to-real failure decomposes into:
    (a) in-period distribution shift  --- fixable by real-data training
    (b) out-of-period regime novelty  --- not fixable by any training
                                          set that does not contain the
                                          target regime

To test (b), we extend the training pool from 2005-2017 to 2005-2020,
which NOW includes the COVID-19 crash. If Week 9's hypothesis is right,
the 2020 COVID window should tie or beat BS delta when it lies inside
the training period, and the new "genuinely OOT" windows 2022 and
2023--2024 should show the same residual failure pattern.

Train : 2005-01-01 to 2020-12-31
Test  : 2021-01-01 to 2024-12-31
       + retrospective evaluation on 2008 / 2017 / 2018 / 2020 (now
       all in-train) to compare vs Week 9.

Usage:
    python main_expanded_corpus.py --epochs 400 --N 256 --lam 0.5
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import yfinance as yf
from scipy.stats import norm

from policy.network_varswap import HedgingPolicyVarSwap
from market.historical_bootstrap import (
    HistoricalBootstrap, payoff_call_atm,
    T, S0, K, SIGMA_V_FIXED, ewma_variance,
)
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci
from main_backtest_training import (
    cvar_loss, rollout_batch, build_windows, bs_pnl_per_window,
    deep_hedger_pnl, download_prices, plot_learning, COST_RATE, BS_SIGMA,
)

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_START = "2005-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2024-12-31"


def load_expanded_corpus(use_cache=True):
    cache = "results/spy_expanded_cache.npz"
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return z["train_ret"], z["test_ret"]
    train_px = download_prices(TRAIN_START, TRAIN_END)
    test_px  = download_prices(TEST_START,  TEST_END)
    tr = np.diff(np.log(train_px))
    te = np.diff(np.log(test_px))
    np.savez(cache, train_ret=tr, test_ret=te)
    return tr, te


def train(sampler_tr, sampler_te, epochs, N, lam, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    history = {"epoch": [], "loss": [], "cvar_train": [],
               "cvar_val": [], "mean_train": []}

    for ep in range(epochs):
        alpha = 0.80 + 0.15 * min(ep / max(epochs - 1, 1), 1.0)
        S, VS = sampler_tr.sample_batch(N, device=device)
        pnl = rollout_batch(policy, S, VS, payoff_call_atm)
        c = cvar_loss(pnl, alpha)
        loss = c + lam * torch.abs(pnl.mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                Sv, VSv = sampler_te.sample_batch(512, device=device)
                pv = rollout_batch(policy, Sv, VSv, payoff_call_atm)
                cv = cvar_loss(pv, 0.95).item()
            history["epoch"].append(ep + 1)
            history["loss"].append(loss.item())
            history["cvar_train"].append(-c.item())
            history["cvar_val"].append(-cv)
            history["mean_train"].append(pnl.mean().item())
            print(f"ep {ep+1:4d}  loss={loss.item():+.3f}  "
                  f"CVaR_train={-c.item():+.3f}  "
                  f"CVaR_val(2021-24)={-cv:+.3f}")
    return policy, history


def evaluate(policy):
    periods = {
        "2008 GFC (in-train)":   ("2007-09-01", "2009-06-30"),
        "2017 Calm (in-train)":  ("2016-07-01", "2017-12-31"),
        "2018 Volmageddon (in-train)": ("2018-01-01", "2018-12-31"),
        "2020 COVID (in-train NOW)":   ("2019-11-01", "2020-12-31"),
        "2022 Rate shock (OOT)": ("2022-01-01", "2022-12-31"),
        "2023 SVB / banking (OOT)": ("2023-01-01", "2023-12-31"),
        "2024 Full year (OOT)":  ("2024-01-01", "2024-12-31"),
    }
    res = {}
    for label, (s, e) in periods.items():
        px = download_prices(s, e)
        Sw, VSw = build_windows(px)
        p_bs = bs_pnl_per_window(Sw)
        p_dh = deep_hedger_pnl(policy, Sw, VSw)
        c_bs, lo_bs, hi_bs = bootstrap_cvar_ci(p_bs, 0.95, B=500)
        c_dh, lo_dh, hi_dh = bootstrap_cvar_ci(p_dh, 0.95, B=500)
        d, dlo, dhi = bootstrap_diff_ci(p_dh, p_bs, 0.95, B=500)
        res[label] = dict(n=len(Sw),
                          bs=(c_bs, lo_bs, hi_bs), dh=(c_dh, lo_dh, hi_dh),
                          diff=(d, dlo, dhi),
                          std_bs=float(p_bs.std()), std_dh=float(p_dh.std()))
        print(f"{label:<32}  n={len(Sw):3d}  "
              f"BS={c_bs:+7.2f}[{lo_bs:+6.2f},{hi_bs:+6.2f}]  "
              f"DH={c_dh:+7.2f}[{lo_dh:+6.2f},{hi_dh:+6.2f}]  "
              f"Delta={d:+6.2f}[{dlo:+6.2f},{dhi:+6.2f}]")
    return res


def write_metrics(res, path):
    with open(path, "w") as f:
        f.write("Sprint 1 -- Expanded training corpus (SPY 2005-2020)\n")
        f.write("=" * 72 + "\n\n")
        f.write("Held-out test pool: 2021-01-01 -> 2024-12-31\n\n")
        for L, r in res.items():
            c_bs, lo_bs, hi_bs = r["bs"]
            c_dh, lo_dh, hi_dh = r["dh"]
            d, dlo, dhi = r["diff"]
            f.write(f"{L}  (n={r['n']} windows)\n")
            f.write(f"  BS    CVaR95 = {c_bs:+7.3f}  [{lo_bs:+.3f}, {hi_bs:+.3f}]  std={r['std_bs']:.3f}\n")
            f.write(f"  Deep  CVaR95 = {c_dh:+7.3f}  [{lo_dh:+.3f}, {hi_dh:+.3f}]  std={r['std_dh']:.3f}\n")
            f.write(f"  Delta(Deep-BS)   = {d:+7.3f}  [{dlo:+.3f}, {dhi:+.3f}]\n\n")


def plot_cvar_bars(res, path):
    labels = list(res.keys())
    c_bs = [res[L]["bs"][0] for L in labels]
    c_dh = [res[L]["dh"][0] for L in labels]
    err_bs = [[c_bs[i] - res[L]["bs"][1] for i, L in enumerate(labels)],
              [res[L]["bs"][2] - c_bs[i] for i, L in enumerate(labels)]]
    err_dh = [[c_dh[i] - res[L]["dh"][1] for i, L in enumerate(labels)],
              [res[L]["dh"][2] - c_dh[i] for i, L in enumerate(labels)]]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w/2, c_bs, w, yerr=err_bs, capsize=4, color="#2196F3",
           edgecolor="black", linewidth=0.5, label="BS delta")
    ax.bar(x + w/2, c_dh, w, yerr=err_dh, capsize=4, color="#FF5722",
           edgecolor="black", linewidth=0.5, label="Deep (2005-2020 trained)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("CVaR₉₅ (higher = better)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Sprint 1: Expanded corpus -- direct test of the regime-coverage hypothesis")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=256)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save",   default="results/expanded_corpus.pth")
    args = ap.parse_args()

    print(f"device={device}  epochs={args.epochs}  N={args.N}")
    tr, te = load_expanded_corpus()
    print(f"Train pool: {len(tr)} returns (2005-2020)")
    print(f"Test  pool: {len(te)} returns (2021-2024)")
    s_tr = HistoricalBootstrap(tr)
    s_te = HistoricalBootstrap(te)

    policy, history = train(s_tr, s_te, args.epochs, args.N,
                            args.lam, args.lr, args.seed)
    torch.save(policy.state_dict(), args.save)
    print(f"Saved -> {args.save}")

    print("\n=== Evaluation ===")
    res = evaluate(policy)

    write_metrics(res, "results/expanded_corpus_metrics.txt")
    plot_learning(history, "results/expanded_corpus_learning.png")
    plot_cvar_bars(res, "results/expanded_corpus_cvar.png")
    print("\nArtifacts in results/expanded_corpus_*.{txt,png}")


if __name__ == "__main__":
    main()
