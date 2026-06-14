"""
main_vix_futures.py
--------------------
Sprint 2: VIX-as-vega, weekly rebalancing.

Tests the Sprint 1 hypothesis: the residual COVID hedge failure is
caused by the interaction of daily rebalancing with a lagging
EWMA-variance proxy. Replacing the vega channel with the VIX index
itself (a forward-looking market quote of 30-day implied vol) and
rebalancing weekly should close the COVID gap if the hypothesis is
right.

Train: weekly (SPY, VIX) pairs 2005-2020
Test : weekly (SPY, VIX) pairs 2021-2024
      + rolling 30-day retrospective windows 2007-2024 as in prior work

Usage:
    python main_vix_futures.py --epochs 400 --N 256 --lam 0.5
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import pandas as pd

from scipy.stats import norm
import yfinance as yf

from policy.network_varswap import HedgingPolicyVarSwap
from market.vix_bootstrap import (
    VIXBootstrap, payoff_call_atm, S0, K, T,
)
from data.vix_windows import load as load_spy_vix
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci
from main_backtest_training import (
    cvar_loss, rollout_batch, BS_SIGMA, COST_RATE,
)

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Policy: we reuse HedgingPolicyVarSwap, but it expects VS0 for
# normalisation. Since every sampled window has VS[0]=100 (VIX
# normalised per-window), VS0 = 100 is the right choice. We set that
# via construction.


def make_policy():
    p = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    p.VS0 = 100.0           # override: VIX is normalised to 100 per-window
    return p


# --- Training ----------------------------------------------------------- #

def train(s_tr, s_te, epochs=400, N=256, lam=0.5, lr=3e-4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = make_policy()
    opt = optim.Adam(policy.parameters(), lr=lr)
    hist = {"epoch": [], "loss": [], "cvar_train": [], "cvar_val": [],
            "mean_train": []}
    for ep in range(epochs):
        alpha = 0.80 + 0.15 * min(ep / max(epochs - 1, 1), 1.0)
        S, VS = s_tr.sample_batch(N, device=device)
        pnl = rollout_batch(policy, S, VS, payoff_call_atm)
        c = cvar_loss(pnl, alpha)
        loss = c + lam * torch.abs(pnl.mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                Sv, VSv = s_te.sample_batch(512, device=device)
                pv = rollout_batch(policy, Sv, VSv, payoff_call_atm)
                cv = cvar_loss(pv, 0.95).item()
            hist["epoch"].append(ep + 1)
            hist["loss"].append(loss.item())
            hist["cvar_train"].append(-c.item())
            hist["cvar_val"].append(-cv)
            hist["mean_train"].append(pnl.mean().item())
            print(f"ep {ep+1:4d}  loss={loss.item():+.3f}  "
                  f"CVaR_train={-c.item():+.3f}  "
                  f"mean={pnl.mean().item():+.3f}  "
                  f"CVaR_val(2021-24)={-cv:+.3f}")
    return policy, hist


# --- Historical window evaluation (weekly rebalance) -------------------- #

def download_spy_vix(start, end):
    spy = yf.download("SPY",  start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^VIX", start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    df = pd.concat([spy, vix], axis=1).dropna()
    df.columns = ["SPY", "VIX"]
    return df.resample("W-FRI").last().dropna()


def build_weekly_windows(df, T_win=T, step=1):
    """Rolling overlapping windows of T_win+1 weekly closes, step=1 week."""
    spy = df["SPY"].values
    vix = df["VIX"].values
    Sw, VSw = [], []
    for i in range(0, len(spy) - T_win - 1, step):
        S_  = spy[i:i + T_win + 1]
        V_  = vix[i:i + T_win + 1]
        Sn  = 100.0 * S_ / S_[0]
        Vn  = 100.0 * V_ / V_[0]
        Sw.append(Sn); VSw.append(Vn)
    return np.array(Sw), np.array(VSw)


def bs_delta(S, K, tau, sigma):
    tau = max(tau, 1e-6)
    d1 = (np.log(np.maximum(S, 1e-8) / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bs_pnl_weekly(Sw):
    """BS delta hedge with weekly rebalancing."""
    N, Tp1 = Sw.shape
    Tn = Tp1 - 1
    pnl = np.zeros(N)
    for i in range(N):
        prev_d = 0.0
        for t in range(Tn):
            tau = (Tn - t) / Tn
            d = bs_delta(Sw[i, t], K, tau, BS_SIGMA)
            pnl[i] += prev_d * (Sw[i, t + 1] - Sw[i, t])
            pnl[i] -= COST_RATE * abs(d - prev_d) * (Sw[i, t] / S0)
            prev_d = d
        pnl[i] -= max(Sw[i, -1] - K, 0.0)
    return pnl


def deep_pnl(policy, Sw, VSw):
    S_t  = torch.tensor(Sw,  dtype=torch.float32, device=device)
    VS_t = torch.tensor(VSw, dtype=torch.float32, device=device)
    with torch.no_grad():
        p = rollout_batch(policy, S_t, VS_t, payoff_call_atm)
    return p.cpu().numpy()


def evaluate(policy):
    periods = {
        "2008 GFC (in-train)":        ("2007-09-01", "2009-06-30"),
        "2017 Calm (in-train)":       ("2016-07-01", "2017-12-31"),
        "2018 Volmageddon (in-train)":("2018-01-01", "2018-12-31"),
        "2020 COVID (in-train)":      ("2019-11-01", "2020-12-31"),
        "2022 Rate shock (OOT)":      ("2022-01-01", "2022-12-31"),
        "2023 SVB / banking (OOT)":   ("2023-01-01", "2023-12-31"),
        "2024 Full year (OOT)":       ("2024-01-01", "2024-12-31"),
    }
    res = {}
    for label, (s, e) in periods.items():
        df = download_spy_vix(s, e)
        if len(df) < T + 2:
            print(f"[skip] {label} too few weeks ({len(df)})")
            continue
        Sw, VSw = build_weekly_windows(df)
        p_bs = bs_pnl_weekly(Sw)
        p_dh = deep_pnl(policy, Sw, VSw)
        c_bs, lo_bs, hi_bs = bootstrap_cvar_ci(p_bs, 0.95, B=500)
        c_dh, lo_dh, hi_dh = bootstrap_cvar_ci(p_dh, 0.95, B=500)
        d, dlo, dhi = bootstrap_diff_ci(p_dh, p_bs, 0.95, B=500)
        res[label] = dict(n=len(Sw),
                          bs=(c_bs, lo_bs, hi_bs), dh=(c_dh, lo_dh, hi_dh),
                          diff=(d, dlo, dhi),
                          std_bs=float(p_bs.std()), std_dh=float(p_dh.std()))
        print(f"{label:<32} n={len(Sw):3d}  "
              f"BS={c_bs:+7.2f}[{lo_bs:+6.2f},{hi_bs:+6.2f}]  "
              f"DH={c_dh:+7.2f}[{lo_dh:+6.2f},{hi_dh:+6.2f}]  "
              f"Delta={d:+6.2f}[{dlo:+6.2f},{dhi:+6.2f}]")
    return res


def write_metrics(res, history, path):
    with open(path, "w") as f:
        f.write("Sprint 2 -- VIX-as-vega, weekly rebalancing\n")
        f.write("=" * 72 + "\n\n")
        f.write("Training: weekly SPY+VIX block bootstrap, 2005-2020\n")
        f.write("Testing : overlapping weekly 30-day windows, all periods\n\n")
        for L, r in res.items():
            c_bs, lo_bs, hi_bs = r["bs"]
            c_dh, lo_dh, hi_dh = r["dh"]
            d, dlo, dhi = r["diff"]
            f.write(f"{L}  (n={r['n']} windows)\n")
            f.write(f"  BS    CVaR95 = {c_bs:+7.3f}  [{lo_bs:+.3f}, {hi_bs:+.3f}]  std={r['std_bs']:.3f}\n")
            f.write(f"  Deep  CVaR95 = {c_dh:+7.3f}  [{lo_dh:+.3f}, {hi_dh:+.3f}]  std={r['std_dh']:.3f}\n")
            f.write(f"  Delta(Deep-BS)   = {d:+7.3f}  [{dlo:+.3f}, {dhi:+.3f}]\n\n")


def plot_learning(history, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["cvar_train"], label="CVaR₉₅ (train bootstrap)", color="#1976D2")
    ax.plot(history["epoch"], history["cvar_val"],   label="CVaR₉₅ (test bootstrap)",  color="#D32F2F")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CVaR₉₅ (higher = better)")
    ax.set_title("Sprint 2: VIX-as-vega, weekly rebalancing -- learning curve")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_cvar(res, path):
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
           edgecolor="black", linewidth=0.5, label="BS delta (weekly)")
    ax.bar(x + w/2, c_dh, w, yerr=err_dh, capsize=4, color="#4CAF50",
           edgecolor="black", linewidth=0.5, label="Deep (SPY+VIX, weekly)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("CVaR₉₅ (higher = better)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Sprint 2: VIX-as-vega vs Black-Scholes delta (weekly rebalance)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=256)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save",   default="results/vix_futures.pth")
    args = ap.parse_args()

    tr_s, tr_v, te_s, te_v = load_spy_vix()
    print(f"Train weekly obs: {len(tr_s)}")
    print(f"Test  weekly obs: {len(te_s)}")
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)

    policy, hist = train(s_tr, s_te, args.epochs, args.N, args.lam,
                         args.lr, args.seed)
    torch.save(policy.state_dict(), args.save)
    print(f"Saved -> {args.save}")

    print("\n=== Evaluation ===")
    res = evaluate(policy)

    write_metrics(res, hist, "results/vix_futures_metrics.txt")
    plot_learning(hist, "results/vix_futures_learning.png")
    plot_cvar(res, "results/vix_futures_cvar.png")
    print("\nArtifacts in results/vix_futures_*.{txt,png}")


if __name__ == "__main__":
    main()
