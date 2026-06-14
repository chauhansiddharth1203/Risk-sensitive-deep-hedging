"""
main_backtest_training.py
--------------------------
Week 9: Train the deep hedger directly on resampled real SPY returns
(stationary block bootstrap), no simulated paths. The goal is to
convert the Week 8 sim-to-real negative into a second positive result
by eliminating the simulator distribution shift entirely.

Design:
    - Train pool : 2005-2017 daily SPY returns
    - Test pool  : 2018-2024 daily SPY returns (held-out, never seen)
    - Sampler    : stationary block bootstrap (Politis-Romano 1994)
                   preserves volatility clustering
    - Policy     : unchanged HedgingPolicyVarSwap (S + VS proxy)
    - Loss       : CVaR_alpha + lambda|E[Π]|, alpha annealed 0.80 -> 0.95
    - Eval       : same 2008/2020/2017 SPY windows as Week 7-8, plus
                   held-out 2018 vol-mageddon and 2022 rate shock

Usage:
    python main_backtest_training.py --epochs 400 --lam 0.5 --N 256
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from scipy.stats import norm

from policy.network_varswap import HedgingPolicyVarSwap
from market.historical_bootstrap import (
    get_train_sampler, get_test_sampler, payoff_call_atm,
    T, S0, K, SIGMA_V_FIXED, EWMA_LAMBDA, ewma_variance,
)
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci

import yfinance as yf

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

COST_RATE = 0.0002
BS_SIGMA  = 0.20


# --- CVaR loss ---------------------------------------------------------- #

def cvar_loss(pnl, alpha):
    k = max(int((1.0 - alpha) * pnl.shape[0]), 1)
    sorted_p, _ = torch.sort(pnl)
    return -sorted_p[:k].mean()   # negate because loss is minimised


# --- Vectorised rollout across a batch ---------------------------------- #

def rollout_batch(policy, S, VS, payoff_fn):
    """S, VS: (N, T+1). Returns pnl (N,) with grad."""
    N, Tp1 = S.shape
    Tn = Tp1 - 1
    pnl  = torch.zeros(N, device=S.device)
    prev_S = torch.zeros(N, device=S.device)
    prev_V = torch.zeros(N, device=S.device)
    for t in range(Tn):
        state = torch.stack([
            S[:, t]  / policy.S0,
            VS[:, t] / policy.VS0,
            torch.full((N,), t / Tn, device=S.device),
            prev_S,
            prev_V,
        ], dim=1)
        a = policy.forward(state)
        dS = torch.tanh(a[:, 0]) * 5.0
        dV = torch.tanh(a[:, 1]) * 5.0

        gain_S = prev_S * (S[:, t + 1] - S[:, t])
        gain_V = prev_V * (VS[:, t + 1] - VS[:, t])
        tc = (policy.cost_rate * torch.abs(dS - prev_S) * (S[:, t] / policy.S0)
            + policy.cost_rate * torch.abs(dV - prev_V) * (VS[:, t] / policy.VS0))
        pnl = pnl + gain_S + gain_V - tc
        prev_S, prev_V = dS, dV
    pnl = pnl - payoff_fn(S[:, -1])
    return pnl


# --- Training loop ------------------------------------------------------ #

def train(epochs=400, N=256, lam=0.5, lr=3e-4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)

    sampler = get_train_sampler()
    test_sampler = get_test_sampler()

    policy = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    history = {"epoch": [], "loss": [], "cvar_train": [], "mean_train": [],
               "cvar_val": []}

    for ep in range(epochs):
        alpha = 0.80 + 0.15 * min(ep / max(epochs - 1, 1), 1.0)
        S, VS = sampler.sample_batch(N, device=device)
        pnl = rollout_batch(policy, S, VS, payoff_call_atm)

        cvar = cvar_loss(pnl, alpha)
        mean_pen = lam * torch.abs(pnl.mean())
        loss = cvar + mean_pen

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                Sv, VSv = test_sampler.sample_batch(512, device=device)
                pv = rollout_batch(policy, Sv, VSv, payoff_call_atm)
                c_val = cvar_loss(pv, 0.95).item()
            history["epoch"].append(ep + 1)
            history["loss"].append(loss.item())
            history["cvar_train"].append(-cvar.item())
            history["mean_train"].append(pnl.mean().item())
            history["cvar_val"].append(-c_val)
            print(f"ep {ep+1:4d}  loss={loss.item():+.3f}  "
                  f"CVaR_train={-cvar.item():+.3f}  "
                  f"mean_train={pnl.mean().item():+.3f}  "
                  f"CVaR_val(bootstrap)={-c_val:+.3f}")

    return policy, history


# --- Evaluation on historical windows (reuse Week 7 pipeline) ----------- #

def download_prices(start, end):
    df = yf.download("SPY", start=start, end=end,
                     auto_adjust=True, progress=False)
    return df["Close"].values.ravel()


def build_windows(prices, T_win=T, step=5):
    rets = np.diff(np.log(prices))
    v_hat = ewma_variance(rets)
    v_aligned = np.concatenate([[v_hat[0]], v_hat])
    Sw, VSw = [], []
    for i in range(0, len(prices) - T_win - 1, step):
        S_ = prices[i:i + T_win + 1]
        v_ = v_aligned[i:i + T_win + 1]
        Sw.append(100.0 * S_ / S_[0])
        VSw.append(v_ * S0 / SIGMA_V_FIXED)
    return np.array(Sw), np.array(VSw)


def bs_delta(S, K, tau, sigma):
    tau = max(tau, 1e-6)
    d1 = (np.log(np.maximum(S, 1e-8) / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bs_pnl_per_window(Sw):
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


def deep_hedger_pnl(policy, Sw, VSw):
    N = Sw.shape[0]
    S_t  = torch.tensor(Sw,  dtype=torch.float32, device=device)
    VS_t = torch.tensor(VSw, dtype=torch.float32, device=device)
    with torch.no_grad():
        p = rollout_batch(policy, S_t, VS_t, payoff_call_atm)
    return p.cpu().numpy()


def evaluate(policy):
    periods = {
        "2008 GFC (in-train)":  ("2007-09-01", "2009-06-30"),
        "2017 Calm (in-train)": ("2016-07-01", "2017-12-31"),
        "2018 Volmageddon":     ("2018-01-01", "2018-12-31"),
        "2020 COVID (OOT)":     ("2019-11-01", "2020-12-31"),
        "2022 Rate shock":      ("2022-01-01", "2022-12-31"),
    }
    res = {}
    for label, (start, end) in periods.items():
        px = download_prices(start, end)
        Sw, VSw = build_windows(px)
        p_bs = bs_pnl_per_window(Sw)
        p_dh = deep_hedger_pnl(policy, Sw, VSw)
        c_bs, lo_bs, hi_bs = bootstrap_cvar_ci(p_bs, 0.95, B=500)
        c_dh, lo_dh, hi_dh = bootstrap_cvar_ci(p_dh, 0.95, B=500)
        d, dlo, dhi = bootstrap_diff_ci(p_dh, p_bs, 0.95, B=500)
        res[label] = dict(n=len(Sw),
                          bs=(c_bs, lo_bs, hi_bs),
                          dh=(c_dh, lo_dh, hi_dh),
                          diff=(d, dlo, dhi),
                          std_bs=float(p_bs.std()),
                          std_dh=float(p_dh.std()))
        print(f"{label:<25}  n={len(Sw):3d}  "
              f"BS CVaR={c_bs:+7.2f} [{lo_bs:+6.2f},{hi_bs:+6.2f}]  "
              f"Deep CVaR={c_dh:+7.2f} [{lo_dh:+6.2f},{hi_dh:+6.2f}]  "
              f"Delta={d:+6.2f} [{dlo:+6.2f},{dhi:+6.2f}]")
    return res


def write_metrics(res, history, path="results/backtest_training_metrics.txt"):
    with open(path, "w") as f:
        f.write("Week 9 -- Backtest-trained deep hedger vs BS delta\n")
        f.write("=" * 72 + "\n\n")
        f.write("Training: stationary block bootstrap on SPY 2005-2017\n")
        f.write("Testing : same rolling windows as Week 7/8\n\n")
        for L, r in res.items():
            c_bs, lo_bs, hi_bs = r["bs"]
            c_dh, lo_dh, hi_dh = r["dh"]
            d, dlo, dhi = r["diff"]
            f.write(f"{L}  (n={r['n']} windows)\n")
            f.write(f"  BS    CVaR95 = {c_bs:+7.3f}  [{lo_bs:+.3f}, {hi_bs:+.3f}]  std={r['std_bs']:.3f}\n")
            f.write(f"  Deep  CVaR95 = {c_dh:+7.3f}  [{lo_dh:+.3f}, {hi_dh:+.3f}]  std={r['std_dh']:.3f}\n")
            f.write(f"  Delta(Deep-BS)   = {d:+7.3f}  [{dlo:+.3f}, {dhi:+.3f}]\n\n")


def plot_learning(history, path="results/backtest_training_learning.png"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["cvar_train"], label="CVaR₉₅ (train bootstrap)", color="#1976D2")
    ax.plot(history["epoch"], history["cvar_val"],   label="CVaR₉₅ (test bootstrap)",  color="#D32F2F")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CVaR₉₅ (higher = better)")
    ax.set_title("Backtest-trained deep hedger -- learning curve")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_cvar_bars(res, path="results/backtest_training_cvar.png"):
    labels = list(res.keys())
    c_bs = [res[L]["bs"][0] for L in labels]
    c_dh = [res[L]["dh"][0] for L in labels]
    err_bs = [[c_bs[i] - res[L]["bs"][1] for i, L in enumerate(labels)],
              [res[L]["bs"][2] - c_bs[i] for i, L in enumerate(labels)]]
    err_dh = [[c_dh[i] - res[L]["dh"][1] for i, L in enumerate(labels)],
              [res[L]["dh"][2] - c_dh[i] for i, L in enumerate(labels)]]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, c_bs, w, yerr=err_bs, capsize=4, color="#2196F3",
           edgecolor="black", linewidth=0.5, label="BS delta")
    ax.bar(x + w/2, c_dh, w, yerr=err_dh, capsize=4, color="#FF5722",
           edgecolor="black", linewidth=0.5, label="Deep (backtest-trained)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("CVaR₉₅ (higher = better)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Week 9: backtest-trained deep hedger vs BS delta on SPY windows")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# --- Main --------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--N",      type=int, default=256)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save",   default="results/backtest_trained.pth")
    args = ap.parse_args()

    print(f"device={device}  epochs={args.epochs}  N={args.N}  lam={args.lam}")
    policy, history = train(args.epochs, args.N, args.lam, args.lr, args.seed)
    torch.save(policy.state_dict(), args.save)
    print(f"Saved policy -> {args.save}")

    print("\n=== Evaluation on real SPY windows ===")
    res = evaluate(policy)

    write_metrics(res, history)
    plot_learning(history)
    plot_cvar_bars(res)
    print("\nArtifacts:")
    print("  results/backtest_training_metrics.txt")
    print("  results/backtest_training_learning.png")
    print("  results/backtest_training_cvar.png")


if __name__ == "__main__":
    main()
