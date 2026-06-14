"""
main_historical_stress.py
--------------------------
Week 7 Experiment: Historical stress test on real SPX data.

Motivation:
    All prior CVaR claims rest on Heston-simulated paths. An examiner
    will legitimately ask whether the policy generalises to REAL crisis
    periods. We test three trained policies on rolling 30-trading-day
    windows during the 2008 GFC and the 2020 COVID crash -- regimes the
    training distribution has never seen.

Data:
    yfinance SPY (SPDR S&P 500 ETF) adjusted close, daily.
      2008 window : 2007-09-01  to  2009-06-30
      2020 window : 2020-01-01  to  2020-06-30
      Normal ref  : 2017-01-01  to  2017-12-31   (calm year)

Setup:
    At each anchor day t0, grab the next 30 trading days as a "path".
    Rescale so S[0] = 100 (strike K = 100, ATM).
    Build a VS proxy from a 10-day EWMA of squared log-returns
        v_t_hat = EWMA_var(returns)         (annualised x 252)
        VS_t    = v_t_hat * S0 / 0.30       (same scaling as training)

Strategies:
    (1) BS delta hedge        (constant σ=0.20)
    (2) Deep hedger  (S+VS, trained on Heston with randomised params)
        loaded from results/varswap_cvar_robust.pth
    (3) Unhedged (pure option short)

Outputs:
    results/historical_stress_cvar.png
    results/historical_stress_pnl.png
    results/historical_stress_metrics.txt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import yfinance as yf
from scipy.stats import norm

from policy.network_varswap import HedgingPolicyVarSwap
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -- Parameters matching training --------------------------------------- #
T_WINDOW    = 30
COST_RATE   = 0.0002
S0          = 100.0
K           = 100.0
SIGMA_V_FIXED = 0.30
BS_SIGMA    = 0.20
EWMA_LAMBDA = 0.94


def download_prices(start, end):
    df = yf.download("SPY", start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No SPY data between {start} and {end}")
    close = df["Close"].values.ravel()
    return close


def ewma_variance(returns, lam=EWMA_LAMBDA):
    """RiskMetrics-style EWMA variance; returns *annualised* variance."""
    v = np.zeros(len(returns))
    v[0] = returns[0] ** 2
    for i in range(1, len(returns)):
        v[i] = lam * v[i - 1] + (1 - lam) * returns[i] ** 2
    return v * 252.0


def build_windows(prices, T=T_WINDOW, step=5):
    """
    Rolling windows of length T+1, stepping `step` days each time
    (overlapping). Each window normalised so S[0]=100.
    step=5 (weekly) gives many quasi-independent episodes while limiting
    overlap to <1 week of shared returns.
    """
    rets = np.diff(np.log(prices))
    v_hat = ewma_variance(rets)
    v_aligned = np.concatenate([[v_hat[0]], v_hat])

    windows_S, windows_VS = [], []
    for i in range(0, len(prices) - T - 1, step):
        S_ = prices[i:i + T + 1]
        v_ = v_aligned[i:i + T + 1]
        S_norm = 100.0 * S_ / S_[0]
        VS_    = v_ * S0 / SIGMA_V_FIXED
        windows_S.append(S_norm)
        windows_VS.append(VS_)
    return np.array(windows_S), np.array(windows_VS)


def bs_delta(S, K, tau, sigma):
    tau = max(tau, 1e-6)
    d1 = (np.log(np.maximum(S, 1e-8) / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bs_pnl_per_window(S_windows):
    """BS delta hedge on each window.  Returns (N,) P&L."""
    N, Tp1 = S_windows.shape
    T = Tp1 - 1
    pnl = np.zeros(N)
    for i in range(N):
        prev_d = 0.0
        for t in range(T):
            tau = (T - t) / T
            d = bs_delta(S_windows[i, t], K, tau, BS_SIGMA)
            pnl[i] += prev_d * (S_windows[i, t + 1] - S_windows[i, t])
            pnl[i] -= COST_RATE * abs(d - prev_d) * (S_windows[i, t] / S0)
            prev_d = d
        pnl[i] -= max(S_windows[i, -1] - K, 0.0)
    return pnl


def unhedged_pnl(S_windows):
    return -np.maximum(S_windows[:, -1] - K, 0.0)


def deep_hedger_pnl(policy, S_windows, VS_windows):
    N, Tp1 = S_windows.shape
    pnl = np.zeros(N)
    for i in range(N):
        S_t  = torch.tensor(S_windows[i],  dtype=torch.float32, device=device)
        VS_t = torch.tensor(VS_windows[i], dtype=torch.float32, device=device)
        def payoff_fn(S_T):
            return torch.clamp(S_T - K, min=0.0)
        with torch.no_grad():
            p = policy.rollout(S_t, VS_t, payoff_fn)
        pnl[i] = p.item()
    return pnl


def cvar_np(pnl, alpha=0.95):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", default="results/varswap_cvar_robust.pth",
                    help="Deep hedger .pth to evaluate")
    args = ap.parse_args()

    if not os.path.exists(args.policy_path):
        # fall back to vanilla varswap_cvar.pth
        args.policy_path = "results/varswap_cvar.pth"
        print(f"[info] robust policy not found, falling back to {args.policy_path}")

    print(f"Loading policy from {args.policy_path}")
    policy = HedgingPolicyVarSwap(cost_rate=COST_RATE).to(device)
    policy.load_state_dict(torch.load(args.policy_path, map_location=device))
    policy.eval()

    periods = {
        "2008 GFC":      ("2007-09-01", "2009-06-30"),
        "2020 COVID":    ("2019-11-01", "2020-12-31"),
        "2017 Calm":     ("2016-07-01", "2017-12-31"),
    }

    all_res = {}
    for label, (start, end) in periods.items():
        print(f"\n=== {label}  ({start} -> {end}) ===")
        prices = download_prices(start, end)
        print(f"  {len(prices)} trading days")
        S_w, VS_w = build_windows(prices)
        print(f"  {len(S_w)} rolling 30-day windows (step=5)")

        pnl_bs = bs_pnl_per_window(S_w)
        pnl_un = unhedged_pnl(S_w)
        pnl_dh = deep_hedger_pnl(policy, S_w, VS_w)

        def stats(x):
            c, lo, hi = bootstrap_cvar_ci(x, 0.95, B=500)
            return dict(mean=float(x.mean()), std=float(x.std()),
                        cvar=c, cvar_lo=lo, cvar_hi=hi)

        all_res[label] = {
            "BS":       stats(pnl_bs),
            "Unhedged": stats(pnl_un),
            "DeepHedge": stats(pnl_dh),
            "pnl_bs":   pnl_bs, "pnl_un": pnl_un, "pnl_dh": pnl_dh,
            "n_win":    len(S_w),
        }

        for k in ["Unhedged", "BS", "DeepHedge"]:
            s = all_res[label][k]
            print(f"  {k:<10}  mean={s['mean']:+7.2f}  std={s['std']:5.2f}  "
                  f"CVaR95={s['cvar']:+7.2f}  "
                  f"[{s['cvar_lo']:+6.2f}, {s['cvar_hi']:+6.2f}]")

    # -- bar chart: CVaR by period with CIs -- #
    labels = list(all_res.keys())
    methods = ["Unhedged", "BS", "DeepHedge"]
    colours = {"Unhedged": "#9E9E9E", "BS": "#2196F3", "DeepHedge": "#FF5722"}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    w = 0.27
    for j, m in enumerate(methods):
        cv    = [all_res[L][m]['cvar']    for L in labels]
        cv_lo = [all_res[L][m]['cvar_lo'] for L in labels]
        cv_hi = [all_res[L][m]['cvar_hi'] for L in labels]
        err = [[cv[i] - cv_lo[i] for i in range(len(cv))],
               [cv_hi[i] - cv[i] for i in range(len(cv))]]
        ax.bar(x + (j - 1) * w, cv, width=w, color=colours[m],
               yerr=err, capsize=4, edgecolor="black", linewidth=0.5,
               label=m)
        for i, v in enumerate(cv):
            ax.text(i + (j - 1) * w, v - 0.5, f"{v:.1f}",
                    ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}\n(n={all_res[L]['n_win']})" for L in labels])
    ax.set_ylabel("CVaR₉₅ (higher = better)")
    ax.set_title("Historical stress test: CVaR₉₅ on real SPX windows\n"
                 "Deep hedger trained only on simulated Heston, never on real data")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/historical_stress_cvar.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # -- P&L histograms per period -- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, L in zip(axes, labels):
        r = all_res[L]
        all_pnl = np.concatenate([r["pnl_un"], r["pnl_bs"], r["pnl_dh"]])
        lo, hi = all_pnl.min() - 1, all_pnl.max() + 1
        bins = np.linspace(lo, hi, 30)
        ax.hist(r["pnl_un"], bins=bins, alpha=0.40, color="#9E9E9E",
                label=f"Unhedged (CVaR={r['Unhedged']['cvar']:.1f})")
        ax.hist(r["pnl_bs"], bins=bins, alpha=0.55, color="#2196F3",
                label=f"BS (CVaR={r['BS']['cvar']:.1f})")
        ax.hist(r["pnl_dh"], bins=bins, alpha=0.55, color="#FF5722",
                label=f"DeepHedge (CVaR={r['DeepHedge']['cvar']:.1f})")
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_title(L)
        ax.set_xlabel("P&L"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
    plt.suptitle("Terminal P&L distributions on historical SPX windows")
    plt.tight_layout()
    plt.savefig("results/historical_stress_pnl.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # -- metrics -- #
    with open("results/historical_stress_metrics.txt", "w") as f:
        f.write("Week 7 -- Historical stress test on SPX\n")
        f.write("=" * 60 + "\n")
        f.write(f"Policy: {args.policy_path}\n\n")
        for L in labels:
            r = all_res[L]
            f.write(f"{L}  (n={r['n_win']} windows)\n")
            for k in ["Unhedged", "BS", "DeepHedge"]:
                s = r[k]
                f.write(f"  {k:<10}  mean={s['mean']:+7.3f}  "
                        f"std={s['std']:6.3f}  CVaR95={s['cvar']:+7.3f}  "
                        f"[{s['cvar_lo']:+6.3f}, {s['cvar_hi']:+6.3f}]\n")
            # paired diff: DeepHedge vs BS
            d, lo, hi = bootstrap_diff_ci(r["pnl_dh"], r["pnl_bs"],
                                          0.95, B=500)
            f.write(f"  Delta(Deep-BS) CVaR = {d:+.3f}  "
                    f"[{lo:+.3f}, {hi:+.3f}]\n\n")

    print("\nSaved:")
    print("  results/historical_stress_cvar.png")
    print("  results/historical_stress_pnl.png")
    print("  results/historical_stress_metrics.txt")


if __name__ == "__main__":
    main()
