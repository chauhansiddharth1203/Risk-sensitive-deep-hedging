"""
main_vix_futures_v2.py
-----------------------
Sprint 2b: VIX-as-vega with tightened action scales.

Changes vs v1:
  - Stock position clamped to [-2, +2] (was [-5, +5])
  - VIX   position clamped to [-0.3, +0.3] (was [-5, +5])
    Rationale: VIX per-week moves are ~20x SPY moves in %, so position
    size must scale inversely. 0.3 x ~300% crisis VIX move ≈ same $ P&L
    as 2 x ~15% SPY move.
  - Lower learning rate 1e-4 (was 3e-4) to stabilise VIX heavy-tail
    gradients.
  - Larger batch 512 (was 256).
  - 800 epochs (was 400) to give the tighter search space time to
    converge.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from policy.network_vix import VIXHedgingPolicy
from market.vix_bootstrap import (
    VIXBootstrap, payoff_call_atm, S0, K, T,
)
from data.vix_windows import load as load_spy_vix
from utils.bootstrap_ci import bootstrap_cvar_ci, bootstrap_diff_ci
from main_backtest_training import cvar_loss, BS_SIGMA, COST_RATE
from main_vix_futures import (
    download_spy_vix, build_weekly_windows, bs_pnl_weekly,
    write_metrics, plot_learning, plot_cvar,
)

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def rollout_batch_vix(policy, S, VS, payoff_fn):
    """Tighter-action rollout specialised for VIXHedgingPolicy."""
    N, Tp1 = S.shape
    Tn = Tp1 - 1
    pnl = torch.zeros(N, device=S.device)
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
        dS = torch.tanh(a[:, 0]) * policy.stock_scale
        dV = torch.tanh(a[:, 1]) * policy.vix_scale

        gain_S = prev_S * (S[:, t + 1] - S[:, t])
        gain_V = prev_V * (VS[:, t + 1] - VS[:, t])
        tc = (policy.cost_rate * torch.abs(dS - prev_S) * (S[:, t] / policy.S0)
            + policy.cost_rate * torch.abs(dV - prev_V) * (VS[:, t] / policy.VS0))
        pnl = pnl + gain_S + gain_V - tc
        prev_S, prev_V = dS, dV
    pnl = pnl - payoff_fn(S[:, -1])
    return pnl


def train(s_tr, s_te, epochs, N, lam, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = VIXHedgingPolicy().to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    hist = {"epoch": [], "loss": [], "cvar_train": [], "cvar_val": [],
            "mean_train": []}
    for ep in range(epochs):
        alpha = 0.80 + 0.15 * min(ep / max(epochs - 1, 1), 1.0)
        S, VS = s_tr.sample_batch(N, device=device)
        pnl = rollout_batch_vix(policy, S, VS, payoff_call_atm)
        c = cvar_loss(pnl, alpha)
        loss = c + lam * torch.abs(pnl.mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            with torch.no_grad():
                Sv, VSv = s_te.sample_batch(512, device=device)
                pv = rollout_batch_vix(policy, Sv, VSv, payoff_call_atm)
                cv = cvar_loss(pv, 0.95).item()
            hist["epoch"].append(ep + 1)
            hist["loss"].append(loss.item())
            hist["cvar_train"].append(-c.item())
            hist["cvar_val"].append(-cv)
            hist["mean_train"].append(pnl.mean().item())
            print(f"ep {ep+1:4d}  loss={loss.item():+.3f}  "
                  f"CVaR_train={-c.item():+.3f}  "
                  f"mean={pnl.mean().item():+.3f}  "
                  f"CVaR_val={-cv:+.3f}")
    return policy, hist


def deep_pnl_vix(policy, Sw, VSw):
    S_t  = torch.tensor(Sw,  dtype=torch.float32, device=device)
    VS_t = torch.tensor(VSw, dtype=torch.float32, device=device)
    with torch.no_grad():
        p = rollout_batch_vix(policy, S_t, VS_t, payoff_call_atm)
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
    for L, (s, e) in periods.items():
        df = download_spy_vix(s, e)
        if len(df) < T + 2:
            continue
        Sw, VSw = build_weekly_windows(df)
        p_bs = bs_pnl_weekly(Sw)
        p_dh = deep_pnl_vix(policy, Sw, VSw)
        c_bs, lo_bs, hi_bs = bootstrap_cvar_ci(p_bs, 0.95, B=500)
        c_dh, lo_dh, hi_dh = bootstrap_cvar_ci(p_dh, 0.95, B=500)
        d, dlo, dhi = bootstrap_diff_ci(p_dh, p_bs, 0.95, B=500)
        res[L] = dict(n=len(Sw),
                      bs=(c_bs, lo_bs, hi_bs), dh=(c_dh, lo_dh, hi_dh),
                      diff=(d, dlo, dhi),
                      std_bs=float(p_bs.std()), std_dh=float(p_dh.std()))
        print(f"{L:<32} n={len(Sw):3d}  "
              f"BS={c_bs:+7.2f}[{lo_bs:+6.2f},{hi_bs:+6.2f}]  "
              f"DH={c_dh:+7.2f}[{lo_dh:+6.2f},{hi_dh:+6.2f}]  "
              f"Delta={d:+6.2f}[{dlo:+6.2f},{dhi:+6.2f}]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--N",      type=int, default=512)
    ap.add_argument("--lam",    type=float, default=0.5)
    ap.add_argument("--lr",     type=float, default=1e-4)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--save",   default="results/vix_futures_v2.pth")
    args = ap.parse_args()

    tr_s, tr_v, te_s, te_v = load_spy_vix()
    s_tr = VIXBootstrap(tr_s, tr_v)
    s_te = VIXBootstrap(te_s, te_v)
    print(f"Train weekly obs: {len(tr_s)}, Test: {len(te_s)}")
    print(f"Action scales: stock +/-{2.0}, VIX +/-{0.3}")
    print(f"lr={args.lr}  N={args.N}  epochs={args.epochs}")

    policy, hist = train(s_tr, s_te, args.epochs, args.N, args.lam,
                         args.lr, args.seed)
    torch.save(policy.state_dict(), args.save)
    print(f"Saved -> {args.save}")

    print("\n=== Evaluation ===")
    res = evaluate(policy)

    write_metrics(res, hist, "results/vix_futures_v2_metrics.txt")
    plot_learning(hist, "results/vix_futures_v2_learning.png")
    plot_cvar(res, "results/vix_futures_v2_cvar.png")
    print("\nArtifacts in results/vix_futures_v2_*.{txt,png}")


if __name__ == "__main__":
    main()
