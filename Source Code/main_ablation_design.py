"""
main_ablation_design.py
------------------------
Week 7 Ablation B: Design-choice ablation.

Defends the core architectural decisions of the single-asset VS hedger
(ATM call, Heston, CVaR loss). Three sub-ablations:

  1. STATE FEATURES   : {S}, {S, VS}, {S, VS, t}, {S, VS, t, prev_pos}
  2. NETWORK DEPTH    : 1, 2, 3, 4 hidden layers (64 units each)
  3. REBALANCING FREQ : T in {10, 20, 30, 60} steps per episode

All runs use the same cost_rate=0.0002, alpha-anneal 0.80->0.95, 300 epochs,
batch 256 for speed. All evaluated on 3000 fresh Heston paths.

Outputs:
  results/ablation_design_state.png
  results/ablation_design_depth.png
  results/ablation_design_freq.png
  results/ablation_design_table.txt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from market.heston_with_var_swap import simulate_heston_with_var_swap
from risk.cvar import cvar as cvar_torch

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

COST_RATE = 0.0002
S0  = 100.0
VS0 = 0.04 * S0 / 0.30      # ≈ 13.33


def cvar_np(pnl, alpha=0.95):
    k = max(int((1 - alpha) * len(pnl)), 1)
    return float(np.sort(pnl)[:k].mean())


# ---------------------------------------------------------------------- #
# Flexible policy builder                                               #
# ---------------------------------------------------------------------- #
def build_mlp(in_dim, depth, hidden=64, out_dim=2):
    """depth = number of hidden ReLU layers."""
    layers = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*layers)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net.to(device)


# ---------------------------------------------------------------------- #
# Rollout -- configurable state                                          #
# ---------------------------------------------------------------------- #
def rollout(net, S, VS, payoff_fn, features, cost_rate=COST_RATE):
    """
    features : subset of {"S", "VS", "t", "prev"}.
    Always hedges with S (and VS if available); output dim matches # instruments.
    """
    N, Tp1 = S.shape
    T = Tp1 - 1
    pnl = torch.zeros(N, device=S.device)
    prev_dS = torch.zeros(N, device=S.device)
    prev_dV = torch.zeros(N, device=S.device)
    use_vs  = "VS" in features
    use_t   = "t"  in features
    use_pr  = "prev" in features

    for t in range(T):
        cols = [S[:, t] / S0]
        if use_vs: cols.append(VS[:, t] / VS0)
        if use_t:  cols.append(torch.full((N,), t / T, device=S.device))
        if use_pr:
            cols.append(prev_dS)
            if use_vs: cols.append(prev_dV)
        state = torch.stack(cols, dim=1)
        act = torch.tanh(net(state)) * 5.0
        dS = act[:, 0]
        dV = act[:, 1] if use_vs else torch.zeros_like(dS)

        pnl = pnl + prev_dS * (S[:, t + 1] - S[:, t])
        if use_vs:
            pnl = pnl + prev_dV * (VS[:, t + 1] - VS[:, t])
        pnl = pnl - cost_rate * torch.abs(dS - prev_dS) * (S[:, t] / S0)
        if use_vs:
            pnl = pnl - cost_rate * torch.abs(dV - prev_dV) * (VS[:, t] / VS0)
        prev_dS, prev_dV = dS, dV

    pnl = pnl - payoff_fn(S[:, -1])
    return pnl


def train(features, depth, T_steps, epochs, batch):
    in_dim  = sum([1, "VS" in features, "t" in features,
                   ("prev" in features) * (1 + ("VS" in features))])
    out_dim = 2 if "VS" in features else 1
    net = build_mlp(in_dim, depth, hidden=64, out_dim=out_dim)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    for epoch in range(epochs):
        a = 0.80 + 0.15 * epoch / max(epochs - 1, 1)
        S, VS, pf = simulate_heston_with_var_swap(N=batch, T=T_steps, device=device)
        # out_dim=1 case needs a placeholder 2nd output channel during rollout
        if out_dim == 1:
            class _Wrap(nn.Module):
                def __init__(self, core):
                    super().__init__()
                    self.core = core
                def forward(self, x):
                    z = self.core(x)
                    return torch.cat([z, torch.zeros_like(z)], dim=-1)
            eff = _Wrap(net)
        else:
            eff = net
        pnl = rollout(eff, S, VS, pf, features)
        loss = -cvar_torch(pnl, a)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
    net.eval()
    return net, out_dim


def evaluate(net, out_dim, features, T_steps, n_test=3000):
    torch.manual_seed(0)
    S, VS, pf = simulate_heston_with_var_swap(N=n_test, T=T_steps, device=device)
    if out_dim == 1:
        class _Wrap(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
            def forward(self, x):
                z = self.core(x)
                return torch.cat([z, torch.zeros_like(z)], dim=-1)
        eff = _Wrap(net)
    else:
        eff = net
    with torch.no_grad():
        pnl = rollout(eff, S, VS, pf, features).cpu().numpy()
    return dict(mean=float(pnl.mean()), std=float(pnl.std()),
                cvar95=cvar_np(pnl, 0.95), cvar99=cvar_np(pnl, 0.99))


# ---------------------------------------------------------------------- #
# Main                                                                  #
# ---------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--n_test", type=int, default=3000)
    p.add_argument("--smoke",  action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.batch, args.n_test = 40, 128, 800

    table_rows = []

    # --- 1. State features --- #
    print("\n=== Ablation 1: state features ===")
    state_configs = [
        ("S",          {"S"}),
        ("S+VS",       {"S", "VS"}),
        ("S+VS+t",     {"S", "VS", "t"}),
        ("S+VS+t+prev",{"S", "VS", "t", "prev"}),
    ]
    state_res = {}
    for name, feats in state_configs:
        net, od = train(feats, depth=2, T_steps=30,
                        epochs=args.epochs, batch=args.batch)
        r = evaluate(net, od, feats, T_steps=30, n_test=args.n_test)
        state_res[name] = r
        print(f"  {name:<13} mean={r['mean']:+.3f}  std={r['std']:.3f}  "
              f"CVaR95={r['cvar95']:+.3f}")
        table_rows.append(("state", name, r))

    # --- 2. Network depth --- #
    print("\n=== Ablation 2: network depth (S+VS+t+prev) ===")
    depth_res = {}
    for d in [1, 2, 3, 4]:
        net, od = train({"S", "VS", "t", "prev"}, depth=d, T_steps=30,
                        epochs=args.epochs, batch=args.batch)
        r = evaluate(net, od, {"S", "VS", "t", "prev"},
                     T_steps=30, n_test=args.n_test)
        depth_res[d] = r
        print(f"  depth={d}     mean={r['mean']:+.3f}  std={r['std']:.3f}  "
              f"CVaR95={r['cvar95']:+.3f}")
        table_rows.append(("depth", f"d={d}", r))

    # --- 3. Rebalancing frequency --- #
    print("\n=== Ablation 3: rebalancing frequency T ===")
    freq_res = {}
    for T_ in [10, 20, 30, 60]:
        net, od = train({"S", "VS", "t", "prev"}, depth=2, T_steps=T_,
                        epochs=args.epochs, batch=args.batch)
        r = evaluate(net, od, {"S", "VS", "t", "prev"},
                     T_steps=T_, n_test=args.n_test)
        freq_res[T_] = r
        print(f"  T={T_:<3d}     mean={r['mean']:+.3f}  std={r['std']:.3f}  "
              f"CVaR95={r['cvar95']:+.3f}")
        table_rows.append(("freq", f"T={T_}", r))

    # --- Plots --- #
    def bar_plot(results, xlabel, filename, title):
        names = list(results.keys())
        cv = [results[n]['cvar95'] for n in names]
        plt.figure(figsize=(7, 4.2))
        plt.bar(range(len(names)), cv, color="#FF5722",
                edgecolor="black", linewidth=0.5)
        for i, v in enumerate(cv):
            plt.text(i, v - 0.3, f"{v:.2f}", ha="center", fontsize=9,
                     fontweight="bold")
        plt.xticks(range(len(names)), [str(n) for n in names])
        plt.xlabel(xlabel)
        plt.ylabel("CVaR₉₅ (higher = better)")
        plt.title(title)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

    bar_plot(state_res, "State features",
             "results/ablation_design_state.png",
             "Ablation: state features (depth=2, T=30)")
    bar_plot(depth_res, "Hidden layers",
             "results/ablation_design_depth.png",
             "Ablation: network depth (S+VS+t+prev, T=30)")
    bar_plot(freq_res,  "Rebalancing steps T",
             "results/ablation_design_freq.png",
             "Ablation: rebalancing frequency (S+VS+t+prev, depth=2)")

    # --- Table --- #
    with open("results/ablation_design_table.txt", "w") as f:
        f.write("Week 7 -- Design-choice ablation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Epochs per run: {args.epochs}, batch: {args.batch}, "
                f"N_test: {args.n_test}\n\n")
        f.write(f"{'group':>8} {'config':>15} {'mean':>10} "
                f"{'std':>10} {'CVaR95':>10} {'CVaR99':>10}\n")
        f.write("-" * 68 + "\n")
        for g, n, r in table_rows:
            f.write(f"{g:>8} {n:>15} {r['mean']:>+10.3f} {r['std']:>10.3f} "
                    f"{r['cvar95']:>+10.3f} {r['cvar99']:>+10.3f}\n")

    print("\nSaved:")
    print("  results/ablation_design_state.png")
    print("  results/ablation_design_depth.png")
    print("  results/ablation_design_freq.png")
    print("  results/ablation_design_table.txt")


if __name__ == "__main__":
    main()
