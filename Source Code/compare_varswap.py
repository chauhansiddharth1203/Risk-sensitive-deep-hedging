"""
compare_varswap.py
------------------
Load all trained models (stock-only and stock+variance-swap) and produce
a side-by-side comparison of CVaR, Mean PnL, and Std PnL.

Also generates:
  1. Bar chart: CVaR improvement over delta hedge for all 6 deep hedge configs
  2. PnL distribution overlay for CVaR objective (stock-only vs stock+varswap)

Run AFTER training all 6 models:
    python main.py
    python main_variance.py
    python main_entropic.py
    python main_varswap_cvar.py
    python main_varswap_variance.py
    python main_varswap_entropic.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from policy.network import HedgingPolicy
from policy.network_varswap import HedgingPolicyVarSwap
from evaluation.evaluate import evaluate_policy
from evaluation.evaluate_varswap import evaluate_varswap_policy
from risk.cvar import cvar

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.95


def load_stock_only(path, cost_rate=0.001):
    policy = HedgingPolicy(state_dim=3, action_dim=1, cost_rate=cost_rate).to(device)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()
    return policy


def load_varswap(path, cost_rate=0.001):
    policy = HedgingPolicyVarSwap(cost_rate=cost_rate).to(device)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()
    return policy


def main():
    # ------------------------------------------------------------------ #
    # 1. Evaluate all models                                              #
    # ------------------------------------------------------------------ #
    print("Evaluating stock-only models ...")
    stock_only = {
        "CVaR (stock)":     evaluate_policy(load_stock_only("results/deep_hedge_var_cvar_annealed.pth", 0.0002), device=device, N=5000),
        "Variance (stock)": evaluate_policy(load_stock_only("results/deep_hedge_variance.pth"),           device=device, N=5000),
        "Entropic (stock)": evaluate_policy(load_stock_only("results/deep_hedge_entropic.pth"),           device=device, N=5000),
    }

    print("Evaluating stock+variance-swap models ...")
    varswap = {
        "CVaR (stock+VS)":     evaluate_varswap_policy(load_varswap("results/varswap_cvar.pth",      0.0002), device=device, N=5000),
        "Variance (stock+VS)": evaluate_varswap_policy(load_varswap("results/varswap_variance.pth"),         device=device, N=5000),
        "Entropic (stock+VS)": evaluate_varswap_policy(load_varswap("results/varswap_entropic.pth"),         device=device, N=5000),
    }

    # evaluate_policy returns (deep_cvar, delta_cvar) -- unpack
    delta_cvar = stock_only["CVaR (stock)"][1]     # same baseline for all

    results = {}
    for name, (deep_cvar, _) in stock_only.items():
        results[name] = {"CVaR": deep_cvar}
    for name, metrics in varswap.items():
        results[name] = {"CVaR": metrics["CVaR"]}

    # ------------------------------------------------------------------ #
    # 2. Print table                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'Model':<28} {'CVaR':>10} {'Delta vs Delta':>12}")
    print("-" * 52)
    print(f"{'Delta Hedge (baseline)':<28} {delta_cvar:>10.4f} {'--':>12}")
    for name, m in results.items():
        delta = m["CVaR"] - delta_cvar
        print(f"{name:<28} {m['CVaR']:>10.4f} {delta:>+12.4f}")

    # ------------------------------------------------------------------ #
    # 3. Bar chart: CVaR improvement over delta hedge                     #
    # ------------------------------------------------------------------ #
    names      = list(results.keys())
    cvar_vals  = [results[n]["CVaR"]          for n in names]
    deltas     = [results[n]["CVaR"] - delta_cvar for n in names]

    colours = ["#2196F3", "#2196F3", "#2196F3",   # stock-only -- blue
               "#FF5722", "#FF5722", "#FF5722"]    # stock+VS   -- orange

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, deltas, color=colours, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("CVaR improvement vs Delta Hedge")
    ax.set_title("Effect of Adding Variance Swap: CVaR Improvement by Objective & Instrument Set")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:+.2f}", ha="center", va="bottom", fontsize=8)

    # Legend patches
    from matplotlib.patches import Patch
    legend = [Patch(color="#2196F3", label="Stock only"),
              Patch(color="#FF5722", label="Stock + Variance Swap")]
    ax.legend(handles=legend)

    plt.tight_layout()
    plt.savefig("results/varswap_cvar_improvement.png", dpi=150)
    print("\nSaved: results/varswap_cvar_improvement.png")

    # ------------------------------------------------------------------ #
    # 4. PnL distribution: CVaR stock-only vs stock+VS                   #
    # ------------------------------------------------------------------ #
    pnl_stock = torch.load("results/pnl_deep.pt")
    pnl_vs    = torch.load("results/pnl_varswap_cvar.pt")
    pnl_delta = torch.load("results/pnl_delta.pt")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-40, 20, 80)
    ax2.hist(pnl_delta.numpy(), bins=bins, alpha=0.4, label="Delta Hedge",         color="gray")
    ax2.hist(pnl_stock.numpy(), bins=bins, alpha=0.5, label="CVaR (stock only)",   color="#2196F3")
    ax2.hist(pnl_vs.numpy(),    bins=bins, alpha=0.5, label="CVaR (stock + VS)",   color="#FF5722")
    ax2.set_xlabel("PnL")
    ax2.set_ylabel("Frequency")
    ax2.set_title("PnL Distribution: CVaR Objective -- Stock vs Stock+VarSwap")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("results/varswap_pnl_comparison.png", dpi=150)
    print("Saved: results/varswap_pnl_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
