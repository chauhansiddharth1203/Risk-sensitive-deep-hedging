"""
main_lstm_cvar.py
-----------------
Train and evaluate an LSTM-based deep hedging policy under CVaR objective.

Research question:
    Does giving the network full path memory (LSTM) improve hedging
    performance compared to the MLP that only sees the current price?

    In the Heston model, variance v_t is LATENT.  An LSTM processing the
    history of stock returns can implicitly infer v_t, and should therefore
    produce a better hedge.  We measure this by comparing CVaR@95 of the
    LSTM policy against the MLP CVaR policy from main.py.

Run:
    python main_lstm_cvar.py

Then compare results with the MLP outputs in results/pnl_deep.pt.
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from policy.network_lstm import HedgingPolicyLSTM
from policy.network import HedgingPolicy
from training.trainer_cvar_lstm import train_cvar_lstm
from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA  = 0.95
N_EVAL = 5000


def evaluate(policy, N=N_EVAL):
    policy.eval()
    S_test, payoff_fn = simulate_heston(N=N, device=device)
    with torch.no_grad():
        pnls = torch.stack([policy.rollout(S_test[i], payoff_fn) for i in range(N)])
    return pnls


def main():
    # ---- Train LSTM ----
    lstm_policy = HedgingPolicyLSTM(cost_rate=0.0002).to(device)
    optimizer   = torch.optim.Adam(lstm_policy.parameters(), lr=3e-4)
    print("=== Training LSTM policy (CVaR) ===")
    train_cvar_lstm(lstm_policy, optimizer, device=device)

    # ---- Evaluate both policies on same test set ----
    torch.manual_seed(99)
    pnl_lstm  = evaluate(lstm_policy)

    # Load pre-trained MLP (from main.py) if available; otherwise re-evaluate
    mlp_policy = HedgingPolicy(state_dim=3, action_dim=1, cost_rate=0.0002).to(device)
    try:
        mlp_policy.load_state_dict(
            torch.load("results/deep_hedge_var_cvar_annealed.pth", map_location=device)
        )
        pnl_mlp = evaluate(mlp_policy)
        print("Loaded pre-trained MLP.")
    except FileNotFoundError:
        print("MLP weights not found -- train main.py first for a fair comparison.")
        pnl_mlp = None

    torch.manual_seed(99)
    S_test, payoff_fn = simulate_heston(N=N_EVAL, device=device)
    pnl_delta = delta_hedge_pnl(S_test, payoff_fn)

    # ---- Print comparison ----
    print("\n=== Architecture Comparison (CVaR objective) ===")
    print(f"  Delta Hedge:   CVaR={cvar(pnl_delta,ALPHA):.4f}  Mean={pnl_delta.mean():.4f}")
    print(f"  LSTM:          CVaR={cvar(pnl_lstm, ALPHA):.4f}  Mean={pnl_lstm.mean():.4f}")
    if pnl_mlp is not None:
        print(f"  MLP:           CVaR={cvar(pnl_mlp,  ALPHA):.4f}  Mean={pnl_mlp.mean():.4f}")

    # ---- Save PnL ----
    torch.save(pnl_lstm.cpu(), "results/pnl_lstm_cvar.pt")

    # ---- Plot comparison ----
    bins = np.linspace(-40, 20, 80)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pnl_delta.cpu().numpy(), bins=bins, alpha=0.4, color="gray",    label="Delta Hedge")
    ax.hist(pnl_lstm.cpu().numpy(),  bins=bins, alpha=0.6, color="#9C27B0", label="LSTM (CVaR)")
    if pnl_mlp is not None:
        ax.hist(pnl_mlp.cpu().numpy(), bins=bins, alpha=0.5, color="#2196F3", label="MLP (CVaR)")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.set_title("PnL Distribution: LSTM vs MLP -- CVaR Objective")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/lstm_vs_mlp_pnl.png", dpi=150)
    print("Saved: results/lstm_vs_mlp_pnl.png")
    plt.show()


if __name__ == "__main__":
    main()
