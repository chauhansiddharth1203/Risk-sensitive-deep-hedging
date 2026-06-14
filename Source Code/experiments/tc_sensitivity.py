"""
experiments/tc_sensitivity.py
------------------------------

Systematic study of how transaction cost rate affects each risk objective.

For each tc_rate in TC_RATES:
    Train CVaR / Variance / Entropic policies (stock-only, fixed architecture)
    Evaluate on a held-out test set
    Record CVaR@95, Mean PnL, Std PnL

Also computes the Black-Scholes delta hedge at the same tc rates.

Saves results to results/tc_sensitivity.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import csv

from policy.network import HedgingPolicy
from training.trainer_cvar import train as train_cvar
from training.trainer_variance import train_variance
from training.trainer_entropic import train_entropic
from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar

os.makedirs("results", exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TC_RATES   = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
EPOCHS     = 200          # fewer epochs per run to keep total time manageable
BATCH      = 64
N_EVAL     = 5000
ALPHA      = 0.95
SEED       = 42


def make_policy(tc_rate):
    return HedgingPolicy(state_dim=3, action_dim=1, cost_rate=tc_rate).to(DEVICE)


def make_optimizer(policy):
    return torch.optim.Adam(policy.parameters(), lr=3e-4)


@torch.no_grad()
def evaluate(policy, tc_rate):
    """Returns dict of metrics for a trained policy."""
    policy.eval()
    S_test, payoff_fn = simulate_heston(N=N_EVAL, device=DEVICE)
    pnls = torch.stack([policy.rollout(S_test[i], payoff_fn) for i in range(N_EVAL)])
    pnl_delta = delta_hedge_pnl(S_test, payoff_fn, K=100.0, sigma=0.2)
    # Override delta hedge tc to match current tc_rate
    # (recompute with matching cost_rate)
    from baselines.delta_hedge import bs_call_delta
    N, T = S_test.shape[0], S_test.shape[1] - 1
    pnl_d = torch.zeros(N, device=DEVICE)
    prev_d = torch.zeros(N, device=DEVICE)
    for t in range(T):
        tau   = (T - t) / T
        d     = bs_call_delta(S_test[:, t], 100.0, tau, 0.0, 0.2)
        pnl_d += prev_d * (S_test[:, t + 1] - S_test[:, t])
        pnl_d -= tc_rate * torch.abs(d - prev_d)
        prev_d = d
    pnl_d -= payoff_fn(S_test[:, -1])

    return {
        "cvar_deep":  cvar(pnls,  ALPHA).item(),
        "mean_deep":  pnls.mean().item(),
        "std_deep":   pnls.std().item(),
        "cvar_delta": cvar(pnl_d, ALPHA).item(),
        "mean_delta": pnl_d.mean().item(),
    }


def run():
    torch.manual_seed(SEED)
    rows = []

    for tc in TC_RATES:
        print(f"\n{'='*60}")
        print(f" tc_rate = {tc}")
        print(f"{'='*60}")

        for objective, train_fn in [
            ("CVaR",     lambda p, o: train_cvar(p, o, device=DEVICE, epochs=EPOCHS, batch_size=BATCH, print_every=50)),
            ("Variance", lambda p, o: train_variance(p, o, device=DEVICE, epochs=EPOCHS, batch_size=BATCH, print_every=50)),
            ("Entropic", lambda p, o: train_entropic(p, o, device=DEVICE, epochs=EPOCHS, batch_size=BATCH, print_every=50)),
        ]:
            print(f"\n--- {objective} ---")
            policy    = make_policy(tc)
            optimizer = make_optimizer(policy)
            train_fn(policy, optimizer)

            metrics = evaluate(policy, tc)
            row = {"tc_rate": tc, "objective": objective, **metrics}
            rows.append(row)
            print(f"  CVaR deep={metrics['cvar_deep']:.4f}  delta={metrics['cvar_delta']:.4f}  "
                  f"improvement={metrics['cvar_deep']-metrics['cvar_delta']:+.4f}")

    # ---- Save to CSV ----
    csv_path = "results/tc_sensitivity.csv"
    fieldnames = ["tc_rate", "objective",
                  "cvar_deep", "mean_deep", "std_deep",
                  "cvar_delta", "mean_delta"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run()
