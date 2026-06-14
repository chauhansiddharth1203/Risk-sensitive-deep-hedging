import torch
import os

from policy.network import HedgingPolicy
from training.trainer_variance import train_variance
from evaluation.evaluate_variance import evaluate_policy_variance
from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # -------------------------
    # Initialize policy
    # -------------------------
    policy = HedgingPolicy(
        state_dim=3,
        action_dim=1   # stock only
    ).to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=3e-4
    )

    # -------------------------
    # Train policy (Variance)
    # -------------------------
    train_variance(
        policy,
        optimizer,
        device=device
    )

    # -------------------------
    # Evaluate policy
    # -------------------------
    deep_var, delta_var = evaluate_policy_variance(
        policy,
        device=device
    )

    print("Deep Hedge (Variance) :", deep_var)
    print("Delta Hedge (Variance):", delta_var)

    # -------------------------
    # Save PnL for plots
    # -------------------------
    os.makedirs("results", exist_ok=True)

    S_test, payoff_fn = simulate_heston(
        N=3000,
        device=device
    )

    pnl_deep = []
    policy.eval()
    with torch.no_grad():
        for i in range(S_test.shape[0]):
            pnl_deep.append(policy.rollout(S_test[i], payoff_fn))

    pnl_deep = torch.stack(pnl_deep).cpu()
    pnl_delta = delta_hedge_pnl(S_test, payoff_fn).cpu()

    torch.save(pnl_deep, "results/pnl_variance_deep.pt")
    torch.save(pnl_delta, "results/pnl_variance_delta.pt")

    print("Variance PnL saved.")


if __name__ == "__main__":
    main()
