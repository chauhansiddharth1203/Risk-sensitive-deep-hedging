import torch
import os

from policy.network import HedgingPolicy
from training.trainer_entropic import train_entropic
from evaluation.evaluate_entropic import evaluate_policy_entropic
from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # -------------------------
    # Initialize policy
    # -------------------------
    policy = HedgingPolicy(
        state_dim=3,
        action_dim=1,
        cost_rate=0.001
    ).to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=3e-4
    )

    # -------------------------
    # Train policy (Entropic)
    # -------------------------
    train_entropic(
        policy,
        optimizer,
        device=device,
        risk_aversion=1.0
    )

    # -------------------------
    # Evaluate policy
    # -------------------------
    results = evaluate_policy_entropic(
        policy,
        device=device
    )

    print("\n=== Entropic Risk Hedging Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # -------------------------
    # Save PnL distributions for plots
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

    torch.save(pnl_deep, "results/pnl_entropic_deep.pt")
    torch.save(pnl_delta, "results/pnl_entropic_delta.pt")

    print("Entropic PnL saved.")


if __name__ == "__main__":
    main()
