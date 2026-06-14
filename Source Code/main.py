import torch
import os

from policy.network import HedgingPolicy
from training.trainer_cvar import train
from evaluation.evaluate import evaluate_policy
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
    cost_rate=0.0002
    ).to(device)


    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=3e-4
    )

    # -------------------------
    # Train policy
    # -------------------------
    train(
        policy,
        optimizer,
        device=device
    )

    # -------------------------
    # Evaluate policy (CVaR)
    # -------------------------
    deep_cvar, delta_cvar = evaluate_policy(
        policy,
        device=device
    )

    print("Deep Hedge (Stock + Variance) CVaR (95%):", deep_cvar)
    print("Delta Hedge CVaR (95%):", delta_cvar)

    # -------------------------
    # Save PnL distributions for plots
    # -------------------------
    print("Saving PnL distributions for plotting...")

    os.makedirs("results", exist_ok=True)

    # IMPORTANT: smaller N for plotting (fast + sufficient)
    N_plot = 3000

    S_test, payoff_fn = simulate_heston(
        N=N_plot,
        device=device
    )

    # ---- Deep hedge PnL ----
    policy.eval()
    pnl_deep = torch.empty(N_plot, device=device)

    with torch.no_grad():
        for i in range(N_plot):
            if i % 500 == 0:
                print(f"Deep hedge rollout {i}/{N_plot}")
            pnl_deep[i] = policy.rollout(S_test[i], payoff_fn)

    pnl_deep = pnl_deep.cpu()

    # ---- Delta hedge PnL ----
    pnl_delta = delta_hedge_pnl(S_test, payoff_fn).cpu()

    torch.save(pnl_deep, "results/pnl_deep.pt")
    torch.save(pnl_delta, "results/pnl_delta.pt")

    print("PnL saved successfully.")


if __name__ == "__main__":
    main()
