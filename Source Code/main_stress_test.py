import torch

from policy.network import HedgingPolicy
from evaluation.evaluate_stress import evaluate_under_stress

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_policy(path, action_dim):
    """
    Load a trained policy with correct action dimension
    """
    policy = HedgingPolicy(
        state_dim=3,
        action_dim=action_dim,
        cost_rate=0.001
    ).to(device)

    policy.load_state_dict(
        torch.load(path, map_location=device)
    )
    policy.eval()
    return policy


def main():
    policies = {
        "Variance": load_policy(
            "results/deep_hedge_variance.pth",
            action_dim=1
        ),
        "Entropic": load_policy(
            "results/deep_hedge_entropic.pth",
            action_dim=1
        ),
        "CVaR": load_policy(
            "results/deep_hedge_var_cvar_annealed.pth",
            action_dim=1
        ),
    }

    print("\n=== Stress Test Results (σ_v x 1.5) ===")

    for name, policy in policies.items():
        res = evaluate_under_stress(
            policy,
            device=device,
            sigma_v_multiplier=1.5
        )

        print(f"\n{name} Hedge:")
        for k, v in res.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
