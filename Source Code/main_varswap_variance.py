import torch
import os
from policy.network_varswap import HedgingPolicyVarSwap
from training.trainer_variance_varswap import train_variance_varswap
from evaluation.evaluate_varswap import evaluate_varswap_policy

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    policy = HedgingPolicyVarSwap(cost_rate=0.001).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    train_variance_varswap(policy, optimizer, device=device)

    results = evaluate_varswap_policy(policy, device=device)
    print("\n=== VarSwap Variance Policy Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    from market.heston_with_var_swap import simulate_heston_with_var_swap
    S_test, VS_test, payoff_fn = simulate_heston_with_var_swap(N=3000, device=device)
    policy.eval()
    with torch.no_grad():
        pnls = torch.stack([
            policy.rollout(S_test[i], VS_test[i], payoff_fn)
            for i in range(3000)
        ])
    torch.save(pnls.cpu(), "results/pnl_varswap_variance.pt")
    print("PnL saved to results/pnl_varswap_variance.pt")


if __name__ == "__main__":
    main()
