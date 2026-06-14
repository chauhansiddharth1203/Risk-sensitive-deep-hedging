import torch

from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl


@torch.no_grad()
def evaluate_policy_variance(
    policy,
    device="cpu",
    N=10000,
):
    """
    Evaluate variance-optimized policy
    """

    policy.eval()

    S_test, payoff_fn = simulate_heston(
        N=N,
        device=device
    )

    pnl_deep = []

    for i in range(N):
        pnl = policy.rollout(
            S_test[i],
            payoff_fn
        )
        pnl_deep.append(pnl)

    pnl_deep = torch.stack(pnl_deep)
    pnl_delta = delta_hedge_pnl(
        S_test,
        payoff_fn
    )

    return pnl_deep.var().item(), pnl_delta.var().item()
