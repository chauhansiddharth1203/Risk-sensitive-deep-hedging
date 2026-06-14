import torch

from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar


@torch.no_grad()
def evaluate_policy_entropic(
    policy,
    device="cpu",
    N=10000,
    alpha=0.95,
):
    """
    Evaluate entropic-risk-trained policy using CVaR and variance
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

    results = {
        "mean": pnl_deep.mean().item(),
        "variance": pnl_deep.var().item(),
        "cvar": cvar(pnl_deep, alpha).item(),
        "delta_cvar": cvar(pnl_delta, alpha).item(),
    }

    return results
