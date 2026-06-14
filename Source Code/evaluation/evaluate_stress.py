import torch
import numpy as np

from market.heston_stress import simulate_heston_stress
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar


@torch.no_grad()
def evaluate_under_stress(
    policy,
    device="cpu",
    N=10000,
    alpha=0.95,
    sigma_v_multiplier=1.5,
):
    """
    Evaluate a policy under stressed market conditions
    """

    policy.eval()

    S_test, payoff_fn = simulate_heston_stress(
        N=N,
        device=device,
        sigma_v_multiplier=sigma_v_multiplier,
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
        "deep_cvar": cvar(pnl_deep, alpha).item(),
        "delta_cvar": cvar(pnl_delta, alpha).item(),
        "deep_mean": pnl_deep.mean().item(),
        "deep_variance": pnl_deep.var().item(),
    }

    return results
