import torch
from market.heston_calibrated import simulate_heston_calibrated
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar


@torch.no_grad()
def evaluate_under_calibration(
    policy,
    regime,
    device="cpu",
    N=10000,
    alpha=0.95,
):
    """
    Evaluate a policy under a calibrated market regime
    """

    policy.eval()

    S_test, payoff_fn = simulate_heston_calibrated(
        regime=regime,
        N=N,
        device=device,
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
