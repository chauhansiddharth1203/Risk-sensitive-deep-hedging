import torch

from market.heston import simulate_heston
from baselines.delta_hedge import delta_hedge_pnl
from risk.cvar import cvar


@torch.no_grad()
def evaluate_policy(
    policy,
    device="cpu",
    N=10000,
    alpha=0.95,
):
    """
    Evaluate deep hedging policy vs delta hedge using CVaR
    """

    policy.eval()

    # ---- Simulate test market paths ----
    S_test, payoff_fn = simulate_heston(
        N=N,
        device=device
    )

    # ---- Deep hedging PnL ----
    pnl_deep = []

    for i in range(N):
        pnl = policy.rollout(
            S_test[i],
            payoff_fn
        )
        pnl_deep.append(pnl)

    pnl_deep = torch.stack(pnl_deep)

    # ---- Delta hedge baseline ----
    pnl_delta = delta_hedge_pnl(
        S_test,
        payoff_fn
    )

    # ---- CVaR ----
    deep_cvar = cvar(pnl_deep, alpha)
    delta_cvar = cvar(pnl_delta, alpha)
    print("Deep Mean PnL:", pnl_deep.mean().item())
    print("Delta Mean PnL:", pnl_delta.mean().item())
    return deep_cvar.item(), delta_cvar.item()
