import torch
from market.heston_with_var_swap import simulate_heston_with_var_swap
from risk.cvar import cvar


@torch.no_grad()
def evaluate_varswap_policy(
    policy,
    device="cpu",
    N=10000,
    alpha=0.95,
):
    """
    Evaluate a stock+variance-swap hedging policy.

    Returns a dict with CVaR, mean PnL, std PnL, and VaR.
    """
    policy.eval()

    S_test, VS_test, payoff_fn = simulate_heston_with_var_swap(
        N=N,
        device=device,
    )

    pnl_list = []
    for i in range(N):
        pnl = policy.rollout(S_test[i], VS_test[i], payoff_fn)
        pnl_list.append(pnl)

    pnls = torch.stack(pnl_list)

    return {
        "CVaR":     cvar(pnls, alpha).item(),
        "Mean PnL": pnls.mean().item(),
        "Std PnL":  pnls.std().item(),
        "VaR":      torch.quantile(pnls, 1 - alpha).item(),
    }
