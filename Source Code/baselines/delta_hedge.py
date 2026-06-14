import torch
import math


def bs_call_delta(S, K, tau, r, sigma, eps=1e-6):
    """
    Black-Scholes delta for a call option
    S: Tensor (batch,)
    tau: float or Tensor
    """

    # Convert tau to tensor if needed
    if not torch.is_tensor(tau):
        tau = torch.tensor(tau, device=S.device)

    d1 = (
        torch.log(S / K)
        + (r + 0.5 * sigma**2) * tau
    ) / (sigma * torch.sqrt(tau + eps))

    return torch.distributions.Normal(0, 1).cdf(d1)


@torch.no_grad()
def delta_hedge_pnl(S, payoff_fn, K=100.0, r=0.0, sigma=0.2):
    """
    S: (N, T+1)
    payoff_fn: callable
    """

    N, T = S.shape[0], S.shape[1] - 1
    pnl = torch.zeros(N, device=S.device)

    prev_delta = torch.zeros(N, device=S.device)

    for t in range(T):
        tau = (T - t) / T

        delta = bs_call_delta(S[:, t], K, tau, r, sigma)

        # PnL from previous position
        pnl += prev_delta * (S[:, t + 1] - S[:, t])

        # Transaction cost (same as deep hedge)
        pnl -= 0.0002 * torch.abs(delta - prev_delta)

        prev_delta = delta

    pnl -= payoff_fn(S[:, -1])
    return pnl
