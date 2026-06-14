"""
market/bates.py
---------------
Bates model simulator (Heston stochastic volatility + Poisson jumps in stock).

Jump parameters:
  lambda_J = 1.0   -- 1 expected jump per year
  mu_J     = -0.05 -- average log-jump size (-5%)
  sigma_J  =  0.08 -- std of log-jump size

Returns (S, payoff_fn) -- stock prices only, for stock-only policy training.
"""

import math
import torch

LAMBDA_J = 1.0
MU_J     = -0.05
SIGMA_J  =  0.08
K_J      = math.exp(MU_J + 0.5 * SIGMA_J ** 2) - 1.0   # ≈ -0.0468


def simulate_bates(
    N=64,
    T=30,
    S0=100.0,
    K=100.0,
    r=0.0,
    device="cpu",
):
    """
    Simulate Bates model stock paths.

    Returns:
        S         : (N, T+1) stock price paths
        payoff_fn : callable -- European call payoff max(S_T - K, 0)
    """

    dt = 1.0 / T

    kappa   = 2.0
    theta   = 0.04
    sigma_v = 0.3
    rho     = -0.7
    v0      = 0.04

    rho_bar = torch.sqrt(torch.tensor(1.0 - rho ** 2, device=device))

    S = torch.zeros(N, T + 1, device=device)
    v = torch.zeros(N, T + 1, device=device)

    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(T):
        z1 = torch.randn(N, device=device)
        z2 = rho * z1 + rho_bar * torch.randn(N, device=device)

        # Heston variance update
        v[:, t + 1] = torch.clamp(
            v[:, t]
            + kappa * (theta - v[:, t]) * dt
            + sigma_v * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )

        # Poisson jump count per path per step
        n_jumps = torch.poisson(
            torch.full((N,), LAMBDA_J * dt, device=device)
        )

        # Total log-jump: sum of n_jumps iid N(mu_J, sigma_J^2)
        # => N(n_jumps * mu_J,  n_jumps * sigma_J^2)
        total_jump = (
            n_jumps * MU_J
            + SIGMA_J * torch.sqrt(n_jumps) * torch.randn(N, device=device)
        )

        # Risk-neutral stock update (drift corrected for jump compensator)
        S[:, t + 1] = S[:, t] * torch.exp(
            (r - 0.5 * v[:, t] - LAMBDA_J * K_J) * dt
            + torch.sqrt(v[:, t] * dt) * z1
            + total_jump
        )

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, payoff_fn
