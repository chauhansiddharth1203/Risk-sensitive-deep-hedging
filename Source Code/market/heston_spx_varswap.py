"""
market/heston_spx_varswap.py
-----------------------------
Heston model with SPX-calibrated parameters + variance swap.

Parameters sourced from standard SPX calibration literature
(consistent with Gatheral & Jacquier 2011, typical pre-crisis SPX):

    kappa   = 1.5    -- slower mean-reversion (textbook was 2.0)
    theta   = 0.05   -- slightly higher long-run variance (textbook: 0.04)
    sigma_v = 0.40   -- higher vol-of-vol (textbook: 0.30)
    rho     = -0.80  -- stronger leverage effect (textbook: -0.70)
    v0      = 0.05   -- higher starting variance (textbook: 0.04)

VS scaling is kept identical to the training simulator (v_t * S0 / 0.3)
so that a pre-trained policy can be evaluated without retraining.
This keeps the input normalization consistent.
"""

import torch

# SPX-calibrated Heston parameters
KAPPA   = 1.5
THETA   = 0.05
SIGMA_V = 0.40
RHO     = -0.80
V0      = 0.05

# VS normalisation constant -- kept the same as training to avoid input mismatch
VS_SCALE_SIGMA = 0.30    # same as training simulator


def simulate_heston_spx_varswap(
    N=64,
    T=30,
    S0=100.0,
    K=100.0,
    r=0.0,
    device="cpu",
):
    """
    Simulate Heston paths with SPX-calibrated parameters + variance swap.

    Returns:
        S         : (N, T+1) stock prices
        VS        : (N, T+1) variance swap prices  (= v_t * S0 / 0.3)
        payoff_fn : callable -- European call payoff max(S_T - K, 0)
    """

    dt = 1.0 / T

    rho_bar = torch.sqrt(torch.tensor(1.0 - RHO ** 2, device=device))

    S  = torch.zeros(N, T + 1, device=device)
    v  = torch.zeros(N, T + 1, device=device)
    VS = torch.zeros(N, T + 1, device=device)

    S[:, 0]  = S0
    v[:, 0]  = V0
    VS[:, 0] = V0 * S0 / VS_SCALE_SIGMA     # ≈ 16.67 under SPX params

    for t in range(T):
        z1 = torch.randn(N, device=device)
        z2 = RHO * z1 + rho_bar * torch.randn(N, device=device)

        v[:, t + 1] = torch.clamp(
            v[:, t]
            + KAPPA * (THETA - v[:, t]) * dt
            + SIGMA_V * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )

        S[:, t + 1] = S[:, t] * torch.exp(
            (r - 0.5 * v[:, t]) * dt
            + torch.sqrt(v[:, t] * dt) * z1
        )

        VS[:, t + 1] = v[:, t + 1] * S0 / VS_SCALE_SIGMA

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, VS, payoff_fn
