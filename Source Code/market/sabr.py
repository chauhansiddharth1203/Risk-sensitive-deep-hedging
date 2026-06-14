"""
market/sabr.py
--------------
SABR stochastic volatility model simulator.

Used for model misspecification robustness tests:
  - The policy is trained on Heston, then tested on SABR paths
  - Different vol dynamics (CEV-type S diffusion, log-normal vol)

SABR model:
    dS_t  = σ_t · S_t^β · dW_1
    dσ_t  = alpha · σ_t · dW_2
    corr(dW_1, dW_2) = ρ

Parameters calibrated so ATM implied vol ≈ 20% (matches Heston training):
    β     = 0.5   (CEV exponent -- square-root in stock)
    alpha     = 0.30  (vol-of-vol)
    ρ     = -0.60 (negative skew leverage)
    σ_0   = 0.20  (initial vol)
    S0    = 100.0, K = 100.0, T = 30 steps (30/252 year)

Returns (S, VS_pseudo, payoff_fn) where:
    VS_pseudo_t = σ_t² · S0 / σ_v_heston
This lets a Heston-trained stock+VS policy receive meaningful vol signals
even on SABR paths (same scaling convention as the training environment).
"""

import torch

# SABR parameters
BETA      = 0.5
ALPHA     = 0.30
RHO_SABR  = -0.60
SIGMA_0   = 0.20

# Matching Heston environment
S0              = 100.0
K               = 100.0
N_STEPS         = 30
T_YEARS         = 30 / 252
DT              = T_YEARS / N_STEPS
SIGMA_V_HESTON  = 0.30   # Heston sigma_v used for VS scaling


def simulate_sabr(N=1000, device="cpu",
                  beta=BETA, alpha=ALPHA, rho=RHO_SABR,
                  sigma_0=SIGMA_0):
    """
    Simulate N SABR paths.

    Parameters
    ----------
    N       : number of paths
    device  : torch device
    beta, alpha, rho, sigma_0 : SABR parameters

    Returns
    -------
    S          : (N, N_STEPS+1) stock paths
    VS_pseudo  : (N, N_STEPS+1) pseudo variance swap  (= σ_t² · S0 / σ_v_heston)
    payoff_fn  : callable, ATM call payoff
    """
    dt   = DT
    sdt  = dt ** 0.5
    rho2 = (1.0 - rho ** 2) ** 0.5

    S     = torch.zeros(N, N_STEPS + 1)
    sigma = torch.zeros(N, N_STEPS + 1)

    S[:, 0]     = S0
    sigma[:, 0] = sigma_0

    for t in range(N_STEPS):
        z1 = torch.randn(N)
        z2 = rho * z1 + rho2 * torch.randn(N)   # correlated normals

        S_t   = S[:, t]
        sig_t = sigma[:, t]

        # SABR stock dynamics: dS = σ * S^β * dW_1  (Euler-Maruyama)
        dS = sig_t * torch.clamp(S_t, min=1e-4) ** beta * sdt * z1
        S[:, t + 1] = torch.clamp(S_t + dS, min=1.0)   # absorb at 1

        # Log-normal vol dynamics: σ stays positive
        sigma[:, t + 1] = sig_t * torch.exp(
            alpha * sdt * z2 - 0.5 * alpha ** 2 * dt
        )
        sigma[:, t + 1] = torch.clamp(sigma[:, t + 1], min=1e-4)

    # Pseudo VS: v_t = σ_t²,  VS_t = v_t · S0 / σ_v_heston
    VS_pseudo = (sigma ** 2) * (S0 / SIGMA_V_HESTON)

    S         = S.to(device)
    VS_pseudo = VS_pseudo.to(device)

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, VS_pseudo, payoff_fn
