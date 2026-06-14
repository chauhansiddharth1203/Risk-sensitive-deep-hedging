"""
market/bates_spx_random.py
---------------------------
Combined simulator for the Week 8 sim-to-real sprint.

Three ingredients fused:
  (1) Bates dynamics        -- Heston stochastic vol + Poisson log-jumps
  (2) SPX-calibrated params -- around typical post-2008 short-maturity fits
  (3) Domain randomization  -- uniform-perturbation around the calibrated
                              centre for every batch

This addresses the Week 7 historical-stress finding that policies trained
on vanilla Heston (with σ_v fixed, no jumps, narrow param range) do not
generalise to real SPX crash windows.

Parameter centre (post-2008 SPY 1m ATM fits):
    κ_c = 1.50,  theta_c = 0.035,  σ_v,c = 0.45,  ρ_c = -0.75,  v0_c = 0.025
    lambda_J = 1.00,  μ_J = -0.05,  σ_J  = 0.08

Randomisation halfwidths (uniform around centre):
    κ  +/-0.60,  theta +/-0.015,  σ_v +/-0.15,  ρ +/-0.10,  v0 +/-0.015
    lambda_J scaled in [0.5, 2.0],  μ_J in [-0.08, -0.02], σ_J in [0.05, 0.12]

Variance-swap convention (fixed scaling, as in Week 5 robust training):
    VS_t = v_t · S0 / SIGMA_V_FIXED    with SIGMA_V_FIXED = 0.30.
"""

import math
import torch

# SPX-calibrated centres
KAPPA_C, THETA_C, SIGMA_V_C, RHO_C, V0_C = 1.50, 0.035, 0.45, -0.75, 0.025
LAM_J_C, MU_J_C, SIG_J_C                 = 1.00, -0.05, 0.08

# half-widths
KAPPA_HW, THETA_HW, SIGMA_V_HW, RHO_HW, V0_HW = 0.60, 0.015, 0.15, 0.10, 0.015

S0 = 100.0
K  = 100.0
T  = 30
SIGMA_V_FIXED = 0.30


def sample_params():
    """Uniformly sample one parameter draw within the randomisation box."""
    kappa   = torch.empty(1).uniform_(KAPPA_C - KAPPA_HW, KAPPA_C + KAPPA_HW).item()
    theta   = torch.empty(1).uniform_(THETA_C - THETA_HW, THETA_C + THETA_HW).item()
    sigma_v = torch.empty(1).uniform_(SIGMA_V_C - SIGMA_V_HW, SIGMA_V_C + SIGMA_V_HW).item()
    rho     = torch.empty(1).uniform_(RHO_C   - RHO_HW,   RHO_C   + RHO_HW).item()
    v0      = torch.empty(1).uniform_(V0_C    - V0_HW,    V0_C    + V0_HW).item()
    # jump scaling
    lam_J   = LAM_J_C * torch.empty(1).uniform_(0.5, 2.0).item()
    mu_J    = torch.empty(1).uniform_(-0.08, -0.02).item()
    sigma_J = torch.empty(1).uniform_(0.05, 0.12).item()
    return kappa, theta, sigma_v, rho, v0, lam_J, mu_J, sigma_J


def simulate_bates_spx_random(N=64, device="cpu", params=None):
    """
    Simulate N Bates paths with randomised SPX-calibrated params.

    Returns (S, VS, payoff_fn, params_used).
    """
    if params is None:
        params = sample_params()
    kappa, theta, sigma_v, rho, v0, lam_J, mu_J, sigma_J = params

    dt      = 1.0 / T
    rho_bar = (1.0 - rho ** 2) ** 0.5
    k_J     = math.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0   # jump compensator

    S  = torch.zeros(N, T + 1, device=device)
    v  = torch.zeros(N, T + 1, device=device)
    VS = torch.zeros(N, T + 1, device=device)

    S[:, 0]  = S0
    v[:, 0]  = v0
    VS[:, 0] = v0 * S0 / SIGMA_V_FIXED

    for t in range(T):
        z1 = torch.randn(N, device=device)
        z2 = rho * z1 + rho_bar * torch.randn(N, device=device)

        v[:, t + 1] = torch.clamp(
            v[:, t] + kappa * (theta - v[:, t]) * dt
            + sigma_v * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )

        # Poisson jump count per path per step
        n_jumps = torch.poisson(torch.full((N,), lam_J * dt, device=device))
        # total log-jump: N(n·μ_J, n·σ_J²)
        total_jump = (
            n_jumps * mu_J
            + sigma_J * torch.sqrt(n_jumps) * torch.randn(N, device=device)
        )

        # Risk-neutral drift with jump compensator
        S[:, t + 1] = S[:, t] * torch.exp(
            (-0.5 * v[:, t] - lam_J * k_J) * dt
            + torch.sqrt(v[:, t] * dt) * z1
            + total_jump
        )
        VS[:, t + 1] = v[:, t + 1] * S0 / SIGMA_V_FIXED

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, VS, payoff_fn, params
