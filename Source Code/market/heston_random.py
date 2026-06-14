"""
market/heston_random.py
-----------------------
Heston simulator with randomly-sampled parameters for domain randomization.

For each call, parameters (κ, theta, σ_v, ρ, v0) are sampled uniformly from:
    κ     in [0.50, 3.00]   (slow to fast mean-reversion)
    theta     in [0.02, 0.12]   (14% to 35% long-run vol)
    σ_v   in [0.20, 0.80]   (spans normal AND crisis vol-of-vol!)
    ρ     in [-0.90, -0.40] (mild to severe leverage)
    v0    in [0.02, 0.10]   (14% to 32% initial vol)

VS scaling convention:
    VS_t = v_t * S0 / σ_v_FIXED   (σ_v_FIXED = 0.30)

Using a FIXED scaling (not the sampled σ_v) keeps the policy's VS input on
a consistent scale across all parameter draws -- the network sees VS as a
linearly rescaled view of v_t regardless of which Heston regime generated
the path. This is the key to σ_v-invariant learning.
"""

import torch

# Sampling ranges (spans crisis regime)
KAPPA_RANGE   = (0.50, 3.00)
THETA_RANGE   = (0.02, 0.12)
SIGMA_V_RANGE = (0.20, 0.80)
RHO_RANGE     = (-0.90, -0.40)
V0_RANGE      = (0.02, 0.10)

# Fixed constants
S0 = 100.0
K  = 100.0
T  = 30
SIGMA_V_FIXED = 0.30   # used for VS scaling only


def sample_heston_params():
    """Uniformly sample one Heston parameter set."""
    kappa   = torch.empty(1).uniform_(*KAPPA_RANGE).item()
    theta   = torch.empty(1).uniform_(*THETA_RANGE).item()
    sigma_v = torch.empty(1).uniform_(*SIGMA_V_RANGE).item()
    rho     = torch.empty(1).uniform_(*RHO_RANGE).item()
    v0      = torch.empty(1).uniform_(*V0_RANGE).item()
    return kappa, theta, sigma_v, rho, v0


def simulate_heston_random(N=64, device="cpu", params=None):
    """
    Simulate N Heston paths with randomized parameters.

    If `params` is provided as (κ, theta, σ_v, ρ, v0), use those deterministically.
    Otherwise sample a new parameter set.

    Returns (S, VS, payoff_fn, params_used) where:
        S   : (N, T+1) stock paths
        VS  : (N, T+1) variance swap (scaled with FIXED σ_v for consistency)
        payoff_fn : ATM call
        params_used : tuple (κ, theta, σ_v, ρ, v0)
    """
    if params is None:
        params = sample_heston_params()
    kappa, theta, sigma_v, rho, v0 = params

    dt      = 1.0 / T
    rho_bar = (1.0 - rho ** 2) ** 0.5

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
        S[:, t + 1] = S[:, t] * torch.exp(
            -0.5 * v[:, t] * dt + torch.sqrt(v[:, t] * dt) * z1
        )
        VS[:, t + 1] = v[:, t + 1] * S0 / SIGMA_V_FIXED

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, VS, payoff_fn, params
