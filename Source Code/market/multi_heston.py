"""
market/multi_heston.py
-----------------------
Two-asset Heston simulator with cross-asset correlation, for spread-option
hedging experiments (Week 6).

Each asset i in {1,2} follows its own Heston SDE:
    dS_i = sqrt(v_i) S_i dW_i^S
    dv_i = kappa_i (theta_i - v_i) dt + sigma_v_i sqrt(v_i) dW_i^v
    d<W_i^S, W_i^v> = rho_i dt                       (leverage)
    d<W_1^S, W_2^S> = rho_12 dt                      (cross-asset)

Vol shocks (W_i^v) are independent across assets for simplicity -- only the
spot Brownian motions carry the cross-asset correlation rho_12.

Variance-swap convention (matches single-asset experiments):
    VS_i(t) = v_i(t) * S0_i / SIGMA_V_FIXED
with SIGMA_V_FIXED = 0.30.

The hedger is given 4 instruments: S^1, S^2, VS^1, VS^2.
The target payoff is a spread call
    max( S^1_T - S^2_T - K, 0 ),
which is the simplest multi-asset European product with non-trivial
correlation dependence.
"""

import torch

# ---- defaults (both assets at roughly "equity index" vol) ------------- #
KAPPA_1, THETA_1, SIGMA_V_1, RHO_1, V0_1 = 2.0, 0.04, 0.30, -0.70, 0.04
KAPPA_2, THETA_2, SIGMA_V_2, RHO_2, V0_2 = 2.0, 0.04, 0.30, -0.70, 0.04
RHO_12 = 0.5     # cross-asset spot correlation

S0_1, S0_2 = 100.0, 100.0
K_SPREAD   = 0.0     # ATM spread
T_STEPS    = 30
SIGMA_V_FIXED = 0.30


def simulate_multi_heston(
    N=64,
    T=T_STEPS,
    rho_12=RHO_12,
    device="cpu",
    params_1=None,
    params_2=None,
    K=K_SPREAD,
):
    """
    Simulate N paths of two correlated Heston assets.

    Returns:
        S1, S2   : (N, T+1) spot paths for asset 1, 2
        VS1, VS2 : (N, T+1) variance-swap paths (fixed scaling)
        v1, v2   : (N, T+1) instantaneous variances
        payoff_fn: callable taking (S1, S2) and returning spread-call payoff
        meta     : dict with parameters used
    """

    if params_1 is None:
        params_1 = (KAPPA_1, THETA_1, SIGMA_V_1, RHO_1, V0_1)
    if params_2 is None:
        params_2 = (KAPPA_2, THETA_2, SIGMA_V_2, RHO_2, V0_2)

    k1, th1, sv1, r1, v01 = params_1
    k2, th2, sv2, r2, v02 = params_2

    dt   = 1.0 / T
    rb1  = (1.0 - r1 ** 2) ** 0.5
    rb2  = (1.0 - r2 ** 2) ** 0.5
    rb12 = (1.0 - rho_12 ** 2) ** 0.5

    S1  = torch.zeros(N, T + 1, device=device)
    S2  = torch.zeros(N, T + 1, device=device)
    v1  = torch.zeros(N, T + 1, device=device)
    v2  = torch.zeros(N, T + 1, device=device)
    VS1 = torch.zeros(N, T + 1, device=device)
    VS2 = torch.zeros(N, T + 1, device=device)

    S1[:, 0], S2[:, 0] = S0_1, S0_2
    v1[:, 0], v2[:, 0] = v01,   v02
    VS1[:, 0] = v01 * S0_1 / SIGMA_V_FIXED
    VS2[:, 0] = v02 * S0_2 / SIGMA_V_FIXED

    for t in range(T):
        # independent normals
        zS1 = torch.randn(N, device=device)
        zS2_indep = torch.randn(N, device=device)
        # cross-asset correlation on spot shocks
        zS2 = rho_12 * zS1 + rb12 * zS2_indep

        # independent vol shocks per asset, each correlated with its own spot
        zv1_indep = torch.randn(N, device=device)
        zv2_indep = torch.randn(N, device=device)
        zv1 = r1 * zS1 + rb1 * zv1_indep
        zv2 = r2 * zS2 + rb2 * zv2_indep

        v1[:, t + 1] = torch.clamp(
            v1[:, t] + k1 * (th1 - v1[:, t]) * dt
            + sv1 * torch.sqrt(v1[:, t] * dt) * zv1,
            min=1e-6,
        )
        v2[:, t + 1] = torch.clamp(
            v2[:, t] + k2 * (th2 - v2[:, t]) * dt
            + sv2 * torch.sqrt(v2[:, t] * dt) * zv2,
            min=1e-6,
        )

        S1[:, t + 1] = S1[:, t] * torch.exp(
            -0.5 * v1[:, t] * dt + torch.sqrt(v1[:, t] * dt) * zS1
        )
        S2[:, t + 1] = S2[:, t] * torch.exp(
            -0.5 * v2[:, t] * dt + torch.sqrt(v2[:, t] * dt) * zS2
        )

        VS1[:, t + 1] = v1[:, t + 1] * S0_1 / SIGMA_V_FIXED
        VS2[:, t + 1] = v2[:, t + 1] * S0_2 / SIGMA_V_FIXED

    def payoff_fn(S1_T, S2_T):
        return torch.clamp(S1_T - S2_T - K, min=0.0)

    meta = dict(
        params_1=params_1, params_2=params_2, rho_12=rho_12,
        S0_1=S0_1, S0_2=S0_2, K=K, T=T,
    )
    return S1, S2, VS1, VS2, v1, v2, payoff_fn, meta
