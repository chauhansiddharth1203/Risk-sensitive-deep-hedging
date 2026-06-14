"""
market/regime_switching_heston_varswap.py
------------------------------------------
Two-regime Markov-switching Heston model with variance swap.

Regime 0 -- CALM:
    kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.70  (same as training)

Regime 1 -- STRESSED:
    kappa=1.5, theta=0.09, sigma_v=0.50, rho=-0.85
    (higher long-run vol, stronger leverage, more erratic vol-of-vol)

Transition probabilities per step (dt = 1/30):
    P(calm  -> stressed) = 0.05  =>  calm regime lasts ~20 steps on average
    P(stressed -> calm)  = 0.25  =>  stress lasts ~4 steps on average
    Stationary: ~83% calm, ~17% stressed

All paths start in the calm regime.
VS scaling: v_t * S0 / 0.3  (same as training -- keeps input normalization fixed).

Returns:
    S         : (N, T+1)
    VS        : (N, T+1)
    payoff_fn : callable
    regimes   : (N, T)  int tensor -- 0=calm, 1=stressed at each step
"""

import torch

# Calm regime parameters
CALM = dict(kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.70)

# Stressed regime parameters
STRESS = dict(kappa=1.5, theta=0.09, sigma_v=0.50, rho=-0.85)

# Transition probabilities per step
P_CALM_TO_STRESS  = 0.05
P_STRESS_TO_CALM  = 0.25

# VS normalisation (fixed to match pre-trained policy)
VS_SCALE_SIGMA = 0.30


def simulate_regime_switching_heston_varswap(
    N=64,
    T=30,
    S0=100.0,
    K=100.0,
    r=0.0,
    device="cpu",
):
    dt = 1.0 / T

    # Pre-compute rho_bar for each regime
    rho_bar_calm   = torch.sqrt(torch.tensor(1.0 - CALM["rho"]   ** 2, device=device))
    rho_bar_stress = torch.sqrt(torch.tensor(1.0 - STRESS["rho"] ** 2, device=device))

    S       = torch.zeros(N, T + 1, device=device)
    v       = torch.zeros(N, T + 1, device=device)
    VS      = torch.zeros(N, T + 1, device=device)
    regimes = torch.zeros(N, T, dtype=torch.long, device=device)

    S[:, 0]  = S0
    v[:, 0]  = CALM["theta"]               # start at long-run calm variance
    VS[:, 0] = CALM["theta"] * S0 / VS_SCALE_SIGMA

    # All paths start in calm regime
    current_regime = torch.zeros(N, dtype=torch.long, device=device)   # 0 = calm

    for t in range(T):
        # --- Regime transition ---
        u = torch.rand(N, device=device)
        calm_mask   = (current_regime == 0)
        stress_mask = (current_regime == 1)

        switch_to_stress = calm_mask   & (u < P_CALM_TO_STRESS)
        switch_to_calm   = stress_mask & (u < P_STRESS_TO_CALM)

        current_regime[switch_to_stress] = 1
        current_regime[switch_to_calm]   = 0

        regimes[:, t] = current_regime

        # --- Build per-path parameters ---
        kappa   = torch.where(calm_mask, torch.tensor(CALM["kappa"],   device=device),
                                          torch.tensor(STRESS["kappa"],  device=device))
        theta   = torch.where(calm_mask, torch.tensor(CALM["theta"],   device=device),
                                          torch.tensor(STRESS["theta"],  device=device))
        sigma_v = torch.where(calm_mask, torch.tensor(CALM["sigma_v"], device=device),
                                          torch.tensor(STRESS["sigma_v"], device=device))
        rho     = torch.where(calm_mask, torch.tensor(CALM["rho"],     device=device),
                                          torch.tensor(STRESS["rho"],    device=device))
        rho_bar = torch.where(calm_mask, rho_bar_calm, rho_bar_stress)

        # --- Correlated Brownian increments ---
        z1 = torch.randn(N, device=device)
        z2 = rho * z1 + rho_bar * torch.randn(N, device=device)

        # --- Variance update ---
        v[:, t + 1] = torch.clamp(
            v[:, t]
            + kappa * (theta - v[:, t]) * dt
            + sigma_v * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )

        # --- Stock update ---
        S[:, t + 1] = S[:, t] * torch.exp(
            (r - 0.5 * v[:, t]) * dt
            + torch.sqrt(v[:, t] * dt) * z1
        )

        # --- VS (tracks diffusion variance, fixed scaling) ---
        VS[:, t + 1] = v[:, t + 1] * S0 / VS_SCALE_SIGMA

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, VS, payoff_fn, regimes
