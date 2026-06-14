import torch


def simulate_heston_with_var_swap(
    N=64,
    T=30,
    S0=100.0,
    K=100.0,
    r=0.0,
    device="cpu",
):
    """
    Simulate Heston model paths and return both stock and variance swap prices.

    Scaling choice:  VS_t = v_t * S0 / sigma_v

    This normalises the two instruments so that their per-step P&L standard
    deviations are equal:
        Std(DeltaS)  ≈ S0 * sqrt(v0 * dt)            ≈ 3.65
        Std(DeltaVS) = sigma_v * sqrt(v0 * dt) * S0/sigma_v = S0 * sqrt(v0 * dt) ≈ 3.65

    Equal volatility prevents the network from over-weighting the variance swap
    and keeps training stable.

    Returns:
        S         : (N, T+1) stock price paths
        VS        : (N, T+1) variance swap price paths  (= v * S0 / sigma_v)
        payoff_fn : callable -- European call payoff max(S_T - K, 0)
    """

    dt = 1.0 / T

    # Heston parameters (same as base model)
    kappa   = 2.0
    theta   = 0.04
    sigma_v = 0.3
    rho     = -0.7
    v0      = 0.04

    rho_bar = torch.sqrt(
        torch.tensor(1.0 - rho**2, device=device)
    )

    S  = torch.zeros(N, T + 1, device=device)
    v  = torch.zeros(N, T + 1, device=device)
    VS = torch.zeros(N, T + 1, device=device)

    S[:, 0]  = S0
    v[:, 0]  = v0
    VS[:, 0] = v0 * S0 / sigma_v          # = 0.04 * 100 / 0.3 ≈ 13.33

    for t in range(T):
        z1 = torch.randn(N, device=device)
        z2 = rho * z1 + rho_bar * torch.randn(N, device=device)

        v[:, t + 1] = torch.clamp(
            v[:, t]
            + kappa * (theta - v[:, t]) * dt
            + sigma_v * torch.sqrt(v[:, t] * dt) * z2,
            min=1e-6,
        )

        S[:, t + 1] = S[:, t] * torch.exp(
            (r - 0.5 * v[:, t]) * dt
            + torch.sqrt(v[:, t] * dt) * z1
        )

        VS[:, t + 1] = v[:, t + 1] * S0 / sigma_v

    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, VS, payoff_fn
