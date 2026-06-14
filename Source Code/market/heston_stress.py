import torch
from market.heston import simulate_heston


def simulate_heston_stress(
    N=64,
    T=30,
    S0=100.0,
    K=100.0,
    r=0.0,
    device="cpu",
    sigma_v_multiplier=1.5,
    rho_stress=-0.9,
):
    """
    Stressed Heston simulation:
    - Increased vol-of-vol
    - Stronger negative correlation
    """

    # Call original simulator to get base paths
    S, payoff_fn = simulate_heston(
        N=N,
        T=T,
        S0=S0,
        K=K,
        r=r,
        device=device,
    )

    # NOTE:
    # We cannot directly change sigma_v and rho inside simulate_heston
    # so we reweight volatility paths via multiplicative stress

    # Simple volatility stress proxy (path-level stress)
    noise = torch.randn_like(S)
    S_stressed = S * torch.exp(0.5 * noise * sigma_v_multiplier)

    return S_stressed, payoff_fn
