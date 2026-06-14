import torch
from market.heston import simulate_heston


def simulate_heston_calibrated(
    regime="normal",
    N=64,
    T=30,
    S0=100.0,
    K=100.0,
    r=0.0,
    device="cpu",
):
    """
    Regime-based calibrated Heston simulation.
    Uses volatility scaling to reflect market regimes.
    """

    regime_scale = {
        "low": 0.7,
        "normal": 1.0,
        "stress": 1.8,
    }

    if regime not in regime_scale:
        raise ValueError(f"Unknown regime: {regime}")

    scale = regime_scale[regime]

    S, payoff_fn = simulate_heston(
        N=N,
        T=T,
        S0=S0,
        K=K,
        r=r,
        device=device,
    )

    # Apply volatility scaling at path level
    noise = torch.randn_like(S)
    S_calibrated = S * torch.exp(0.5 * scale * noise)

    return S_calibrated, payoff_fn
