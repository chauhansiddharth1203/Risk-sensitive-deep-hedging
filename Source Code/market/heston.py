import torch


def simulate_heston(
    N=64,
    T=30,
    S0=100.0,
    K=100.0,
    r=0.0,
    device="cpu",
):
    """
    Simulate Heston model paths and return payoff function
    """

    dt = 1.0 / T

    # Heston parameters
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.7
    v0 = 0.04

    # Precompute tensor constant (FIX)
    rho_bar = torch.sqrt(
        torch.tensor(1.0 - rho**2, device=device)
    )

    # Allocate
    S = torch.zeros(N, T + 1, device=device)
    v = torch.zeros(N, T + 1, device=device)

    S[:, 0] = S0
    v[:, 0] = v0

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
            (0.10 - 0.5 * v[:, t]) * dt
            + torch.sqrt(v[:, t] * dt) * z1
        )

    # ---- Payoff function ----
    def payoff_fn(S_T):
        return torch.clamp(S_T - K, min=0.0)

    return S, payoff_fn
