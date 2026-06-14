"""
baselines/heston_delta_vega_hedge.py
--------------------------------------
Analytical Heston delta-vega hedge.

At each step t, computes the theoretically correct hedge using the
current instantaneous variance v_t:

    delta_S = BS_delta(S_t, sigma=sqrt(v_t), tau=remaining_time)
    delta_V = BS_vega(S_t, sigma=sqrt(v_t), tau) * sigma_v / (S0 * 2 * sqrt(v_t))

The VS position formula comes from:
    We want:  delta_V * dVS = (dC/dv) * dv
    Since VS = v * S0/sigma_v  =>  dVS/dv = S0/sigma_v
    And dC/dv = BS_vega * d(sigma)/dv = BS_vega / (2*sqrt(v))
    => delta_V = BS_vega * sigma_v / (S0 * 2 * sqrt(v))

This is the "instantaneous Heston hedge" -- it uses the exact current variance
to compute hedge ratios, which is information the BS delta hedge ignores.
It represents the best a model-aware analytical hedger can do at each step.

VS values can be recovered from the simulator output:
    v_t = VS_t * sigma_v / S0  (inverse of the scaling formula)
"""

import torch
import numpy as np
from scipy.stats import norm


# Heston parameters (must match simulator)
SIGMA_V = 0.3
S0      = 100.0
K       = 100.0


def _bs_delta(S, sigma, tau):
    """Black-Scholes delta with given instantaneous vol."""
    if tau < 1e-6:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def _bs_vega(S, sigma, tau):
    """Black-Scholes vega (∂C/∂sigma)."""
    if tau < 1e-6:
        return 0.0
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)


@torch.no_grad()
def heston_delta_vega_pnl(
    S,
    VS,
    payoff_fn,
    K_strike=100.0,
    sigma_v=SIGMA_V,
    S0=S0,
    cost_rate=0.0002,
):
    """
    Run the analytical Heston delta-vega hedge on N paths.

    Parameters
    ----------
    S         : (N, T+1) stock paths
    VS        : (N, T+1) variance swap paths  (= v_t * S0 / sigma_v)
    payoff_fn : callable
    K_strike  : option strike
    sigma_v   : Heston vol-of-vol (for VS inverse scaling)
    S0        : initial stock price
    cost_rate : proportional transaction cost

    Returns
    -------
    pnl : (N,) tensor of final P&L
    """
    global K
    K = K_strike

    N  = S.shape[0]
    Tt = S.shape[1] - 1
    device = S.device

    pnl          = torch.zeros(N, device=device)
    prev_delta_S = torch.zeros(N, device=device)
    prev_delta_V = torch.zeros(N, device=device)

    for t in range(Tt):
        tau = max((Tt - t) / Tt, 1e-6)

        # Recover instantaneous variance from VS
        v_t = VS[:, t] * sigma_v / S0    # (N,)

        # Compute hedge ratios analytically for each path
        delta_S_np = np.array([
            _bs_delta(S[i, t].item(), max(float(v_t[i].item()) ** 0.5, 1e-4), tau)
            for i in range(N)
        ])
        delta_V_np = np.array([
            _bs_vega(S[i, t].item(), max(float(v_t[i].item()) ** 0.5, 1e-4), tau)
            * sigma_v / (S0 * 2.0 * max(float(v_t[i].item()) ** 0.5, 1e-4))
            for i in range(N)
        ])

        delta_S = torch.tensor(delta_S_np, dtype=torch.float32, device=device)
        delta_V = torch.tensor(delta_V_np, dtype=torch.float32, device=device)

        # P&L from previous positions
        pnl += prev_delta_S * (S[:, t + 1] - S[:, t])
        pnl += prev_delta_V * (VS[:, t + 1] - VS[:, t])

        # Transaction costs
        pnl -= cost_rate * torch.abs(delta_S - prev_delta_S) * (S[:, t] / S0)
        pnl -= cost_rate * torch.abs(delta_V - prev_delta_V) * (VS[:, t] / (S0 * 0.04 / sigma_v))

        prev_delta_S = delta_S
        prev_delta_V = delta_V

    pnl -= payoff_fn(S[:, -1])
    return pnl
