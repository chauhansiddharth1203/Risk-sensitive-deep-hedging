"""
calibration/heston_pricer.py
-----------------------------
Semi-analytical Heston option pricer via Gil-Pelaez Fourier inversion.

Reference: Heston (1993), Gatheral (2006) "The Volatility Surface"

Call price:
    C = S0 * P1 - K * exp(-r*T) * P2

where P1, P2 are computed by integrating the characteristic function.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq


def heston_char_func(u, S0, v0, kappa, theta, sigma_v, rho, r, T):
    """
    Heston characteristic function (Heston 1993 formulation).
    u : complex argument
    Returns the characteristic function value (complex).
    """
    i   = 1j
    lnu = np.log(S0)

    xi  = kappa - sigma_v * rho * i * u
    d   = np.sqrt(xi ** 2 + sigma_v ** 2 * (u ** 2 + i * u))

    # Avoid branch-cut issues
    g = (xi - d) / (xi + d)

    exp_dT = np.exp(-d * T)

    C = (r * i * u * T
         + (kappa * theta / sigma_v ** 2) * (
             (xi - d) * T
             - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
         ))
    D = ((xi - d) / sigma_v ** 2
         * (1.0 - exp_dT) / (1.0 - g * exp_dT))

    return np.exp(C + D * v0 + i * u * lnu)


def heston_call_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho,
                      n_quad=200, quad_limit=200):
    """
    Heston call price via Gil-Pelaez inversion.

    Parameters
    ----------
    S0, K, T, r  : float  spot / strike / maturity / risk-free rate
    v0           : float  initial variance
    kappa        : float  mean-reversion speed
    theta        : float  long-run variance
    sigma_v      : float  vol-of-vol
    rho          : float  correlation in (-1, 1)

    Returns
    -------
    price : float  European call price
    """
    if T <= 0:
        return max(S0 - K, 0.0)

    log_K = np.log(K)

    def integrand_P1(u):
        phi = heston_char_func(u - 1j, S0, v0, kappa, theta, sigma_v, rho, r, T)
        phi0 = heston_char_func(-1j,   S0, v0, kappa, theta, sigma_v, rho, r, T)
        val = np.exp(-1j * u * log_K) * phi / (1j * u * phi0)
        return np.real(val)

    def integrand_P2(u):
        phi = heston_char_func(u, S0, v0, kappa, theta, sigma_v, rho, r, T)
        val = np.exp(-1j * u * log_K) * phi / (1j * u)
        return np.real(val)

    P1, _ = quad(integrand_P1, 1e-8, n_quad, limit=quad_limit)
    P2, _ = quad(integrand_P2, 1e-8, n_quad, limit=quad_limit)

    P1 = 0.5 + P1 / np.pi
    P2 = 0.5 + P2 / np.pi

    return S0 * P1 - K * np.exp(-r * T) * P2


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r, tol=1e-6):
    """
    Black-Scholes implied volatility via Brent's method.
    Returns np.nan if no solution found in [1e-4, 5.0].
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + tol:
        return np.nan

    try:
        iv = brentq(
            lambda sigma: bs_call_price(S, K, T, r, sigma) - price,
            1e-4, 5.0, xtol=tol, maxiter=100,
        )
        return iv
    except Exception:
        return np.nan


def heston_implied_vol(S0, K, T, r, v0, kappa, theta, sigma_v, rho):
    """Heston call price expressed as Black-Scholes implied vol."""
    price = heston_call_price(S0, K, T, r, v0, kappa, theta, sigma_v, rho)
    return implied_vol(price, S0, K, T, r)
