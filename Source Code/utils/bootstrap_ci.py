"""
utils/bootstrap_ci.py
---------------------
Bootstrap confidence intervals for tail-risk estimators (CVaR, VaR, mean).

All headline CVaR numbers in the thesis are sample estimates. A bootstrap
CI tells the examiner "this difference is robust to sampling noise."

Functions:
    cvar_point(pnl, alpha=0.95) -> float
    bootstrap_cvar_ci(pnl, alpha=0.95, B=1000, ci=0.95, seed=0)
        -> (point, low, high)
    bootstrap_diff_ci(pnl_a, pnl_b, alpha=0.95, B=1000, ci=0.95, seed=0)
        -> (mean_diff, low, high)  -- paired bootstrap on CVaR difference

Usage:
    from utils.bootstrap_ci import bootstrap_cvar_ci
    pnl = np.load("results/pnl.npy")
    cv, lo, hi = bootstrap_cvar_ci(pnl)
    print(f"CVaR95 = {cv:+.2f}   [95% CI: {lo:+.2f}, {hi:+.2f}]")
"""

import numpy as np


def cvar_point(pnl, alpha=0.95):
    """Sample CVaR at confidence alpha."""
    arr = np.asarray(pnl).ravel()
    k = max(int((1.0 - alpha) * len(arr)), 1)
    return float(np.sort(arr)[:k].mean())


def bootstrap_cvar_ci(pnl, alpha=0.95, B=1000, ci=0.95, seed=0):
    """
    Nonparametric bootstrap CI on CVaR_alpha.

    Args:
        pnl   : (N,) array of P&L samples
        alpha : CVaR confidence level (e.g. 0.95 -> worst 5%)
        B     : bootstrap replicates
        ci    : CI coverage (e.g. 0.95 -> 2.5th and 97.5th percentiles)
        seed  : RNG seed for reproducibility

    Returns:
        point : sample CVaR
        low   : lower CI endpoint
        high  : upper CI endpoint
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(pnl).ravel()
    N = len(arr)
    if N < 5:
        # too few for meaningful CI -- return degenerate interval
        pt = cvar_point(arr, alpha)
        return float(pt), float(pt), float(pt)

    stats = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        stats[b] = cvar_point(arr[idx], alpha)

    point = cvar_point(arr, alpha)
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    low, high = np.percentile(stats, [lo_q, hi_q])
    return float(point), float(low), float(high)


def bootstrap_diff_ci(pnl_a, pnl_b, alpha=0.95, B=1000, ci=0.95, seed=0):
    """
    Paired bootstrap CI on  CVaR(pnl_a) - CVaR(pnl_b).

    Use when pnl_a and pnl_b are evaluated on the SAME test paths
    (e.g. deep hedger vs baseline evaluated on common Monte Carlo).
    Pairing reduces variance dramatically.

    Returns (diff_point, low, high).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(pnl_a).ravel()
    b = np.asarray(pnl_b).ravel()
    if len(a) != len(b):
        raise ValueError("Paired bootstrap requires equal-length samples")
    N = len(a)

    diffs = np.empty(B)
    for rep in range(B):
        idx = rng.integers(0, N, size=N)
        diffs[rep] = cvar_point(a[idx], alpha) - cvar_point(b[idx], alpha)

    point = cvar_point(a, alpha) - cvar_point(b, alpha)
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    low, high = np.percentile(diffs, [lo_q, hi_q])
    return float(point), float(low), float(high)


def format_ci(point, low, high, digits=2):
    """Format a CVaR point-estimate + CI as a string."""
    return f"{point:+.{digits}f}  [{low:+.{digits}f}, {high:+.{digits}f}]"


if __name__ == "__main__":
    # sanity check
    rng = np.random.default_rng(0)
    pnl_a = rng.normal(-2.0, 5.0, 4000)
    pnl_b = rng.normal(-5.0, 6.0, 4000)

    p, l, h = bootstrap_cvar_ci(pnl_a, 0.95, B=500)
    print(f"CVaR95 (a): {format_ci(p, l, h)}")
    p, l, h = bootstrap_cvar_ci(pnl_b, 0.95, B=500)
    print(f"CVaR95 (b): {format_ci(p, l, h)}")
    p, l, h = bootstrap_diff_ci(pnl_a, pnl_b, 0.95, B=500)
    print(f"Diff a-b  : {format_ci(p, l, h)}")
